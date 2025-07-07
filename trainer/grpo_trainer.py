# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from datasets import load_dataset

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

# --- metrics.py -----------------------------------------------------------
from evaluate import load as load_metric
import sacrebleu, numpy as np, itertools as it

rouge = load_metric("rouge")      # supports rouge1/2/L/Lsum
def rouge_l(preds, refs):
    out = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return out["rougeL"]          # F1 by default

def self_bleu(preds):
    scores = []
    for i, hyp in enumerate(preds):
        refs = preds[:i] + preds[i+1:]
        scores.append(sacrebleu.corpus_bleu([hyp], [refs]).score)
    return float(np.mean(scores))


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        if not (0.0 <= args.explore_beta <= 10.0):
            raise ValueError("--explore_beta should be in [0,10]")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions
        
        self.val_data = load_dataset("trl-lib/tldr", split="validation[:2%]")
        # self.eval_every = args.logging_steps * 10
        self.eval_every = args.logging_steps * 50

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # --- after defining self.reward_funcs ---
        if args.intrinsic_reward_type in ["epistemic", "both"]:
            self.bald_ema_mean = torch.tensor(0.0)
            self.bald_ema_var = torch.tensor(1.0)
            self.bald_ema_steps = torch.tensor(0)  # Step counter for bias correction
            self.ema_beta = 0.99  # decay rate

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        if args.per_device_train_batch_size < self.num_generations:
            raise ValueError(
                f"per_device_train_batch_size ({args.per_device_train_batch_size}) must be >= num_generations ({self.num_generations}). "
                f"Each batch needs at least num_generations samples for proper reshaping during training."
            )
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {global_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). "
                f"Additionally, per_device_train_batch_size ({args.per_device_train_batch_size}) must be >= num_generations ({self.num_generations}). "
                f"Given the current train batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            if args.per_device_eval_batch_size < self.num_generations:
                raise ValueError(
                    f"per_device_eval_batch_size ({args.per_device_eval_batch_size}) must be >= num_generations ({self.num_generations}). "
                    f"Each batch needs at least num_generations samples for proper reshaping during evaluation."
                )
            global_batch_size = args.per_device_eval_batch_size * args.gradient_accumulation_steps * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Multi-step
        self.num_iterations = getattr(args, 'num_iterations', 1)  # = 𝜇 in the GRPO paper
        self.epsilon_low = getattr(args, 'epsilon', 0.2)
        self.epsilon_high = getattr(args, 'epsilon_high', self.epsilon_low)
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Additional attributes for compatibility
        self.scale_rewards = getattr(args, 'scale_rewards', True)
        self.token_entropy_percentile_threshold = getattr(args, 'token_entropy_percentile_threshold', 0.0)
        self.loss_type = getattr(args, 'loss_type', 'grpo')
        
        # Set steps_per_generation as an attribute on args if not present
        if not hasattr(args, 'steps_per_generation'):
            args.steps_per_generation = 1

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                # profiling_patch = patch(
                #     "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                # )
                with world_size_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                # with world_size_patch, profiling_patch:
                #     self.llm = LLM(
                #         model=model.name_or_path,
                #         device=vllm_device,
                #         gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                #         dtype=self.args.vllm_dtype,
                #         # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                #         # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                #         # This is particularly useful here because we generate completions from the same prompts.
                #         enable_prefix_caching=True,
                #         max_model_len=self.args.vllm_max_model_len,
                #     )
                
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log‐probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            temperature = getattr(self, 'temperature', 1.0)
            logits = logits / temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                # Compute entropy from logits: H = -sum(p * log(p))
                probs = torch.softmax(logits, dim=-1)
                entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return {"logps": logps, "entropies": entropies}

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            # Add missing attributes with defaults
            steps_per_generation = getattr(self.args, 'steps_per_generation', 1)
            num_iterations = getattr(self, 'num_iterations', 1)
            
            generate_every = steps_per_generation * num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(inputs)
                # Add missing shuffle function
                generation_batch = self.shuffle_tensor_dict(generation_batch)
                self._buffered_inputs = self.split_tensor_dict(generation_batch, steps_per_generation)
            inputs = self._buffered_inputs[self._step % steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def shuffle_tensor_dict(self, tensor_dict: dict[str, Optional[torch.Tensor]]) -> dict[str, Optional[torch.Tensor]]:
        """Shuffles a dictionary of tensors along the first dimension in unison."""
        first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
        batch_size = first_tensor.shape[0]
        permutation = torch.randperm(batch_size)
        return {key: tensor[permutation] if tensor is not None else None for key, tensor in tensor_dict.items()}

    def split_tensor_dict(
        self, tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
    ) -> list[dict[str, Optional[torch.Tensor]]]:
        """Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts."""
        first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
        chunk_size = first_tensor.shape[0] // num_chunks
        return [
            {
                key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
                for key, tensor in tensor_dict.items()
            }
            for i in range(num_chunks)
        ]

    def _generate_and_score_completions(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> dict[str, Union[torch.Tensor, Any]]:
        prev_training_mode = self.model.training   
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
        # deterministic generation → disable dropout
        self.model.eval()

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            # Compute old_per_token_logps for the probability ratio in the loss
            # This represents π_old(a|s) - the policy that generated the completions
            steps_per_generation = getattr(self.args, 'steps_per_generation', 1)
            num_iterations = getattr(self, 'num_iterations', 1)
            gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
            
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if num_iterations > 1 or steps_per_generation > gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None
            
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel
                # Ensure reward model is in eval mode
                reward_func.eval()
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        # ── 1. extrinsic reward exactly as before ─────────────────────────────
        R_e = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        # ── 2. intrinsic BALD  (only if enabled) ──────────────────────────────
        # NOTE: BALD computation will handle its own dropout management internally
        if self.args.intrinsic_reward_type in ["epistemic", "both"]:
            self._metrics["debug_intrinsic_type"].append(str(self.args.intrinsic_reward_type))
            self._metrics["debug_explore_beta"].append(getattr(self.args, 'explore_beta', -999))
            self._metrics["debug_epi_reward_alpha"].append(getattr(self.args, 'epi_reward_alpha', -999))
            self._metrics["debug_epi_reward_mode"].append(str(getattr(self.args, 'epi_reward_mode', 'NOT_SET')))
            self._metrics["debug_epi_reward_num_samples"].append(getattr(self.args, 'epi_reward_num_samples', -999))
            self._metrics["debug_computing_bald_samples"].append(len(prompt_ids))
            
            # BALD computation will temporarily enable dropout internally
            bald_raw = self.compute_intrinsic_reward(prompt_ids, completion_ids, attention_mask)
            self._metrics["debug_bald_raw_mean"].append(bald_raw.mean().item())
            self._metrics["debug_bald_raw_std"].append(bald_raw.std().item())
            self._metrics["debug_bald_raw_shape_0"].append(bald_raw.shape[0])
            
            # Gather BALD for EMA computation across all processes
            bald_raw_gathered = gather(bald_raw)
            self._metrics["debug_bald_after_gather_mean"].append(bald_raw_gathered.mean().item())
            self._metrics["debug_bald_after_gather_std"].append(bald_raw_gathered.std().item())
            self._metrics["debug_bald_after_gather_shape_0"].append(bald_raw_gathered.shape[0])

            # ---- EMA normalisation (mean & variance) -------------------------
            # Compute local statistics on gathered BALD for consistent EMA across processes
            local_mean = bald_raw_gathered.mean()
            local_var = bald_raw_gathered.var(unbiased=False)
            self._metrics["debug_local_mean"].append(local_mean.item())
            self._metrics["debug_local_var"].append(local_var.item())
            
            # Ensure EMA tensors are on the correct device
            if self.bald_ema_mean.device != device:
                self.bald_ema_mean = self.bald_ema_mean.to(device)
                self.bald_ema_var = self.bald_ema_var.to(device)
                self.bald_ema_steps = self.bald_ema_steps.to(device)
            
            # No need for distributed synchronization since we already gathered
            # Increment step counter
            self.bald_ema_steps += 1
            
            # Update EMA with gathered statistics
            self.bald_ema_mean = self.ema_beta * self.bald_ema_mean + (1 - self.ema_beta) * local_mean
            self.bald_ema_var = self.ema_beta * self.bald_ema_var + (1 - self.ema_beta) * local_var
            
            # Apply bias correction
            bias_correction = 1 - self.ema_beta ** self.bald_ema_steps
            corrected_mean = self.bald_ema_mean / bias_correction
            corrected_var = self.bald_ema_var / bias_correction
            
            self._metrics["debug_ema_raw_mean"].append(self.bald_ema_mean.item())
            self._metrics["debug_ema_raw_var"].append(self.bald_ema_var.item())
            self._metrics["debug_corrected_mean"].append(corrected_mean.item())
            self._metrics["debug_corrected_var"].append(corrected_var.item())
            self._metrics["debug_ema_steps"].append(self.bald_ema_steps.item())
            
            # Normalize using bias-corrected EMA - apply to LOCAL bald_raw only
            bald_z = (bald_raw - corrected_mean) / (corrected_var.sqrt() + 1e-4)
            self._metrics["debug_bald_z_mean"].append(bald_z.mean().item())
            self._metrics["debug_bald_z_std"].append(bald_z.std().item())
            self._metrics["debug_bald_z_min"].append(bald_z.min().item())
            self._metrics["debug_bald_z_max"].append(bald_z.max().item())

            bald_z = torch.clamp(bald_z, -5, 5)
            # ---- keep only above-average MI ----------------------------------
            bald_pos = torch.relu(bald_z).tanh()  # negatives → 0
            self._metrics["debug_bald_pos_mean"].append(bald_pos.mean().item())
            self._metrics["debug_bald_pos_std"].append(bald_pos.std().item())
            self._metrics["debug_bald_pos_nonzero"].append((bald_pos > 0).sum().item())
            self._metrics["debug_bald_pos_total"].append(len(bald_pos))

            R_i = self.args.explore_beta * bald_pos
            self._metrics["debug_R_i_mean"].append(R_i.mean().item())
            self._metrics["debug_R_i_std"].append(R_i.std().item())
            self._metrics["debug_final_explore_beta"].append(self.args.explore_beta)
        else:
            R_i = torch.zeros_like(R_e, device=device)
            self._metrics["debug_intrinsic_disabled"].append(1.0)

        # ── 3. additive mix ──────────────────────────────────────────────────
        total_reward = R_e + R_i

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func_gathered = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards_gathered = (rewards_per_func_gathered * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards_gathered.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards_gathered.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards_gathered - mean_grouped_rewards
        if getattr(self, 'scale_rewards', True):
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics using gathered data for proper global statistics
        reward_per_func = rewards_per_func_gathered.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        # CRITICAL FIX: Log the gathered rewards for proper global statistics, 
        # but use LOCAL advantages for training
        self._metrics["reward"].append(rewards_gathered.mean().item())
        self._metrics["reward_std"].append(rewards_gathered.std().item())
        
        # Debug: Log the difference between local and gathered statistics to monitor the fix
        self._metrics["debug_local_reward_mean"].append(total_reward.mean().item())
        self._metrics["debug_local_reward_std"].append(total_reward.std().item())
        self._metrics["debug_gathered_reward_mean"].append(rewards_gathered.mean().item())
        self._metrics["debug_gathered_reward_std"].append(rewards_gathered.std().item())
        self._metrics["debug_local_batch_size"].append(len(total_reward))
        self._metrics["debug_gathered_batch_size"].append(len(rewards_gathered))
        
        if self.args.intrinsic_reward_type in ["epistemic", "both"]:
            self._metrics["bald_raw"].append(bald_raw_gathered.mean().item())
        else:
            self._metrics["bald_raw"].append(0.0)
        
        # Log local intrinsic reward (since R_i is computed locally)
        R_i_gathered = gather(R_i)
        self._metrics["intrinsic_reward"].append(R_i_gathered.mean().item())

        # ── 6. Wandb logging (using gathered data for completions table) ──────────────────────────────
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging - use gathered data for global view
            table = {
                "step": [str(self.state.global_step)] * len(rewards_gathered),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards_gathered.tolist(),
                "bald_raw": bald_raw_gathered.tolist() if self.args.intrinsic_reward_type in ["epistemic", "both"] else [0.0] * len(rewards_gathered),
                "intrinsic_reward": R_i_gathered.tolist() if self.args.intrinsic_reward_type in ["epistemic", "both"] else [0.0] * len(rewards_gathered),
            }
            
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        # CRITICAL: Restore the original training mode for the main model
        # This ensures consistency between _prepare_inputs and compute_loss
        self.model.train(prev_training_mode)
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
        }

    def compute_intrinsic_reward(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        """
        Epistemic bonus ≈ BALD =  H[ 1/M Σ p_m ]  –  1/M Σ H[p_m]
        where the entropies are **summed over all selected tokens**.
        Returns a 1-D tensor of shape (B,) with the un-scaled BALD score.
        """
        device = prompt_ids.device
        B       = prompt_ids.size(0)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        prompt_len = prompt_ids.size(1)
        comp_len   = completion_ids.size(1)

        # Log BALD computation details
        self._metrics["debug_bald_batch_size"].append(B)
        self._metrics["debug_bald_prompt_len"].append(prompt_len)
        self._metrics["debug_bald_comp_len"].append(comp_len)

        # ---- choose which token positions to include -------------------------
        mode = getattr(self.args, 'epi_reward_mode', 'all')  # Default to 'all' if not set
        if mode == "eos":
            idx = [prompt_len + comp_len - 1]
        elif mode == "n_token":
            n = getattr(self.args, 'epi_reward_n', 5)  # Default to 5 if not set
            idx = list(range(prompt_len + comp_len - n, prompt_len + comp_len))
        elif mode == "all":
            idx = list(range(prompt_len, prompt_len + comp_len))
        else:
            print(f"WARNING: Unknown epi_reward_mode: {mode}, defaulting to 'all'")
            idx = list(range(prompt_len, prompt_len + comp_len))

        self._metrics["debug_bald_mode"].append(str(mode))
        self._metrics["debug_bald_selected_tokens"].append(len(idx))

        M = max(2, getattr(self.args, 'epi_reward_num_samples', 10))   # Default to 10 if not set
        eps = 1e-12                                     # numerical stability
        entropies_per_pass = []
        probs_per_pass      = []

        self._metrics["debug_bald_mc_passes"].append(M)

        prev_mode = self.model.training
        self.model.train()     # switch on dropout
        
        # Ensure dropout is enabled on the actual model doing forward passes
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        prev_unwrapped_mode = unwrapped_model.training
        unwrapped_model.train()
        
        # Debug: Check if model has dropout and if it's enabled
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in unwrapped_model.modules())
        self._metrics["debug_bald_has_dropout"].append(1.0 if has_dropout else 0.0)
        self._metrics["debug_bald_model_training"].append(1.0 if self.model.training else 0.0)
        
        # CRITICAL FIX: Force dropout even if model doesn't have it natively
        # This is essential for BALD to work with models like Qwen that don't have dropout
        dropout_prob = 0.1  # Standard dropout probability
        
        if not has_dropout:
            print(f"WARNING: No dropout modules found in model! Forcing manual dropout with p={dropout_prob}")
            # We'll apply manual dropout to the logits to create variance

        variance_checks = []
        with torch.no_grad():
            for i in range(M):
                if has_dropout:
                    # Model has dropout, use it normally
                    logits = self.model(full_ids, attention_mask=attention_mask).logits
                else:
                    # Model has no dropout, apply manual dropout to logits
                    with torch.enable_grad():  # Enable grad temporarily for dropout
                        logits = self.model(full_ids, attention_mask=attention_mask).logits
                        # Apply dropout to logits to create variance between passes
                        logits = torch.nn.functional.dropout(logits, p=dropout_prob, training=True)
                
                logits_sel = logits[:, idx, :]                          # (B, L', V)
                probs      = torch.softmax(logits_sel, dim=-1)          # (B, L', V)
                H_tokens   = -(probs * (probs + eps).log()).sum(-1)      # (B, L')
                entropies_per_pass.append(H_tokens.sum(-1))             # ← SUM over tokens -> (B,)
                probs_per_pass.append(probs.unsqueeze(0))               # keep for later
                
                # Debug: Log variance between passes
                if i == 1 and len(probs_per_pass) >= 2:
                    var_check = (probs_per_pass[0] - probs_per_pass[1]).abs().mean().item()
                    variance_checks.append(var_check)
                    self._metrics["debug_bald_variance_01"].append(var_check)
                elif i > 1:
                    var_check = (probs_per_pass[0] - probs_per_pass[i]).abs().mean().item()
                    variance_checks.append(var_check)

        avg_variance = sum(variance_checks) / len(variance_checks) if variance_checks else 0.0
        self._metrics["debug_bald_avg_variance"].append(avg_variance)
        if avg_variance < 1e-6:
            print(f"WARNING: Very low variance between MC passes ({avg_variance:.2e})! BALD may not work properly.")
        else:
            print(f"BALD variance check: {avg_variance:.6f} (good if > 1e-6)")

        #   E_p[H]  term -------------------------------------------------------
        H_bar = torch.stack(entropies_per_pass, dim=0).mean(0)          # (B,)
        self._metrics["debug_bald_H_bar_mean"].append(H_bar.mean().item())
        self._metrics["debug_bald_H_bar_std"].append(H_bar.std().item())

        #   H[E_p]  term -------------------------------------------------------
        probs_stack = torch.cat(probs_per_pass, dim=0)                  # (M, B, L', V)
        probs_avg   = probs_stack.mean(0)                               # (B, L', V)
        H_of_avg    = -(probs_avg * (probs_avg + eps).log()).sum(-1)    # (B, L')
        H_of_avg    = H_of_avg.sum(-1)                                  # ← SUM over tokens -> (B,)
        self._metrics["debug_bald_H_of_avg_mean"].append(H_of_avg.mean().item())
        self._metrics["debug_bald_H_of_avg_std"].append(H_of_avg.std().item())

        self.model.train(prev_mode)      # restore original mode
        unwrapped_model.train(prev_unwrapped_mode)  # restore unwrapped model mode

        bald = H_of_avg - H_bar          # (B,)
        self._metrics["debug_bald_final_mean"].append(bald.mean().item())
        self._metrics["debug_bald_final_std"].append(bald.std().item())
        self._metrics["debug_bald_final_min"].append(bald.min().item())
        self._metrics["debug_bald_final_max"].append(bald.max().item())
        
        # Debug: Log BALD computation results (keep this print for immediate feedback)
        if torch.distributed.get_rank() == 0:  # Only log on main process
            print(f"DEBUG BALD FINAL: H_of_avg={H_of_avg.mean().item():.4f}, H_bar={H_bar.mean().item():.4f}, bald_mean={bald.mean().item():.4f}, bald_max={bald.max().item():.4f}")
        
        epi_reward_alpha = getattr(self.args, 'epi_reward_alpha', 1.0)  # Default to 1.0 if not set
        self._metrics["debug_bald_epi_reward_alpha"].append(epi_reward_alpha)
        
        result = (epi_reward_alpha * bald).detach()
        self._metrics["debug_bald_result_mean"].append(result.mean().item())
        self._metrics["debug_bald_result_std"].append(result.std().item())
        return result

    def compute_aleatoric_reward(self, prompt_ids: torch.Tensor, completion_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute an aleatoric intrinsic reward for each sample using a single forward pass.
        
        This function uses the output from a deterministic forward pass (dropout off) and computes the entropy
        for the selected token positions. The token selection is determined by self.args.epi_reward_mode,
        similarly to compute_intrinsic_reward().
        
        Returns:
            A tensor of shape (B,) with the average entropy for the selected tokens.
        """
        device = prompt_ids.device
        B = prompt_ids.size(0)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        total_len = full_ids.size(1)
        prompt_length = prompt_ids.size(1)
        comp_length = completion_ids.size(1)
        
        if self.args.epi_reward_mode == "eos":
            indices = [prompt_length + comp_length - 1]
        elif self.args.epi_reward_mode == "n_token":
            n = self.args.epi_reward_n
            indices = list(range(prompt_length + comp_length - n, prompt_length + comp_length))
        elif self.args.epi_reward_mode == "all":
            indices = list(range(prompt_length, prompt_length + comp_length))
        else:
            raise ValueError(f"Unknown epi_reward_mode: {self.args.epi_reward_mode}")
        
        model = self.model
        # Ensure dropout is off (evaluation mode)
        prev_mode = model.training
        model.eval()
        with torch.no_grad():
            outputs = model(full_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, total_len, vocab)
            target_logits = logits[:, indices, :]  # (B, len(indices), vocab)
            probs = torch.softmax(target_logits, dim=-1)  # (B, len(indices), vocab)
            entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (B, len(indices))
            avg_entropy = entropies.mean(dim=-1)  # (B,)
        model.train(prev_mode)
        return avg_entropy

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Set default values for missing attributes to match original GRPO
        token_entropy_percentile_threshold = getattr(self, 'token_entropy_percentile_threshold', 0.0)
        beta = getattr(self, 'beta', 0.1)
        epsilon_low = getattr(self, 'epsilon_low', 0.2)
        epsilon_high = getattr(self, 'epsilon_high', 0.2)
        loss_type = getattr(self, 'loss_type', 'grpo')

        # Compute the entropy at each position in the completion
        if token_entropy_percentile_threshold > 0.0:
            logps_and_entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True
            )
            per_token_logps = logps_and_entropies["logps"]
            entropies = logps_and_entropies["entropies"]
            # compute the entropy threshold across all tokens in the batch
            entropy_threshold = torch.quantile(entropies.flatten().float(), token_entropy_percentile_threshold)
            entropy_mask = entropies >= entropy_threshold
        else:
            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        # Two-sided clipping
        if hasattr(self.args, 'delta') and self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if beta != 0.0:
            per_token_loss = per_token_loss + beta * per_token_kl

        if loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # Log the metrics - handle gather_for_metrics return type gracefully
        def safe_gather_mean(tensor):
            gathered = self.accelerator.gather_for_metrics(tensor)
            if hasattr(gathered, 'float') and hasattr(gathered, 'mean'):
                return gathered.float().mean().item()
            elif hasattr(gathered, '__iter__') and not isinstance(gathered, str):
                return sum(gathered) / len(gathered) if gathered else 0.0
            else:
                return float(gathered) if gathered else 0.0

        completion_length = safe_gather_mean(completion_mask.sum(1))
        self._metrics["completion_length"].append(completion_length)

        if beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            kl_value = safe_gather_mean(mean_kl)
            self._metrics["kl"].append(kl_value)

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        self._metrics["clip_fraction"].append(safe_gather_mean(clip_ratio))
        self._metrics["ratio_mean"].append(safe_gather_mean(coef_1.mean()))

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
            
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Filter out non-numeric metrics and average only numeric ones
        metrics = {}
        for key, val in self._metrics.items():
            if val and isinstance(val[0], (int, float)):
                metrics[key] = sum(val) / len(val)
            elif val and isinstance(val[0], str):
                # For string metrics, just take the last value
                metrics[key] = val[-1]

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        
        if self.state.global_step % self.eval_every == 0 and self.accelerator.is_main_process:
            self._offline_eval()
            
        self._metrics.clear()
    
    def _offline_eval(self, max_batches=40):
        prev_mode = self.model.training  
        self.model.eval()

        raw_model  = self.accelerator.unwrap_model(self.model)   # returns the underlying AutoModel
        device     = self.accelerator.device

        preds, refs = [], []
        if not hasattr(self, "_val_subset"):
            self._val_subset = self.val_data.select(range(max_batches))
        for ex in self._val_subset:
            prompt = ex["content"];  ref = ex["summary"]
            ids = self.processing_class(prompt, return_tensors="pt").to(device)["input_ids"]

            # --- use raw_model instead of self.model ---
            gen_ids = raw_model.generate(ids,
                                        max_new_tokens=self.max_completion_length,
                                        temperature=0.7)
            gen_txt = self.processing_class.decode(gen_ids[0], skip_special_tokens=True)
            preds.append(gen_txt); refs.append(ref)
        rl   = rouge_l(preds, refs)      # ROUGE-L F1
        sbleu = self_bleu(preds)         # self-BLEU (lower is better)

        wandb.log({"eval/rougeL": rl, "eval/selfBLEU": sbleu,
                "global_step": self.state.global_step})
        self.model.train(prev_mode)      # restore dropout state

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        from datasets import Dataset as HFDataset
        if isinstance(train_dataset, HFDataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        steps_per_generation = getattr(self.args, 'steps_per_generation', 1)
        
        dataloader_params = {
            "batch_size": self._train_batch_size * steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = self.seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        from torch.utils.data import DataLoader
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def seed_worker(self, worker_id):
        """Simple seed worker function"""
        import random
        import numpy as np
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)