# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import logging
import math
from typing import List, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
    TRAINER_STATE_NAME,
)
try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import seed_worker

from transformers.trainer import *


def _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
    warmup_lr_rate: Optional[float] = None,
):
    current_step = float(current_step)
    num_warmup_steps = float(num_warmup_steps)
    num_training_steps = float(num_training_steps)

    if current_step < num_warmup_steps:
        if warmup_lr_rate is None:
            return (current_step + 1.0) / max(1.0, num_warmup_steps)
        else:
            warmup_lr_rate = float(warmup_lr_rate)
            return warmup_lr_rate + (1.0 - warmup_lr_rate) * (current_step) / (max(1, num_warmup_steps - 1))
    progress = (current_step - num_warmup_steps + 1.0) / (max(1.0, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: Optional[float] = None,
    min_lr_rate: Optional[float] = None,
    warmup_lr_rate: Optional[float] = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.
        warmup_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the start learning rate. If not set, `warmup_lr_rate` will be treated as float(1/num_warmup_steps).

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        warmup_lr_rate=warmup_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['vision_projector', 'vision_encoder']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_sorted_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    indices = lengths.tolist()
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_sorted_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class ReconstructionLossLoggingCallback(TrainerCallback):
    """Accumulate reconstruction loss and merge into the main log entry (same line as loss)."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        trainer = self.trainer
        if not hasattr(trainer, "sum_image_loss") or trainer.loss_count == 0:
            return
        image_loss = trainer.sum_image_loss / trainer.loss_count
        teacher_loss = trainer.sum_teacher_loss / trainer.loss_count
        relation_loss = trainer.sum_relation_loss / trainer.loss_count
        caption_loss = trainer.sum_caption_loss / trainer.loss_count
        logs["image_loss"] = image_loss
        logs["teacher_loss"] = teacher_loss
        logs["relation_loss"] = relation_loss
        logs["caption_loss"] = caption_loss
        trainer.sum_image_loss = 0
        trainer.sum_teacher_loss = 0
        trainer.sum_relation_loss = 0
        trainer.sum_caption_loss = 0
        trainer.loss_count = 0


class PenguinVLTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_image_loss = 0
        self.sum_teacher_loss = 0
        self.sum_relation_loss = 0
        self.sum_caption_loss = 0
        self.loss_count = 0
        self.callback_handler.callbacks.insert(0, ReconstructionLossLoggingCallback(self))

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            try:
                # transformers 4.55.2
                return super()._get_train_sampler(train_dataset)
            except TypeError:
                # transformers 4.51.3
                return super()._get_train_sampler()

    def get_batch_samples(self, epoch_iterator, num_batches, *args):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0:
            if self.args.loss_reduction_scope == "batch":
                assert "labels" in batch_samples[0]
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            elif self.args.loss_reduction_scope == "sequence":
                assert "position_ids" in batch_samples[0]
                num_items_in_batch = sum([(batch["position_ids"] == 0).sum() for batch in batch_samples])
            else:
                raise ValueError(f"Unknown reduction scope: {self.args.loss_reduction_scope}")

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch.cuda()).sum().item()

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            optimized_parameters = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            optimizer_grouped_parameters = []

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.llm_lr is not None:
                lm_parameters = [
                    name for name, _ in optimized_parameters if "vision_encoder" not in name and "vision_projector" not in name
                ]
                decay_lm_parameters = [name for name in lm_parameters if name in decay_parameters]
                nodecay_lm_parameters = [name for name in lm_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_lm_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.llm_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_lm_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.llm_lr,
                    }
                ])

            if self.args.vision_projector_lr is not None:
                vision_projector_parameters = [
                    name for name, _ in optimized_parameters if "vision_projector" in name
                ]
                decay_vision_projector_parameters = [name for name in vision_projector_parameters if name in decay_parameters]
                nodecay_vision_projector_parameters = [name for name in vision_projector_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_vision_projector_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_projector_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_vision_projector_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_projector_lr,
                    }
                ])

            if self.args.vision_encoder_lr is not None:
                if self.args.embedding_lr is not None:
                    vision_encoder_parameters = [name for name, _ in optimized_parameters if "vision_encoder" in name and "embeddings" not in name]
                else:
                    vision_encoder_parameters = [name for name, _ in optimized_parameters if "vision_encoder" in name]
                decay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name in decay_parameters]
                nodecay_vision_encoder_parameters = [name for name in vision_encoder_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_vision_encoder_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_encoder_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_vision_encoder_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_encoder_lr,
                    }
                ])
            
            if self.args.embedding_lr is not None:
                embedding_parameters = [
                    name for name, _ in optimized_parameters if "vision_distill_layer" in name or "vision_encoder.embeddings" in name
                ]
                decay_embedding_parameters = [name for name in embedding_parameters if name in decay_parameters]
                nodecay_embedding_parameters = [name for name in embedding_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_embedding_parameters], 
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.embedding_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_embedding_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.embedding_lr,
                    }
                ])

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == "cosine_with_min_lr":
                self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
                    self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    **self.args.lr_scheduler_kwargs,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["current_epoch"] = self.state.epoch
        loss, outputs = super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True)
        if hasattr(outputs, "image_loss") and hasattr(outputs, "teacher_loss") and hasattr(outputs, "relation_loss"):
            self.sum_image_loss = outputs.image_loss.item()
            self.sum_teacher_loss = outputs.teacher_loss.item()
            self.sum_relation_loss = outputs.relation_loss.item()
            self.sum_caption_loss = outputs.caption_loss.item()
            self.loss_count += 1

        return loss
    
    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if is_torch_xla_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])

        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
        if is_torch_npu_available():
            set_rng_state_for_device("NPU", torch.npu, checkpoint_rng_state, is_distributed)
        if is_torch_hpu_available():
            set_rng_state_for_device("HPU", torch.hpu, checkpoint_rng_state, is_distributed)
        if is_torch_mlu_available():
            set_rng_state_for_device("MLU", torch.mlu, checkpoint_rng_state, is_distributed)
        if is_torch_musa_available():
            set_rng_state_for_device("MUSA", torch.musa, checkpoint_rng_state, is_distributed)