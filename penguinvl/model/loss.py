# Copyright (c) Penguin-VL team at Tencent AI Lab
import torch
import torch.nn.functional as F

from penguinvl.constants import IGNORE_INDEX


def cross_entropy_loss(
    hidden_states,
    lm_head,
    position_ids,
    labels,
    reduction_scope="sequence",
    **loss_kwargs
):
    batch_size = hidden_states.size(0)

    shift_hidden_states = hidden_states[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = shift_labels != IGNORE_INDEX
    shift_hidden_states = shift_hidden_states[mask].contiguous()
    shift_labels = shift_labels[mask].contiguous()

    if mask.sum() == 0:
        print(f"Get labels={labels}. Found no sample to calculate loss!")
        pseudo_logits = lm_head(hidden_states[:, 0:1])
        loss = 0.0 * pseudo_logits.mean()
        return loss

    if "num_items_in_batch" not in loss_kwargs:
        reduction = "mean"
        denominator = None

    elif reduction_scope == "batch":
        reduction = "sum"
        denominator = loss_kwargs["num_items_in_batch"]

    elif reduction_scope == "sequence":
        reduction = "none"

        if batch_size == 1:
            # NOTE: packed sequence
            start_indices = torch.nonzero(position_ids[0] == 0)[:, 0]
            end_indices = F.pad(start_indices[1:], (0, 1), value=position_ids.size(1))
            batch_indices = torch.cat(
                [
                    torch.full((e - s,), fill_value=i, device=position_ids.device, dtype=torch.long)
                    for i, (s, e) in enumerate(zip(start_indices, end_indices))
                ],
            ).unsqueeze(0)
        else:
            batch_indices = torch.arange(batch_size, device=position_ids.device)
            batch_indices = batch_indices.unsqueeze(1).expand(-1, hidden_states.size(1))

        shift_batch_indices = batch_indices[..., :-1]
        shift_batch_indices = shift_batch_indices[mask].contiguous()
        num_tokens = F.one_hot(shift_batch_indices).sum(dim=0)
        denominator = num_tokens[shift_batch_indices] * loss_kwargs["num_items_in_batch"]

    else:
        raise ValueError(f"Unknown reduction scope: {reduction_scope}")

    shift_logits = lm_head(shift_hidden_states)
    loss = torch.nn.functional.cross_entropy(
        shift_logits,
        shift_labels,
        reduction=reduction,
    )

    if denominator is not None:
        loss = loss / denominator
        if loss.ndim > 0:
            loss = loss.sum()

    return loss
