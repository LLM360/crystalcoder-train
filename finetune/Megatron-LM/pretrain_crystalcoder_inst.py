# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)
# import datasets


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()
    assert args.use_mup
    print_rank_0('building GPT model with muP ...')
    config = core_transformer_config_from_args(get_args())

    model = megatron.model.GPTModelWithMuP(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        mup_embeddings_scale = args.mup_embeddings_scale,
        mup_output_alpha= args.mup_output_alpha,
        mup_width_scale = args.mup_width_scale,
        mup_scale_qk_dot_by_d = args.mup_scale_qk_dot_by_d,
    )

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    # Items and their type.
    keys = ['token_ids', 'label_ids', 'loss_mask']
    datatype = torch.int32

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        data = {"token_ids": data[:,0,:], "label_ids": data[:, 2, :], "loss_mask": data[:, 1, :]}
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['token_ids'].long()
    labels = data_b['label_ids'].long()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss)

    loss_mask = data_b['loss_mask'].float()

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    args = get_args()
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)



def instruction_train_valid_test_datasets_provider(num_samples):
    from h5_map_dataset.utils import set_defaults, get_params
    from h5_map_dataset.dataset import HDF5Dataset
    # args = get_args()
    params = get_params("/workspace/crystalcoder-train/pretrain/params/phase2/params.yaml")
    set_defaults(params)

    # print_rank_0(f'Building Instruction Datasets from: {args.data_path} ...')

    # train_ds = datasets.load_dataset(
    #     "json",
    #     data_files=args.data_path,
    #     split='train',
    #     # num_proc=min(len(data_files), os.cpu_count()),
    #     cache_dir='./train_cache/' + args.data_path[0].replace('/', '_'),
    # )
    # train_ds = train_ds.with_format("np")
    train_ds = HDF5Dataset(params["train_input"])
    # streaming data does not have length
    # if hasattr(train_ds, '__len__'):
    #     print_rank_0(f'huggingface dataset built, size = {len(train_ds)}')
    # else:
    #     print_rank_0(f'huggingface dataset is streaming')

    valid_ds, test_ds = None, None
    print_rank_0("> finished creating pretrain datasets ...")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(instruction_train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'LLaMATokenizer'})
