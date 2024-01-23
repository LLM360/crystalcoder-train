# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


def add_arguments(parser):
    group = parser.add_argument_group(title='CrystalCoder HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    # group.add_argument('--vocab-file', type=str, default=None,
    #                    help='Path to the vocab file. If specified will use this to get vocab size and '
    #                    'trim padding from the embedding table.')
    # group.add_argument('--tokenizer-model', required=True,
    #                    help='Sentencepiece tokenizer model.')
    group.add_argument('--tokenizer-model', default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


def load_args_from_checkpoint(args):

    # Read crystalcoder args.
    crystalcoder_args_path = os.path.join(args.load, "config.json")
    with open(crystalcoder_args_path) as f:
        crystalcoder_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 2048
    args.max_position_embeddings = crystalcoder_args['n_positions']
    args.hidden_size = crystalcoder_args["n_embd"]
    args.num_attention_heads = crystalcoder_args["n_head"]
    args.num_layers = crystalcoder_args["n_layer"]
    args.global_batch_size = 1024
    args.norm_epsilon = crystalcoder_args["layer_norm_epsilon"]
    args.iteration = 1 # '0', 'release' don't work # TODO: what does this mean?
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    args.tokenizer_type = "NullTokenizer"
    args.fp16 = True
    args.normalization = "LayerNorm"
    args.add_bias_linear = True
    args.apply_query_key_layer_scaling = False
    args.untie_embeddings_and_output_weights = (crystalcoder_args['tie_word_embeddings'] == False)
    args.vocab_size = crystalcoder_args["vocab_size"]
    args.padded_vocab_size = crystalcoder_args["vocab_size"]
    args.crystalcoder_args = crystalcoder_args
    args.ffn_hidden_size = crystalcoder_args["n_inner"]
    args.rotary_percent = 0.25
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.95
    args.adam_eps = 1e-08
    args.attention_dropout = crystalcoder_args["attn_pdrop"]
    args.hidden_dropout = 0.0


    if "num_key_value_heads" in crystalcoder_args:
        args.group_query_attention = True
        args.num_query_groups = crystalcoder_args["num_key_value_heads"]


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    # model.language_model.embedding.word_embeddings.weight.data.copy_(
    #     hf_model.model.embed_tokens.weight)

    modified = set()

    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.transformer.wte.weight)
    modified.add(model.language_model.embedding.word_embeddings.weight)

    return modified

def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    # model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
   
    modified = set()
   
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.transformer.ln_f.weight)
    modified.add(model.language_model.encoder.final_norm.weight)

    model.language_model.encoder.final_norm.bias.data.copy_(hf_model.transformer.ln_f.bias)
    modified.add(model.language_model.encoder.final_norm.bias)

    if args.untie_embeddings_and_output_weights:
        model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight) # when embedding is tied, there is no "output_layer" in model.language_model
        modified.add(model.language_model.output_layer.weight)

    return modified


def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    modified = set()
    # Copy weights (re-order dimensions for Megatron).

    # the ordering of qkv matric in megatron and crystalcoder is quite different
    # the crystalcoder is [q, k, v], while megatron is [qi, ki, vi] * num_heads
    # where q has for example 4096 dimensions, qi has only 128 dimensions (kv_channels)

    # from matplotlib import pyplot as plt
    # weight size: [hidden_size, hidden_size * 3]
    # hf_attn.c_attn.weight.t(): [hidden_size * 3, hidden_size]
    q_proj_w = hf_attn.c_attn.weight.t()[:args.hidden_size, :]
    k_proj_w = hf_attn.c_attn.weight.t()[args.hidden_size: 2*args.hidden_size, :]
    v_proj_w = hf_attn.c_attn.weight.t()[2*args.hidden_size:, :]

    # bias size: [hidden_size * 3]
    q_proj_bias = hf_attn.c_attn.bias[:args.hidden_size]
    k_proj_bias = hf_attn.c_attn.bias[args.hidden_size: 2*args.hidden_size]
    v_proj_bias = hf_attn.c_attn.bias[2*args.hidden_size:]
    # attn.query_key_value.weight.data.copy_(hf_attn.c_attn.weight.t())
    # attn.query_key_value.bias.data.copy_(hf_attn.c_attn.bias)
  
    # qkv_concat = torch.cat ([q_proj_w, k_proj_w, v_proj_w], dim=0)
    megatron_qkv = torch.cat([ 
        q_proj_w.reshape((ng, dim*nh//ng, -1)),
        k_proj_w.reshape((ng, dim, -1)),
        v_proj_w.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size))

    # diff = (qkv_concat != megatron_qkv)
    # diff_cnt = diff.sum()
    # plt.imsave("diff.png", diff.detach().numpy())
    attn.query_key_value.weight.data.copy_(megatron_qkv)
    attn.query_key_value.bias.data.copy_(torch.cat([
        q_proj_bias.reshape((ng, dim*nh//ng)),
        k_proj_bias.reshape((ng, dim)),
        v_proj_bias.reshape((ng, dim)),
    ], dim=1).reshape((-1,)))

    modified.add(attn.query_key_value.weight)
    modified.add(attn.query_key_value.bias)


    attn.dense.weight.data.copy_(hf_attn.c_proj.weight.t())
    attn.dense.bias.data.copy_(hf_attn.c_proj.bias)
    modified.add(attn.dense.weight)
    modified.add(attn.dense.bias)

    # attn.dense.weight.data.copy_(hf_attn.o_proj.weight)

    return modified

def set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    modified = set()
    mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
        hf_mlp.c_fc2.weight.t(),
        hf_mlp.c_fc.weight.t(),
    ], dim=0))
    mlp.dense_h_to_4h.bias.data.copy_(torch.cat([
        hf_mlp.c_fc2.bias,
        hf_mlp.c_fc.bias,
    ], dim=0))
    modified.add(mlp.dense_h_to_4h.weight)
    modified.add(mlp.dense_h_to_4h.bias)
    # mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
    #     hf_mlp.gate_proj.weight,
    #     hf_mlp.up_proj.weight,
    # ], dim=0))
    mlp.dense_4h_to_h.weight.data.copy_(hf_mlp.c_proj.weight.t())
    mlp.dense_4h_to_h.bias.data.copy_(hf_mlp.c_proj.bias)
    modified.add(mlp.dense_4h_to_h.weight)
    modified.add(mlp.dense_4h_to_h.bias)

    return modified


def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.language_model.encoder.layers[layer_idx]
    # hf_layer = hf_model.model.layers[layer_idx]
    hf_layer = hf_model.transformer.h[layer_idx]

    modified = set()
    modified.update(set_attn_state(args, layer, hf_layer))
    modified.update(set_mlp_state(args, layer, hf_layer))

    layer.input_norm.weight.data.copy_(hf_layer.ln_1.weight)
    layer.input_norm.bias.data.copy_(hf_layer.ln_1.bias)

    layer.post_attention_norm.weight.data.copy_(hf_layer.ln_2.weight)
    layer.post_attention_norm.bias.data.copy_(hf_layer.ln_2.bias)

    modified.add(layer.input_norm.weight)
    modified.add(layer.input_norm.bias)
    modified.add(layer.post_attention_norm.weight)
    modified.add(layer.post_attention_norm.bias)

    return modified


def load_checkpoint_to_model(args):
    '''Set model params. From huggingface checkpoint to Megatron-LM model.'''

    from pretrain_gpt import model_provider
    from transformers import AutoModelForCausalLM

    # Load Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(args.load, device_map="cpu", trust_remote_code=True)

    # Init Megatron model.
    model = model_provider(True, True).to(args.params_dtype)

    all_named_params = list(model.named_parameters())

    # Set model state.
    modified = set()
    modified.update(set_preprocess_state(args, model, hf_model))
    modified.update(set_postprocess_state(args, model, hf_model))
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        modified.update(set_layer_state(args, model, hf_model, layer_idx))

    unmodified = [ (name, param) for name, param in all_named_params if param not in modified]

    assert len(unmodified) == 0, f"Unconverted parameters: {unmodified}"
    return model


def _load_checkpoint(queue, args):

    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us.
    sys.argv = ['script.py',
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--load', args.load_dir
                ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes.
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('disable_bias_linear', False)
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)

    # Determine how to make our models.
    assert args.model_type == 'GPT', 'CrystalCoder is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warning about torch.distributed not being initialized.
    module.MegatronModule.embedding_warning_printed = True

    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Short aliases.
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = True
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None # skips padding in saver
    md.make_vocab_size_divisible_by = None
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = {
        "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    else:
        assert not hasattr(model.language_model.embedding, 'position_embeddings')

    queue_put("embeddings", message)

    for layer_num in range(margs.num_layers):
        message = {}

        # Get non-parallel tensors from tp_rank 0.
        layer = model.language_model.encoder.layers[layer_num]
        message["input norm weight"] = layer.input_norm.weight.data
        message["input norm bias"] = layer.input_norm.bias.data
        message["post norm weight"] = layer.post_attention_norm.weight.data
        message["post norm bias"] = layer.post_attention_norm.bias.data
        if md.linear_bias:
            message["dense bias"] = layer.self_attention.dense.bias.data
            message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

        # Grab all parallel tensors for this layer.
        qkv_weight = []
        qkv_bias = []
        dense_weight = []
        mlp_l0_weight = []
        mlp_l0_bias = []
        mlp_l1_weight = []
        layer = model.language_model.encoder.layers[layer_num]
        qkv_weight.append(layer.self_attention.query_key_value.weight.data)
        dense_weight.append(layer.self_attention.dense.weight.data)
        mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
        mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)
        if md.linear_bias:
            qkv_bias.append(layer.self_attention.query_key_value.bias.data)
            mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)

        # Handle gated linear units.
        if md.swiglu:
            # Concat all the first halves ('W's) and all the second halves ('V's).
            for tp_rank in range(tp_size):
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # Simple concat of the rest.
        message["qkv weight"] = torch.cat(qkv_weight, dim=0)
        message["dense weight"] = torch.cat(dense_weight, dim=1)
        message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
        if md.linear_bias:
            message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            if md.swiglu:
                for tp_rank in range(tp_size):
                    mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
                message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
                message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
            else:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

        queue_put(f"transformer layer {layer_num}", message)

    # Send final norm from tp_rank 0.
    message = {
        "weight": model.language_model.encoder.final_norm.weight.data,
        "bias": model.language_model.encoder.final_norm.bias.data
    }
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": model.language_model.output_layer.weight.data
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
