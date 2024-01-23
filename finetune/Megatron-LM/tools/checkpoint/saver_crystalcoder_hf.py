
import argparse
from collections.abc import Mapping
import concurrent.futures
import os
import sys

import torch
import transformers
from transformers.modeling_utils import no_init_weights

def add_arguments(parser):
    group = parser.add_argument_group(title='Huggingface CrystalCoder saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    
    group.add_argument('--hf-config-path', type=str, default=None,
                       help='Huggingface config directory')


def save_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # try:
    #     from megatron.arguments import (parse_args, validate_args)
    #     from megatron.checkpointing import save_checkpoint
    #     from megatron.global_vars import set_global_variables, get_args
    #     from megatron.core.enums import ModelType
    #     from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    #     from megatron import fused_kernels
    #     from megatron.core import mpu
    # except ModuleNotFoundError:
    #     print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
    #     exit(1)

    hf_config = transformers.AutoConfig.from_pretrained(args.hf_config_path, trust_remote_code=True)
    
  
    hf_model = transformers.AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    hf_model.requires_grad_(False)
    # zero the params
    for param in hf_model.parameters():
        param.data.zero_()

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)


    md = queue_get()



    # Embeddings
    #-----------
    # Send embeddings.
    # message = {
    #     "word embeddings": model.language_model.embedding.word_embeddings.weight.data
    # }
    # if md.position_embedding_type == 'learned_absolute':
    #     message["position embeddings"] = model.language_model.embedding.position_embeddings.weight.data
    # else:
    #     assert not hasattr(model.language_model.embedding, 'position_embeddings')

    # queue_put("embeddings", message)

    embeddings_msg = queue_get("embeddings")
    hf_model.transformer.wte.weight.copy_(embeddings_msg['word embeddings'])

    # send 
    # for layer_num in range(md.num_layers):
    #     message = {}

    #     # Get non-parallel tensors from tp_rank 0.
    #     layer = model.language_model.encoder.layers[layer_num]
    #     message["input norm weight"] = layer.input_norm.weight.data
    #     message["input norm bias"] = layer.input_norm.bias.data
    #     message["post norm weight"] = layer.post_attention_norm.weight.data
    #     message["post norm bias"] = layer.post_attention_norm.bias.data
    #     if md.linear_bias:
    #         message["dense bias"] = layer.self_attention.dense.bias.data
    #         message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

    #     # Grab all parallel tensors for this layer.
    #     qkv_weight = []
    #     qkv_bias = []
    #     dense_weight = []
    #     mlp_l0_weight = []
    #     mlp_l0_bias = []
    #     mlp_l1_weight = []
    #     layer = model.language_model.encoder.layers[layer_num]
    #     qkv_weight.append(layer.self_attention.query_key_value.weight.data)
    #     dense_weight.append(layer.self_attention.dense.weight.data)
    #     mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
    #     mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)
    #     if md.linear_bias:
    #         qkv_bias.append(layer.self_attention.query_key_value.bias.data)
    #         mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)

    #     # Handle gated linear units.
    #     if md.swiglu:
    #         # Concat all the first halves ('W's) and all the second halves ('V's).
    #         for tp_rank in range(tp_size):
    #             mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
    #         message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
    #         message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
    #     else:
    #         message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

    #     # Simple concat of the rest.
    #     message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    #     message["dense weight"] = torch.cat(dense_weight, dim=1)
    #     message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
    #     if md.linear_bias:
    #         message["qkv bias"] = torch.cat(qkv_bias, dim=0)
    #         if md.swiglu:
    #             for tp_rank in range(tp_size):
    #                 mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
    #             message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias],dim=0)
    #             message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias],dim=0)
    #         else:
    #             message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

    #     queue_put(f"transformer layer {layer_num}", message)

    
    # recieve

    def set_hf_layer_state(hf_model, layer_idx, layer_state):
        hf_layer = hf_model.transformer.h[layer_idx]
        hf_layer.ln_1.weight.copy_(layer_state["input norm weight"])
        hf_layer.ln_1.bias.copy_(layer_state["input norm bias"])
        hf_layer.ln_2.weight.copy_(layer_state["post norm weight"])
        hf_layer.ln_2.bias.copy_(layer_state["post norm bias"])
        
        ######### set attn
        hf_attn = hf_layer.attn
        megatron_qkv_weight = layer_state["qkv weight"]
        megatron_qkv_bias = layer_state["qkv bias"]

        nh = hf_config.n_head
        ng = hf_config.n_head # no group query attention
        dim = hf_config.n_embd // nh

        megatron_qkv_weight = megatron_qkv_weight.view(ng, 3*dim, -1)
        q_proj_w_megatron = megatron_qkv_weight[:, :dim, :]
        k_proj_w_megatron = megatron_qkv_weight[:, dim:2*dim, :]
        v_proj_w_megatron = megatron_qkv_weight[:, 2*dim:, :]

        hf_attn_c_attn_weight = torch.cat([
            q_proj_w_megatron.reshape(-1, hf_config.n_embd),
            k_proj_w_megatron.reshape(-1, hf_config.n_embd),
            v_proj_w_megatron.reshape(-1, hf_config.n_embd)
        ], dim=0).t()


        megatron_qkv_bias = megatron_qkv_bias.view(ng, 3*dim)
        q_proj_bias_megatron = megatron_qkv_bias[:, :dim]
        k_proj_bias_megatron = megatron_qkv_bias[:, dim:2*dim]
        v_proj_bias_megatron = megatron_qkv_bias[:, 2*dim:]

        # Reshape back to the original format and concatenate
        hf_attn_c_attn_bias = torch.cat([
            q_proj_bias_megatron.reshape(-1),
            k_proj_bias_megatron.reshape(-1),
            v_proj_bias_megatron.reshape(-1)
        ])

        # Assign to Hugging Face model
        hf_attn.c_attn.weight.data.copy_(hf_attn_c_attn_weight)
        hf_attn.c_attn.bias.data.copy_(hf_attn_c_attn_bias)

        hf_attn.c_proj.weight.data.copy_(layer_state["dense weight"].t())
        hf_attn.c_proj.bias.data.copy_(layer_state["dense bias"])

        ######## set mlp
        hf_mlp = hf_layer.mlp

        # Splitting the concatenated weights and biases
        megatron_dense_h_to_4h_weight = torch.concat([layer_msg["mlp l0 weight W"], layer_msg["mlp l0 weight V"]], dim=0)
        megatron_dense_h_to_4h_bias = torch.concat([layer_msg["mlp l0 bias W"], layer_msg["mlp l0 bias V"]], dim=0)

        hf_mlp_c_fc2_weight = megatron_dense_h_to_4h_weight[:hf_config.n_inner, :].t()
        hf_mlp_c_fc_weight = megatron_dense_h_to_4h_weight[hf_config.n_inner:, :].t()

        hf_mlp_c_fc2_bias = megatron_dense_h_to_4h_bias[:hf_config.n_inner]
        hf_mlp_c_fc_bias = megatron_dense_h_to_4h_bias[hf_config.n_inner:]

        # Assigning the split weights and biases to the corresponding Hugging Face layers
        hf_mlp.c_fc2.weight.data.copy_(hf_mlp_c_fc2_weight)
        hf_mlp.c_fc2.bias.data.copy_(hf_mlp_c_fc2_bias)
        hf_mlp.c_fc.weight.data.copy_(hf_mlp_c_fc_weight)
        hf_mlp.c_fc.bias.data.copy_(hf_mlp_c_fc_bias)

        # Copying the weights and biases for dense_4h_to_h layer
        hf_mlp.c_proj.weight.data.copy_(layer_msg["mlp l1 weight"].data.t())
        hf_mlp.c_proj.bias.data.copy_(layer_msg["mlp l1 bias"].data)


    for layer_num in range(md.num_layers):
        layer_msg = queue_get(f"transformer layer {layer_num}")
        set_hf_layer_state(hf_model, layer_num, layer_msg)

    final_norm_msg = queue_get("final norm")
    hf_model.transformer.ln_f.weight.copy_(final_norm_msg["weight"])
    hf_model.transformer.ln_f.bias.copy_(final_norm_msg["bias"])

    if md.output_layer:
        output_layer_msg = queue_get("output layer")
        hf_model.lm_head.weight.copy_(output_layer_msg["weight"])
    else:
        # tied embeddings
        # just assign the input embeddings to the output layer
        if (hf_model.lm_head.weight != hf_model.transformer.wte.weight).any():
            print("ERROR: output layer is not tied to input embeddings")

    msg = queue_get()
    if msg != "done":
        print("ERROR: got some more data but was expecting to be done")

    # Send final norm from tp_rank 0.
    # message = {
    #     "weight": model.language_model.encoder.final_norm.weight.data,
    #     "bias": model.language_model.encoder.final_norm.bias.data
    # }
    # queue_put("final norm", message)

    # if md.output_layer:
    #     message = {
    #         "weight": model.language_model.output_layer.weight.data
    #     }
    #     queue_put("output layer", message)

    # queue.put("done")

    # make sure all params.abs().sum() is not zero
    for name, param in hf_model.named_parameters():
        if param.abs().sum() == 0:
            print(f"ERROR: param {name} is zero")


    print("Saving model to ", args.save_dir)
    hf_model.save_pretrained(args.save_dir)
    # load and save tokenizer as well

    print("Saving tokenizer to ", args.save_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_config_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.save_dir)

    print("Saver Done!")
