
import argparse
from collections.abc import Mapping
import concurrent.futures
import os
import sys

import torch
import transformers
from transformers.modeling_utils import no_init_weights

def add_arguments(parser):
    group = parser.add_argument_group(title='Huggingface LLaMA 2 saver')

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



    embeddings_msg = queue_get("embeddings")
    hf_model.model.embed_tokens.weight.copy_(embeddings_msg['word embeddings'])

    # recieve

    def set_hf_layer_state(hf_model, layer_idx, layer_state):
        hf_layer = hf_model.model.layers[layer_idx]
        hf_layer.input_layernorm.weight.copy_(layer_state["input norm weight"])
        # hf_layer.input_layernorm.bias.copy_(layer_state["input norm bias"])
        hf_layer.post_attention_layernorm.weight.copy_(layer_state["post norm weight"])
        # hf_layer.post_attention_layernorm.bias.copy_(layer_state["post norm bias"])
        
        ######### set attn
        hf_attn = hf_layer.self_attn
        megatron_qkv_weight = layer_state["qkv weight"]


        nh = hf_config.num_attention_heads
        ng = hf_config.num_attention_heads # no group query attention
        dim = hf_config.hidden_size // nh

        megatron_qkv_weight = megatron_qkv_weight.view(ng, 3*dim, -1)
        q_proj_w_megatron = megatron_qkv_weight[:, :dim, :]
        k_proj_w_megatron = megatron_qkv_weight[:, dim:2*dim, :]
        v_proj_w_megatron = megatron_qkv_weight[:, 2*dim:, :]

        hf_attn.q_proj.weight.data.copy_(q_proj_w_megatron.reshape(-1, hf_config.hidden_size))
        hf_attn.k_proj.weight.data.copy_(k_proj_w_megatron.reshape(-1, hf_config.hidden_size))
        hf_attn.v_proj.weight.data.copy_(v_proj_w_megatron.reshape(-1, hf_config.hidden_size))

        hf_attn.o_proj.weight.data.copy_(layer_state["dense weight"])   

        ######## set mlp
        hf_mlp = hf_layer.mlp

        # Splitting the concatenated weights and biases
        megatron_dense_h_to_4h_weight = torch.concat([layer_msg["mlp l0 weight W"], layer_msg["mlp l0 weight V"]], dim=0)

        gate_proj = megatron_dense_h_to_4h_weight[:hf_config.intermediate_size, :]
        up_proj = megatron_dense_h_to_4h_weight[hf_config.intermediate_size:, :]


        # Assigning the split weights and biases to the corresponding Hugging Face layers
        hf_mlp.gate_proj.weight.data.copy_(gate_proj)
        hf_mlp.up_proj.weight.data.copy_(up_proj)


        # Copying the weights and biases for dense_4h_to_h layer
        hf_mlp.down_proj.weight.data.copy_(layer_msg["mlp l1 weight"].data)



    for layer_num in range(md.num_layers):
        layer_msg = queue_get(f"transformer layer {layer_num}")
        set_hf_layer_state(hf_model, layer_num, layer_msg)

    final_norm_msg = queue_get("final norm")
    hf_model.model.norm.weight.copy_(final_norm_msg["weight"])
    # hf_model.transformer.ln_f.bias.copy_(final_norm_msg["bias"])

    if md.output_layer:
        output_layer_msg = queue_get("output layer")
        hf_model.lm_head.weight.copy_(output_layer_msg["weight"])
    else:
        # tied embeddings
        # just assign the input embeddings to the output layer
        if (hf_model.lm_head.weight != hf_model.model.embed_tokens.weight).any():
            print("ERROR: output layer is not tied to input embeddings")

    msg = queue_get()
    if msg != "done":
        print("ERROR: got some more data but was expecting to be done")


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
