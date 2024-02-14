# Copyright 2022 Cerebras Systems.
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
import yaml


def set_attention_params(params):
    '''
    Set attention related params
    :param params: model_params
    :return:
    '''
    # Attention softmax is fp32 by default.
    params["model"]["attention_softmax_fp32"] = True

    if params["runconfig"].get("precision_opt_level", 1) == 2:
        params["model"]["attention_softmax_fp32"] = False

    if (
        params["model"].get("fp16_type", "bfloat16") == "cbfloat16"
        and params["runconfig"].get("precision_opt_level", 1) == 1
    ):
        params["model"]["attention_softmax_fp32"] = False


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    if (
        params.get("train_input", {}).get("data_processor")
        == "Gpt2SyntheticDataProcessor"
    ):
        if "train_input" in params:
            params["train_input"]["vocab_size"] = params["train_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["train_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in train_input ({params['train_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["train_input"]["max_sequence_length"] = params[
                "train_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

        if "eval_input" in params:
            params["eval_input"]["vocab_size"] = params["eval_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["eval_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in eval_input ({params['eval_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["eval_input"]["max_sequence_length"] = params[
                "eval_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

    params["model"]["fp16_type"] = params["model"].get("fp16_type", "bfloat16")
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )
    params["runconfig"]["precision_opt_level"] = params["runconfig"].get(
        "precision_opt_level", 1
    )
    set_attention_params(params)

def read_params_file(params_file: str) -> dict:
    """ Helper for loading params file. """
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def get_params(params_file: str,) -> dict:
    """Reads params from file and returns them as a dict.

    Args:
        params_file: The YAML file holding the params.
        config: Optional config to load from the params. If None, the default
            config is returned. Defaults to None.
    Returns:
        A dict containing the params.
    """
    params = read_params_file(params_file)

    return params