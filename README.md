<h1 align="center">CrystalCoder</h1>

<div align="center">
   <img src="crystalcoder_logo.jpg" alt="logo" width="300"><br><br>
</div>

---

<p align="center">
   <a href="https://github.com/LLM360/Analysis360/blob/dev/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"></a>
</p>
<p align="center">
🤗 <a href="https://huggingface.co/LLM360/CrystalCoder">[Model Download]</a> • 📈 <a href="https://github.com/LLM360/Analysis360/blob/main/README.md#list-of-analysis-and-metrics
">[Analysis and Results]</a> • 📗 <a href="https://huggingface.co/datasets/LLM360/CrystalCoderDatasets">Pretraining Dataset</a>
</p>

This repository contains the training code for [CrystalCoder](https://huggingface.co/LLM360/CrystalCoder), a 7B-parameter language model pretrained on code and natural language.

# Pretraining

## Code 
The training of Phase 1-3 is completed using [Cerebras's Model Zoo](https://github.com/Cerebras/modelzoo) on Cerebras CS-2 hardware.

To launch the training, you can use the following command:

```bash
git clone https://github.com/Cerebras/modelzoo
cd modelzoo

python modelzoo/transformers/pytorch/gpt3/run.py CSX \
  --mode train \
  --num_csx 16 \
  --num_workers_per_csx 1 \ # only needed for rel-1.9, not needed for rel-2.0
  --params <path to params.yaml> \
  --python_paths <path to modelzoo> \
  --mount_dirs <mounts needed for modelzoo and data sets> \
  --model_dir <path to model dir> \
  --checkpoint_path <path to initalization checkpoint> \
```

In the script, you need to specify different `--params` for different phases. The params files we used for each phase are in the `params` folder.

The `--checkpoint_path` is optional, and you can use it to specify the initialization checkpoint for the training. For example, for phase 2, you need to use the last checkpoint of phase 1 as the initialization checkpoint.

## Dataset

The processed dataset for each phase is available at [CrystalCoderDatasets](https://huggingface.co/datasets/LLM360/CrystalCoderDatasets).

If you plan to process the dataset yourself from scratch, you can refer to our [CrystalCoder data preprocessing code](https://github.com/LLM360/crystalcoder-data-prep).


# Instruction Tuning

The code for instruction tuning is in the `finetune` folder.

## CrystalChat Training Code

This is a modified version of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with the addition of muP and instruction dataset support to train CrystalChat. 

To launch the distributed training, run `rank_cmd_phase2.sh` on each node.

## Checkpoint conversion
To convert the checkpoint from Huggingface format to Megatron-LM you can use the following command:

```bash
python Megatron-LM/tools/checkpoint/util.py --model-type GPT --loader crystalcoder_hf --saver megatron --target-tensor-parallel-size 2 --target-pipeline-parallel-size 4 --load-dir path/to/CrystalCoder_phase2_checkpoint_214387_to_hf --save-dir checkpoints/meg/phase2_tp2_pp4_dev
```
After the training, to convert the checkpoint from Megatron-LM to Huggingface format you can use the following command:

```bash
python Megatron-LM/tools/checkpoint/util.py --model-type GPT --loader megatron --saver crystalcoder_hf --load-dir checkpoints/to/megatron/checkpoint --save-dir checkpoints/to/hf/model --max-queue-size=5 --hf-config-path CrystalCoder
```

## Orignal README for Megatron-LM

Go to [README.old.md](finetune/Megatron-LM/README.old.md) for the original README.


# Citation

**BibTeX:**

```bibtex
@misc{liu2023llm360,
      title={LLM360: Towards Fully Transparent Open-Source LLMs}, 
      author={Zhengzhong Liu and Aurick Qiao and Willie Neiswanger and Hongyi Wang and Bowen Tan and Tianhua Tao and Junbo Li and Yuqi Wang and Suqi Sun and Omkar Pangarkar and Richard Fan and Yi Gu and Victor Miller and Yonghao Zhuang and Guowei He and Haonan Li and Fajri Koto and Liping Tang and Nikhil Ranjan and Zhiqiang Shen and Xuguang Ren and Roberto Iriondo and Cun Mu and Zhiting Hu and Mark Schulze and Preslav Nakov and Tim Baldwin and Eric P. Xing},
      year={2023},
      eprint={2312.06550},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
