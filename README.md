# CrystalCoder Training Code

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

We are currently working on instruction-tuned versions of CrystalCoder! The model, training code, and datasets will be released after completion.
