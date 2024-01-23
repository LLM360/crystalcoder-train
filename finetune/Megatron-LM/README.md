# CrystalChat Training Code

This is a modified version of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with the addition of muP and instruction dataset support to train CrystalChat. 

To launch the distributed training, run `rank_cmd_phase2.sh` on each node.

# Checkpoint conversion
To convert the checkpoint from Huggingface format to Megatron-LM you can use the following command:

```bash
python Megatron-LM/tools/checkpoint/util.py --model-type GPT --loader crystalcoder_hf --saver megatron --target-tensor-parallel-size 2 --target-pipeline-parallel-size 4 --load-dir path/to/CrystalCoder_phase2_checkpoint_214387_to_hf --save-dir checkpoints/meg/phase2_tp2_pp4_dev
```
After the training, to convert the checkpoint from Megatron-LM to Huggingface format you can use the following command:

```bash
python tools/checkpoint/util.py --model-type GPT --loader megatron --saver crystalcoder_hf --load-dir checkpoints/to/megatron/checkpoint --save-dir checkpoints/to/hf/model --max-queue-size=5 --hf-config-path CrystalCoder
```

# Orignal README for Megatron-LM

Go to [README.old.md](README.old.md) for the original README.
