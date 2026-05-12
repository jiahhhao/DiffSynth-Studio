# Metadata is local/customized now; do not redownload and overwrite it.
# modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-TI2V-5B/*" --local_dir ./data/diffsynth_example_dataset

# 第三版尝试：启用可训练 growth MLP，输出到新目录，避免和第二版固定 token checkpoint 混用。
# 第二版旧输出路径不再运行，保留为注释，避免和第三版 MLP checkpoint 混用。
# --output_path "./models/train/Wan2.2-TI2V-5B_lora_growth_context_token" \
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --multi_gpu --num_processes 2 examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv \
  --height 576 \
  --width 480 \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_paths '[["/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors","/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors","/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],"/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth","/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]' \
  --tokenizer_path /caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B/google/umt5-xxl \
  --learning_rate 1e-4 \
  --num_epochs 3 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_lora_growth_mlp_context_token" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --train_growth_mlp
