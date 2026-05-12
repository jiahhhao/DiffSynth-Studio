import os
import glob
import torch
from PIL import Image

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


# 这里改成你自己的 Wan2.2-TI2V-5B 真实路径
model_dir = "/caoyunkang/zjh/proj/Wan2.2/Wan2.2-TI2V-5B"

# 输入图片路径，改成你的幼苗图像
image_path = "youmiao.png"

# ---- TEST CONFIGS ----
HIGH = 576
WIDTH = 480
FRAME=9
DAYS=30
STEPS=30
ENV='D'
CROSSATTN='context'
CONFIX='MLP'
FPS=1

# 自动找到 diffusion 分片
dit_paths = sorted(glob.glob(os.path.join(model_dir, "diffusion_pytorch_model-*.safetensors")))

print("找到 diffusion 分片：")
for p in dit_paths:
    print("  ", p)

if len(dit_paths) == 0:
    raise FileNotFoundError(
        f"没有找到 diffusion_pytorch_model-*.safetensors，请检查 model_dir 是否正确：{model_dir}"
    )

# 低显存/稳妥配置
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 1. 主 diffusion 模型，注意这里是具体分片文件列表，不是目录
        ModelConfig(
            path=dit_paths,
            **vram_config,
        ),

        # 2. T5 文本编码器
        ModelConfig(
            path=os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            **vram_config,
        ),

        # 3. Wan2.2 VAE，注意不是 Wan2.1_VAE.pth
        ModelConfig(
            path=os.path.join(model_dir, "Wan2.2_VAE.pth"),
            **vram_config,
        ),
    ],

    # 你的目录里有 google 文件夹，通常里面是 google/umt5-xxl
    tokenizer_config=ModelConfig(
        path=os.path.join(model_dir, "google", "umt5-xxl")
    ),

    # 用当前空闲显存，而不是总显存
    vram_limit=torch.cuda.mem_get_info("cuda")[0] / (1024 ** 3) - 2,
)

lora_path = "/caoyunkang/zjh/proj/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_lora_growth_mlp_context_token/epoch-2.safetensors"

pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

# 读入输入图片
input_image = Image.open(image_path).convert("RGB").resize((WIDTH, HIGH))

if ENV == 'D':
    PROMPT = (
        "Greenhouse potted maize seedling under drought stress, humidity 10%, water shortage, curled leaves, wilted leaves, dry leaf tips, slower growth."
    )
    NEGATIVE_PROMPT = (
        "low quality, blurry, static, distorted plant, extra leaves, deformed structure, flickering, "
        "watermark, text, bad motion, oversaturated, overexposed, noisy background, "
        "healthy vigorous growth, lush green leaves, unrealistic recovery, large scene change"
    )
else:
    PROMPT = (
        "Greenhouse potted maize seedling, healthy green leaves, normal growth, humidity 50%, no water shortage."
    )
    NEGATIVE_PROMPT = (
        "low quality, blurry, static, distorted plant, extra leaves, deformed structure, "
        "flickering, watermark, text, bad motion, oversaturated, overexposed, noisy background"
    )

video = pipe(
    input_image=input_image,
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    # noraml
    # prompt = ("Greenhouse potted maize seedling, healthy green leaves, normal growth, humidity 50%, no water shortage."),
    # prompt = (
    # "Use the input image as a fixed scene. "
    # "Keep the greenhouse environment, pot, background, lighting, and camera viewpoint unchanged. "
    # "Only the corn seedling grows naturally, like a fixed-camera plant growth time-lapse. "
    # "The plant develops into the appearance of about 30 days later, becoming taller and more mature, "
    # "with longer and more expanded green leaves. "
    # "Everything except the plant remains the same."
    # ),
    # negative_prompt=(
    #     "low quality, blurry, static, distorted plant, extra leaves, deformed structure, "
    #     "flickering, watermark, text, bad motion, oversaturated, overexposed, noisy background"
    # ),

    
    # prompt = ("Greenhouse potted maize seedling under drought stress, humidity 10%, water shortage, curled leaves, wilted leaves, dry leaf tips, slower growth."),
    # prompt = (
    # "Use the input image as a fixed scene. "
    # "Keep the greenhouse environment, pot, background, lighting, and camera viewpoint unchanged. "
    # "Only the corn seedling changes naturally under drought-stressed conditions, like a fixed-camera plant drought experiment time-lapse. "
    # "The plant is under continuous water deficit, with irrigation stopped from an early growth stage and soil moisture gradually reduced. "
    # "About 30 days later, the corn plant shows clear drought stress symptoms: inhibited growth, reduced vigor, "
    # "less leaf expansion, leaf rolling, drooping, wilting, and partial yellowing or drying, especially on older leaves. "
    # "The plant should look stressed and less healthy than a well-watered plant. "
    # "If visible, the soil should appear dry. "
    # "Everything except the plant remains the same."
    # ),
    # negative_prompt=(
    # "low quality, blurry, static, distorted plant, extra leaves, deformed structure, flickering, "
    # "watermark, text, bad motion, oversaturated, overexposed, noisy background, "
    # "healthy vigorous growth, lush green leaves, unrealistic recovery, large scene change"
    # ),
    height=HIGH,
    width=WIDTH,
    num_frames=FRAME,
    growth_days=DAYS,
    num_inference_steps=STEPS,
    seed=0,
    tiled=True,
)

save_video(video, f"./results/MLP/wan22_{HIGH}x{WIDTH}_{FRAME}f_{DAYS}d_{STEPS}s_seed0_{CROSSATTN}_{ENV}_{CONFIX}.mp4", 
        fps=FPS, quality=5, save_frames_dir=f"./results/MLP/wan22_{HIGH}x{WIDTH}_{FRAME}f_{DAYS}d_{STEPS}s_seed0_{CROSSATTN}_{ENV}_{CONFIX}_frames")