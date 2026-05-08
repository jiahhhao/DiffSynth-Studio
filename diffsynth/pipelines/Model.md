# WAN 模型阅读指引

说明：下面列出在仓库中与 WAN / Wan-Video 相关的主要源码文件与文档，建议的阅读顺序按从宏观到微观排列，先看 Pipeline 与示例，再阅读各子模块实现。

推荐阅读文件（按顺序）：

- diffsynth/pipelines/wan_video.py — WAN 推理流水线，入口位置，建议首先阅读。
- diffsynth/models/wan_video_dit.py — Wan 的主模型（DIT）实现，含网络结构与前向逻辑。
- diffsynth/models/wan_video_dit_s2v.py — 带 S2V（speech-to-video）或相关 rope 预计算的变体/辅助实现。
- diffsynth/models/wan_video_text_encoder.py — 文本编码器实现，包含 tokenizer/embedding 逻辑。
- diffsynth/models/wan_video_vae.py — VAE（编码/解码）实现，负责帧级编码/解码。
- diffsynth/models/wan_video_image_encoder.py — 图像/帧编码器。
- diffsynth/models/wan_video_vace.py — Vace 相关模型（项目内特殊变体）。
- diffsynth/models/wan_video_motion_controller.py — 运动控制器网络，实现动作/运动相关模块。
- diffsynth/models/wan_video_animate_adapter.py — 动画适配器/adapter 模块（LoRA/adapter 风格扩展）。
- diffsynth/models/wan_video_mot.py — MOT（多目标跟踪）或与运动相关的模块。
- diffsynth/models/wav2vec.py — 音频编码器（WanS2VAudioEncoder），用于音频驱动模型。

辅助与示例：

- inference.py — 项目根目录的推理示例，展示如何从预训练权重建立 `WanVideoPipeline` 并运行。
- examples/wanvideo/ — 示例脚本与数据（若存在）用于快速上手与测试。
- docs/zh/Model_Details/Wan.md — 项目文档中关于 Wan 的详细说明（若存在，优先阅读以获取整体设计与使用说明）。

阅读建议：

1. 先读 `wan_video.py`，理解 pipeline 如何拼接各子模块、加载权重与调度器（FlowMatch）。
2. 阅读 `wan_video_text_encoder.py`、`wan_video_vae.py`、`wan_video_image_encoder.py`：这些是数据流的核心（文本→潜码、图像编码、帧重建）。
3. 深入 `wan_video_dit.py` 与 `wan_video_dit_s2v.py`，理解模型结构、位置编码、注意力与时间轴处理（若有 rope/pos embeddings，注意查看实现）。
4. 阅读控制类（`wan_video_motion_controller.py`、`wan_video_mot.py`、`wan_video_animate_adapter.py`）以了解运动与动画如何注入到生成流程中。
5. 参考 `inference.py` 与 `examples/wanvideo/`，运行推理示例以快速验证阅读成果。

如果你希望，我可以：

- 打开并摘录上述每个文件的关键函数/类与注释摘要；
- 为 `Model.md` 增加快速跳转链接（到仓库内文件）；
- 或直接在本目录生成一个更详细的阅读笔记（包含类图与调用关系）。
