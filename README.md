# AudioLab · 语音采集及其降噪系统

基于 PySide6 的音频实验桌面程序，支持录音、音频导入与播放、噪声添加、维纳滤波、卡尔曼滤波、小波去噪、波形与频谱分析、场景增强，以及多语言语音识别和英文翻译。

## 安装与启动

推荐 Python 3.11。进入仓库根目录后执行：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

Linux 录音需要系统提供 PortAudio；使用麦克风时需要授予系统录音权限。可直接导入自己的音频文件完成降噪实验。

## 可选的语音识别与翻译

基础音频功能不需要 Whisper 模型。当前 AI 实现使用 NVIDIA CUDA GPU，未提供 CPU 自动回退；需配备兼容的 GPU、驱动及 CUDA 12/cuDNN 9 运行库。可通过以下命令安装 AI 依赖和 Python 提供的运行库：

```bash
python -m pip install -r requirements-ai.txt
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
python -m ai.download_models medium
# 需要界面中的“快速”模式时，再下载 base：
python -m ai.download_models base
```

模型放入 `data/models/whisper-medium` 或 `data/models/whisper-base`。识别使用本地模型；未下载模型时不会自动联网下载。约 1.53 GB 的 medium 模型不放进 Git 仓库，下载脚本保留在 `ai/download_models.py`。

## 验证


```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

最终版冻结为 `v1.0.0`。19 项现有测试通过，覆盖界面状态、音频强度插值、翻译线程和语音分段；AI 工作流测试使用模拟模型，不代表本次重新完成了真实 GPU 推理或麦克风实测。冻结信息及文件校验值见 `FINAL_VERSION.json`。

## 版权与项目介绍

Copyright (c) 2026 cin330。原创部分以 [AGPL-3.0](LICENSE) 发布，使用、修改及分发须保留署名并遵守许可。第三方内容保留原有权利。

[三个项目的完整中文介绍](PROJECT_INTRODUCTION.md) · [版权声明](COPYRIGHT.md) · [第三方组件](THIRD_PARTY.md)
