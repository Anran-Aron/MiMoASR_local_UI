# MiMoASR_local_UI

[中文](README.md) | [English](README.en.md)

本项目提供一个本地 Web UI，用于把音频转写为 Markdown。转写使用 `mimo-v2.5-asr`，可选轻量文本标注或 pyannote 本地说话人分离标注。

## 快速开始

macOS 用户双击：

```text
start_ui.command
```

Windows 用户双击：

```text
start_ui.bat
```

首次启动脚本会自动：

- 在项目目录创建 `.venv` 虚拟环境；
- 安装 `requirements.txt` 中的依赖；
- 如果没有 `.env`，从 `.env.example` 创建一份；
- 启动本地服务并打开 `http://127.0.0.1:7860`。

关闭启动脚本打开的终端窗口，或在该窗口按 `Ctrl+C`，即可停止本地服务并释放 `7860` 端口。只关闭浏览器标签页不会停止服务。

## 必要配置

进入网页右上角设置，填写：

- `MIMO_API_KEY`：调用 Xiaomi Mimo ASR 必需；
- `HF_TOKEN`：只有选择 pyannote 方案时需要。

如果要使用 pyannote 方案，需要先完成 Hugging Face 配置：

1. 注册或登录 [Hugging Face](https://huggingface.co/join)。
2. 创建 [Access Token](https://huggingface.co/settings/tokens)，Token 类型选择 `Read` 即可。
3. 打开并接受以下模型/依赖模型的使用条款：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. 回到本地 UI 设置页，把 Token 填入 `HF_TOKEN`。

`.env`、`.venv`、模型缓存和输出缓存均在项目目录内或被 `.gitignore` 忽略，不应提交到 GitHub。

## 使用方式

1. 打开本地 UI。
2. 拖入音频文件，文件可以不在项目目录内。
3. 选择是否需要说话人和时间戳。
4. 点击开始转写。
5. 输出 Markdown 会生成到项目内 `output/` 文件夹。

标注方案区别：

- 不标注：只输出普通转写，速度最快。
- 轻量化方案：不调用额外模型，几乎无额外等待；说话人和时间戳来自文本估算。
- pyannote 方案：本地运行说话人分离模型，耗时更久、首次需下载模型；说话人区分更可靠。

## 支持的输入格式

`mimo-v2.5-asr` 接口本身只支持 `wav` 和 `mp3`。本项目会在本地先尝试把其他音频格式转换为模型可接收的格式，因此 UI 中可以直接拖入常见音频文件。

已知适合尝试的格式包括：

```text
wav、mp3、m4a、aac、flac、ogg、opus、aiff、caf
```

也可以尝试包含音频轨的视频/容器格式，例如：

```text
mp4、mov
```

能否成功取决于 PyAV/FFmpeg 底层是否能解码该文件。损坏文件、加密音频、录音笔专有格式或非常少见的编码可能会失败。

## 运行逻辑

后端处理流程大致如下：

1. 检查音频是否已满足 Mimo ASR 的 `wav/mp3` 输入要求和大小限制；如不满足，优先用 PyAV 转为 `16000Hz`、单声道 `wav`。
2. 为满足接口大小和上下文限制，将音频切成短段，默认单段约 `90` 秒；切分时会尽量在静音/停顿处断开。
3. 分段调用 `mimo-v2.5-asr` 并合并为普通 Markdown；遇到限速会重试，遇到异常重复会自动把问题段再次切小后续写。
4. 如启用轻量化标注，会基于转写文本估算说话人和轮次级时间戳；如启用 pyannote，会先分析原始音频的说话人时间段，再与 Mimo 转写文本合并。
5. 处理成功后会清理上传临时文件；模型缓存和说话人分离缓存会保留，方便下次加速。

输出文件示例：

```text
output/录音名.md
output/录音名_annotated.md
output/录音名_diarized.md
```

说明：轻量化方案和 pyannote 方案中的文本时间戳都不是逐字精确对齐。pyannote 负责更可靠地判断说话人时间段，但最终 Markdown 仍需要与 Mimo 的分段转写文本进行合并。

## 系统与依赖

建议 Python 3.9 或更新版本。首次安装依赖需要联网。

Windows 和 macOS 都可以使用本地 Flask UI。macOS 使用 `.command`，Windows 使用 `.bat`。Linux 用户可参考 `start_ui.command` 手动运行。

pyannote 和 PyTorch 依赖体积较大，首次安装可能耗时较长。模型文件会缓存到项目内 `.model_cache/`，说话人分离结果会缓存到 `.diarization_cache/`。`.diarization_cache/` 中超过 30 天的缓存会在启动 UI 或运行 pyannote 标注时自动清理；这里保存的是说话人时间段 JSON，不是原始音频。

## GPU 加速

pyannote 的设备默认是 `auto`。脚本会按以下顺序自动选择：

```text
NVIDIA CUDA -> Apple MPS -> CPU
```

如果用户安装的是不支持 CUDA 的 PyTorch，即使电脑有 NVIDIA GPU，也会回退到 CPU。Windows/Linux 上想使用 NVIDIA GPU，通常需要按 PyTorch 官方说明安装 CUDA 版本的 torch。

## 命令行用法

原命令行脚本仍可直接使用，例如：

```bash
.venv/bin/python mimoasr_script.py audio_file.m4a -o output/audio_file.md
```

使用 pyannote：

```bash
.venv/bin/python mimoasr_script.py audio_file.m4a -o output/audio_file.md --diarize --num-speakers 2
```

## 端口说明

本地 UI 监听：

```text
127.0.0.1:7860
```

这只对本机开放，一般不会暴露到局域网或公网。若端口被其他程序占用，请先关闭占用 `7860` 的程序，或修改 `app.py` 中的端口。

## License

本项目使用 [MIT License](LICENSE) 开源许可。
