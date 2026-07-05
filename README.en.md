# MiMoASR_local_UI

[Chinese](README.md) | [English](README.en.md)

MiMoASR_local_UI provides a local Web UI for transcribing audio files into Markdown. Transcription uses `mimo-v2.5-asr`, with optional lightweight text-based annotation or local pyannote speaker diarization.

## Quick Start

On macOS, double-click:

```text
start_ui.command
```

On Windows, double-click:

```text
start_ui.bat
```

On first launch, the startup script will automatically:

- Create a `.venv` virtual environment in the project directory;
- Install dependencies from `requirements.txt`;
- Create `.env` from `.env.example` if `.env` does not exist;
- Start the local service and open `http://127.0.0.1:7860`.

Close the terminal window opened by the startup script, or press `Ctrl+C` in that window, to stop the local service and release port `7860`. Closing only the browser tab will not stop the service.

## Required Configuration

Open the settings panel in the top-right corner of the Web UI and fill in:

- `MIMO_API_KEY`: Required for calling Xiaomi Mimo ASR;
- `HF_TOKEN`: Required only when using the pyannote option.

To use pyannote, complete the Hugging Face setup first:

1. Sign up for or log in to [Hugging Face](https://huggingface.co/join).
2. Create an [Access Token](https://huggingface.co/settings/tokens). A `Read` token is enough.
3. Open and accept the terms for the following model and dependency model:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Return to the local UI settings page and paste the token into `HF_TOKEN`.

`.env`, `.venv`, model caches, and output caches are local/project-generated files and are ignored by `.gitignore`. Do not commit them to GitHub.

## Usage

1. Open the local UI.
2. Drag in an audio file. The file does not need to be inside the project directory.
3. Choose whether speaker labels and timestamps are needed.
4. Click start transcription.
5. Markdown output files will be generated in the project `output/` directory.

Annotation options:

- No annotation: Generates a plain transcription Markdown file. Fastest option.
- Lightweight annotation: Does not call an extra model and adds almost no extra waiting time; speaker labels and timestamps are estimated from text.
- pyannote annotation: Runs local speaker diarization; slower and downloads model files on first use, but speaker separation is more reliable.

## Supported Input Formats

The `mimo-v2.5-asr` API itself supports only `wav` and `mp3`. This project first tries to convert other audio formats locally into a format accepted by the model, so the UI can accept common audio files directly.

Known formats worth trying:

```text
wav, mp3, m4a, aac, flac, ogg, opus, aiff, caf
```

You can also try video/container files that contain an audio track, such as:

```text
mp4, mov
```

Whether a file can be processed depends on whether the PyAV/FFmpeg backend can decode it. Corrupted files, encrypted audio, proprietary recorder formats, or rare codecs may fail.

## Processing Flow

The backend flow is roughly:

1. Check whether the audio already satisfies Mimo ASR's `wav/mp3` input format and size limits. If not, prefer converting it locally with PyAV to `16000Hz`, mono `wav`.
2. Split audio into short chunks to satisfy API size and context limits. The default target chunk length is about `90` seconds, and the splitter tries to cut near silence or pauses.
3. Send chunks to `mimo-v2.5-asr` and merge them into a plain Markdown file. Rate limits are retried, and suspicious repeated output is handled by subdividing the problematic chunk and continuing.
4. If lightweight annotation is enabled, speaker labels and turn-level timestamps are estimated from the transcript text. If pyannote is enabled, the original audio is analyzed for speaker time ranges and then merged with the Mimo transcript.
5. Temporary upload files are cleaned after success. Model cache and speaker diarization cache are kept to speed up later runs.

Example output files:

```text
output/recording.md
output/recording_annotated.md
output/recording_diarized.md
```

Note: timestamps in both lightweight and pyannote annotated Markdown are not word-level exact alignments. pyannote provides more reliable speaker time ranges, but the final Markdown still needs to be merged with Mimo's segmented transcript text.

## System and Dependencies

Python 3.9 or newer is recommended. The first dependency installation requires internet access.

Both Windows and macOS can use the local Flask UI. macOS users run `.command`; Windows users run `.bat`. Linux users can refer to `start_ui.command` and run the equivalent shell commands manually.

pyannote and PyTorch are relatively large dependencies, so the first installation may take some time. Model files are cached in `.model_cache/`, and speaker diarization results are cached in `.diarization_cache/`. Cache files in `.diarization_cache/` older than 30 days are automatically cleaned when starting the UI or running pyannote annotation; these files store speaker time ranges as JSON, not original audio.

## GPU Acceleration

The default pyannote device is `auto`. The script tries devices in this order:

```text
NVIDIA CUDA -> Apple MPS -> CPU
```

If the installed PyTorch build does not support CUDA, the script will fall back to CPU even if the machine has an NVIDIA GPU. On Windows/Linux, using an NVIDIA GPU usually requires installing a CUDA-enabled PyTorch build according to the official PyTorch instructions.

## Command-Line Usage

The original command-line script can still be used directly, for example:

```bash
.venv/bin/python mimoasr_script.py audio_file.m4a -o output/audio_file.md
```

With pyannote:

```bash
.venv/bin/python mimoasr_script.py audio_file.m4a -o output/audio_file.md --diarize --num-speakers 2
```

## Port

The local UI listens on:

```text
127.0.0.1:7860
```

This is only exposed to the local machine and is generally not available to the LAN or public internet. If another program is using port `7860`, close that program first or change the port in `app.py`.

## License

This project is licensed under the [MIT License](LICENSE).
