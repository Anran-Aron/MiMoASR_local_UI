#!/usr/bin/env python3
"""Convert local audio when needed, transcribe it with Mimo ASR, and write Markdown."""

from __future__ import annotations

import argparse
import audioop
import base64
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import warnings
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL = "mimo-v2.5-asr"
MAX_BASE64_BYTES = 10 * 1024 * 1024
SAFE_AUDIO_BYTES = (MAX_BASE64_BYTES * 3 // 4) - (128 * 1024)
REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_CHUNK_SECONDS = 90
DEFAULT_MAX_SINGLE_SECONDS = 90
DEFAULT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_SUBDIVIDE_SECONDS = 60
DEFAULT_MIN_SUBDIVIDE_SECONDS = 15
DEFAULT_RETRY_ATTEMPTS = 6
DEFAULT_RETRY_WAIT_SECONDS = 60
DEFAULT_BETWEEN_REQUESTS_SECONDS = 2
DEFAULT_MP3_BITRATE = "48k"
DEFAULT_SILENCE_SEARCH_SECONDS = 45
DEFAULT_SILENCE_DBFS = -40.0
DEFAULT_DIARIZATION_BATCH_SIZE = 32
DEFAULT_DIARIZATION_CACHE_TTL_DAYS = 30
DEFAULT_STRATEGY = "wav"
SUPPORTED_INPUTS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}
STRATEGIES = {"auto", "compress", "split", "wav"}


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    mime_type: str
    index: int
    total: int
    start_seconds: float
    end_seconds: float
    label: str = ""


@dataclass(frozen=True)
class MarkdownSegment:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class AnnotatedTurn:
    segment_index: int
    start_seconds: float
    end_seconds: float
    speaker: int
    text: str


@dataclass(frozen=True)
class DiarizationSegment:
    start_seconds: float
    end_seconds: float
    speaker: int


@dataclass
class SpeakerState:
    current_speaker: int = 1
    pending_question_switch: bool = False
    max_speakers: int = 4


class RateLimitError(RuntimeError):
    pass


class TranscriptionQualityError(RuntimeError):
    pass


def configure_warning_filters() -> None:
    warning_patterns = (
        r"urllib3 v2 only supports OpenSSL.*",
        r"torchaudio\._backend\..* has been deprecated.*",
        r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*",
        r"Module 'speechbrain\.pretrained' was deprecated.*",
        r"std\(\): degrees of freedom is <= 0.*",
    )
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern, category=Warning)


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without adding another dependency."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def convert_to_wav(input_path: Path, work_dir: Path, sample_rate: int) -> Path:
    output_path = work_dir / f"{input_path.stem}.wav"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
    else:
        afconvert = shutil.which("afconvert")
        if not afconvert:
            raise RuntimeError(
                "找不到可用的音频转换工具。请安装 ffmpeg，或在 macOS 上确认 afconvert 可用。"
            )
        command = [
            afconvert,
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{sample_rate}",
            "-c",
            "1",
            str(input_path),
            str(output_path),
        ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        hint = ""
        if not ffmpeg:
            hint = " 当前文件可能不兼容 macOS afconvert，请安装 ffmpeg 后重试。"
        raise RuntimeError(f"音频转换失败：{details or exc}{hint}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("音频转换失败：未生成有效的 wav 文件。")

    return output_path


def compress_to_mp3_with_ffmpeg(input_path: Path, output_path: Path, sample_rate: int, bitrate: str) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg。")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-b:a",
        bitrate,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg MP3 压缩失败：{details or exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg MP3 压缩失败：未生成有效文件。")
    return output_path


def compress_to_mp3_with_pyav(input_path: Path, output_path: Path, sample_rate: int, bitrate: str) -> Path:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV。请运行 `.venv/bin/python -m pip install av` 后重试。") from exc

    bitrate_bps = parse_bitrate(bitrate)
    try:
        with av.open(str(input_path)) as source, av.open(str(output_path), "w", format="mp3") as target:
            stream = target.add_stream("mp3", rate=sample_rate)
            stream.bit_rate = bitrate_bps
            stream.codec_context.layout = "mono"
            resampler = AudioResampler(format="s16p", layout="mono", rate=sample_rate)

            for frame in source.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    for packet in stream.encode(resampled_frame):
                        target.mux(packet)

            for packet in stream.encode(None):
                target.mux(packet)
    except Exception as exc:
        raise RuntimeError(f"PyAV MP3 压缩失败：{exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("PyAV MP3 压缩失败：未生成有效文件。")
    return output_path


def convert_to_mono_wav_with_pyav(input_path: Path, output_path: Path, sample_rate: int) -> Path:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV。请运行 `.venv/bin/python -m pip install av` 后重试。") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with av.open(str(input_path)) as source, av.open(str(output_path), "w", format="wav") as target:
            stream = target.add_stream("pcm_s16le", rate=sample_rate)
            stream.codec_context.layout = "mono"
            resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)

            for frame in source.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    for packet in stream.encode(resampled_frame):
                        target.mux(packet)

            for packet in stream.encode(None):
                target.mux(packet)
    except Exception as exc:
        raise RuntimeError(f"PyAV WAV 转换失败：{exc}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("PyAV WAV 转换失败：未生成有效文件。")
    return output_path


def frame_dbfs(frame: Any) -> float:
    sample_width = 2
    pcm = bytes(frame.planes[0])[: frame.samples * sample_width]
    if not pcm:
        return -120.0
    rms = audioop.rms(pcm, sample_width)
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


def find_silence_aware_boundaries(
    input_path: Path,
    sample_rate: int,
    max_seconds: int,
    silence_search_seconds: int,
    silence_dbfs: float,
) -> list[float]:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV。请运行 `.venv/bin/python -m pip install av` 后重试。") from exc

    boundaries: list[float] = []
    total_samples = 0
    target_seconds = float(max_seconds)
    candidate_seconds: float | None = None
    quietest_seconds: float | None = None
    quietest_dbfs = 0.0
    min_gap_seconds = max(30.0, max_seconds * 0.5)

    with av.open(str(input_path)) as source:
        resampler = AudioResampler(format="s16p", layout="mono", rate=sample_rate)
        for frame in source.decode(audio=0):
            for resampled_frame in resampler.resample(frame):
                total_samples += resampled_frame.samples
                current_seconds = total_samples / sample_rate
                search_start = max(0.0, target_seconds - silence_search_seconds)

                if search_start <= current_seconds <= target_seconds:
                    current_dbfs = frame_dbfs(resampled_frame)
                    if quietest_seconds is None or current_dbfs < quietest_dbfs:
                        quietest_seconds = current_seconds
                        quietest_dbfs = current_dbfs
                    if current_dbfs <= silence_dbfs:
                        candidate_seconds = current_seconds

                if current_seconds >= target_seconds:
                    boundary_seconds = candidate_seconds or quietest_seconds or target_seconds
                    if not boundaries or boundary_seconds - boundaries[-1] >= min_gap_seconds:
                        boundaries.append(boundary_seconds)
                    target_seconds = boundary_seconds + max_seconds
                    candidate_seconds = None
                    quietest_seconds = None
                    quietest_dbfs = 0.0

    duration_seconds = total_samples / sample_rate
    return [boundary for boundary in boundaries if boundary < duration_seconds - 1.0]


def split_to_mp3_with_pyav(
    input_path: Path,
    work_dir: Path,
    sample_rate: int,
    bitrate: str,
    chunk_seconds: int,
    silence_search_seconds: int,
    silence_dbfs: float,
) -> list[AudioChunk]:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV。请运行 `.venv/bin/python -m pip install av` 后重试。") from exc

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    bitrate_bps = parse_bitrate(bitrate)
    boundaries = find_silence_aware_boundaries(
        input_path,
        sample_rate,
        chunk_seconds,
        silence_search_seconds,
        silence_dbfs,
    )
    boundary_index = 0

    paths: list[tuple[Path, float, float]] = []
    target = None
    stream = None
    chunk_index = 0
    chunk_samples = 0
    total_samples = 0

    def open_writer() -> None:
        nonlocal target, stream, chunk_index, chunk_samples
        chunk_index += 1
        chunk_samples = 0
        chunk_path = chunks_dir / f"{input_path.stem}_part{chunk_index:03d}.mp3"
        target = av.open(str(chunk_path), "w", format="mp3")
        stream = target.add_stream("mp3", rate=sample_rate)
        stream.bit_rate = bitrate_bps
        stream.codec_context.layout = "mono"
        paths.append((chunk_path, total_samples / sample_rate, total_samples / sample_rate))

    def close_writer() -> None:
        nonlocal target, stream, paths
        if target is None or stream is None:
            return
        for packet in stream.encode(None):
            target.mux(packet)
        target.close()
        chunk_path, start_seconds, _ = paths[-1]
        paths[-1] = (chunk_path, start_seconds, total_samples / sample_rate)
        target = None
        stream = None

    try:
        with av.open(str(input_path)) as source:
            resampler = AudioResampler(format="s16p", layout="mono", rate=sample_rate)
            for frame in source.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    if target is None or stream is None:
                        open_writer()
                    elif boundary_index < len(boundaries) and total_samples / sample_rate >= boundaries[boundary_index]:
                        close_writer()
                        boundary_index += 1
                        open_writer()

                    for packet in stream.encode(resampled_frame):
                        target.mux(packet)
                    chunk_samples += resampled_frame.samples
                    total_samples += resampled_frame.samples
            close_writer()
    except Exception as exc:
        try:
            close_writer()
        except Exception:
            pass
        raise RuntimeError(f"PyAV MP3 分段压缩失败：{exc}") from exc

    if not paths:
        raise RuntimeError("PyAV MP3 分段压缩失败：未生成任何分段。")

    chunks = []
    total = len(paths)
    for index, (path, start_seconds, end_seconds) in enumerate(paths, start=1):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"PyAV MP3 分段压缩失败：第 {index} 段为空。")
        if not is_within_api_limit(path):
            raise RuntimeError(
                f"第 {index} 段 Base64 后仍超过 10MB，请调小 --max-single-seconds 或 --mp3-bitrate。"
            )
        chunks.append(
            AudioChunk(
                path=path,
                mime_type=SUPPORTED_INPUTS[".mp3"],
                index=index,
                total=total,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return chunks


def safe_wav_chunk_seconds(sample_rate: int) -> int:
    bytes_per_second = sample_rate * 2
    return max(30, int((SAFE_AUDIO_BYTES - 4096) // bytes_per_second))


def split_to_wav_with_pyav(
    input_path: Path,
    work_dir: Path,
    sample_rate: int,
    max_seconds: int,
    silence_search_seconds: int,
    silence_dbfs: float,
) -> list[AudioChunk]:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV。请运行 `.venv/bin/python -m pip install av` 后重试。") from exc

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_seconds = min(max_seconds, safe_wav_chunk_seconds(sample_rate))
    boundaries = find_silence_aware_boundaries(
        input_path,
        sample_rate,
        chunk_seconds,
        silence_search_seconds,
        silence_dbfs,
    )
    boundary_index = 0

    paths: list[tuple[Path, float, float]] = []
    target = None
    stream = None
    chunk_index = 0
    total_samples = 0

    def open_writer() -> None:
        nonlocal target, stream, chunk_index
        chunk_index += 1
        chunk_path = chunks_dir / f"{input_path.stem}_part{chunk_index:03d}.wav"
        target = av.open(str(chunk_path), "w", format="wav")
        stream = target.add_stream("pcm_s16le", rate=sample_rate)
        stream.codec_context.layout = "mono"
        paths.append((chunk_path, total_samples / sample_rate, total_samples / sample_rate))

    def close_writer() -> None:
        nonlocal target, stream, paths
        if target is None or stream is None:
            return
        for packet in stream.encode(None):
            target.mux(packet)
        target.close()
        chunk_path, start_seconds, _ = paths[-1]
        paths[-1] = (chunk_path, start_seconds, total_samples / sample_rate)
        target = None
        stream = None

    try:
        with av.open(str(input_path)) as source:
            resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
            for frame in source.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    if target is None or stream is None:
                        open_writer()
                    elif boundary_index < len(boundaries) and total_samples / sample_rate >= boundaries[boundary_index]:
                        close_writer()
                        boundary_index += 1
                        open_writer()

                    for packet in stream.encode(resampled_frame):
                        target.mux(packet)
                    total_samples += resampled_frame.samples
            close_writer()
    except Exception as exc:
        try:
            close_writer()
        except Exception:
            pass
        raise RuntimeError(f"PyAV WAV 分段失败：{exc}") from exc

    if not paths:
        raise RuntimeError("PyAV WAV 分段失败：未生成任何分段。")

    chunks = []
    total = len(paths)
    for index, (path, start_seconds, end_seconds) in enumerate(paths, start=1):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"PyAV WAV 分段失败：第 {index} 段为空。")
        if not is_within_api_limit(path):
            raise RuntimeError(
                f"第 {index} 段 WAV Base64 后仍超过 10MB；请调低 --sample-rate 或 --max-single-seconds。"
            )
        chunks.append(
            AudioChunk(
                path=path,
                mime_type=SUPPORTED_INPUTS[".wav"],
                index=index,
                total=total,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return chunks


def parse_bitrate(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.endswith("kbps"):
        return int(float(normalized[:-4]) * 1000)
    if normalized.endswith("k"):
        return int(float(normalized[:-1]) * 1000)
    return int(normalized)


def compress_to_mp3(input_path: Path, work_dir: Path, sample_rate: int, bitrate: str) -> Path:
    output_path = work_dir / f"{input_path.stem}_{bitrate}.mp3"
    errors: list[str] = []

    try:
        return compress_to_mp3_with_pyav(input_path, output_path, sample_rate, bitrate)
    except RuntimeError as exc:
        errors.append(str(exc))

    try:
        return compress_to_mp3_with_ffmpeg(input_path, output_path, sample_rate, bitrate)
    except RuntimeError as exc:
        errors.append(str(exc))

    raise RuntimeError("无法压缩为 MP3；" + "；".join(errors))


def split_with_ffmpeg(input_path: Path, work_dir: Path, sample_rate: int, chunk_seconds: int) -> list[AudioChunk]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg。")

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = chunks_dir / f"{input_path.stem}_part%03d.wav"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 音频切分失败：{details or exc}") from exc

    paths = sorted(chunks_dir.glob(f"{input_path.stem}_part*.wav"))
    if not paths:
        raise RuntimeError("ffmpeg 音频切分失败：未生成任何 wav 分段。")

    chunks: list[AudioChunk] = []
    for index, path in enumerate(paths, start=1):
        if not is_within_api_limit(path):
            raise RuntimeError(
                f"切分后的第 {index} 段仍超过接口限制，请调小 --chunk-seconds 或 --sample-rate。"
            )
        start_seconds = (index - 1) * chunk_seconds
        chunks.append(
            AudioChunk(
                path=path,
                mime_type=SUPPORTED_INPUTS[".wav"],
                index=index,
                total=len(paths),
                start_seconds=start_seconds,
                end_seconds=start_seconds + duration_for_wav(path),
            )
        )

    return chunks


def ensure_supported_audio(input_path: Path, work_dir: Path, sample_rate: int) -> tuple[Path, str]:
    suffix = input_path.suffix.lower()
    if suffix in SUPPORTED_INPUTS:
        return input_path, SUPPORTED_INPUTS[suffix]

    converted_path = convert_to_wav(input_path, work_dir, sample_rate)
    return converted_path, SUPPORTED_INPUTS[".wav"]


def base64_size_for_file(audio_path: Path) -> int:
    raw_size = audio_path.stat().st_size
    return 4 * ((raw_size + 2) // 3)


def is_within_api_limit(audio_path: Path) -> bool:
    return base64_size_for_file(audio_path) <= MAX_BASE64_BYTES


def split_wav(wav_path: Path, work_dir: Path, chunk_seconds: int) -> list[AudioChunk]:
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(wav_path), "rb") as source:
        params = source.getparams()
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        bytes_per_frame = source.getnchannels() * source.getsampwidth()
        max_frames_by_size = max(1, SAFE_AUDIO_BYTES // bytes_per_frame)
        max_frames_by_time = max(1, frame_rate * chunk_seconds)
        frames_per_chunk = min(max_frames_by_size, max_frames_by_time)
        total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        chunks: list[AudioChunk] = []
        for index in range(total_chunks):
            start_frame = index * frames_per_chunk
            source.setpos(start_frame)
            frames_to_read = min(frames_per_chunk, total_frames - start_frame)
            chunk_path = chunks_dir / f"{wav_path.stem}_part{index + 1:03d}.wav"

            with wave.open(str(chunk_path), "wb") as target:
                target.setparams(params)
                target.writeframes(source.readframes(frames_to_read))

            if not is_within_api_limit(chunk_path):
                raise RuntimeError(
                    f"切分后的第 {index + 1} 段仍超过接口限制，请调小 --chunk-seconds 或 --sample-rate。"
                )

            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    mime_type=SUPPORTED_INPUTS[".wav"],
                    index=index + 1,
                    total=total_chunks,
                    start_seconds=start_frame / frame_rate,
                    end_seconds=(start_frame + frames_to_read) / frame_rate,
                )
            )

    return chunks


def split_wav_chunk_for_retry(chunk: AudioChunk, work_dir: Path, chunk_seconds: int) -> list[AudioChunk]:
    if chunk.mime_type != SUPPORTED_INPUTS[".wav"]:
        raise RuntimeError("当前问题段不是 WAV，无法自动二次切分；请使用 --strategy wav 后重试。")

    retry_dir = work_dir / f"retry_part{chunk.index:03d}"
    retry_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(chunk.path), "rb") as source:
        params = source.getparams()
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        frames_per_chunk = max(1, frame_rate * chunk_seconds)
        total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        if total_chunks <= 1:
            raise RuntimeError("当前问题段已经短于二次切分长度，无法继续自动缩短。")

        chunks: list[AudioChunk] = []
        for sub_index in range(total_chunks):
            start_frame = sub_index * frames_per_chunk
            source.setpos(start_frame)
            frames_to_read = min(frames_per_chunk, total_frames - start_frame)
            chunk_path = retry_dir / f"{chunk.path.stem}_retry{sub_index + 1:02d}.wav"

            with wave.open(str(chunk_path), "wb") as target:
                target.setparams(params)
                target.writeframes(source.readframes(frames_to_read))

            if not is_within_api_limit(chunk_path):
                raise RuntimeError(
                    f"第 {chunk.index} 段二次切分后的第 {sub_index + 1} 小段仍超过接口限制。"
                )

            start_seconds = chunk.start_seconds + start_frame / frame_rate
            end_seconds = chunk.start_seconds + (start_frame + frames_to_read) / frame_rate
            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    mime_type=SUPPORTED_INPUTS[".wav"],
                    index=chunk.index * 1000 + sub_index + 1,
                    total=chunk.total,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    label=f"{chunk.index}.{sub_index + 1}/{chunk.total}",
                )
            )

    return chunks


def duration_for_wav(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as source:
        return source.getnframes() / source.getframerate()


def duration_for_media(media_path: Path) -> float:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("未安装 PyAV，无法检测音频时长。") from exc

    with av.open(str(media_path)) as source:
        stream = source.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if source.duration is not None:
            return source.duration / 1_000_000
    raise RuntimeError("无法检测音频时长。")


def prepare_audio_chunks(
    input_path: Path,
    work_dir: Path,
    sample_rate: int,
    chunk_seconds: int,
    strategy: str,
    mp3_bitrate: str,
    max_single_seconds: int,
    silence_search_seconds: int,
    silence_dbfs: float,
) -> list[AudioChunk]:
    suffix = input_path.suffix.lower()
    compression_error = ""
    if strategy == "wav":
        return split_to_wav_with_pyav(
            input_path,
            work_dir,
            sample_rate,
            max_single_seconds,
            silence_search_seconds,
            silence_dbfs,
        )

    if suffix in SUPPORTED_INPUTS and is_within_api_limit(input_path):
        try:
            if duration_for_media(input_path) > max_single_seconds:
                if strategy == "auto":
                    return split_to_wav_with_pyav(
                        input_path,
                        work_dir,
                        sample_rate,
                        max_single_seconds,
                        silence_search_seconds,
                        silence_dbfs,
                    )
                return split_to_mp3_with_pyav(
                    input_path,
                    work_dir,
                    sample_rate,
                    mp3_bitrate,
                    max_single_seconds,
                    silence_search_seconds,
                    silence_dbfs,
                )
        except RuntimeError:
            pass
        return [
            AudioChunk(
                path=input_path,
                mime_type=SUPPORTED_INPUTS[suffix],
                index=1,
                total=1,
                start_seconds=0,
                end_seconds=0,
            )
        ]

    if strategy == "auto":
        try:
            return split_to_wav_with_pyav(
                input_path,
                work_dir,
                sample_rate,
                max_single_seconds,
                silence_search_seconds,
                silence_dbfs,
            )
        except RuntimeError as exc:
            compression_error = str(exc)

    if strategy in {"auto", "compress"}:
        try:
            if duration_for_media(input_path) > max_single_seconds:
                return split_to_mp3_with_pyav(
                    input_path,
                    work_dir,
                    sample_rate,
                    mp3_bitrate,
                    max_single_seconds,
                    silence_search_seconds,
                    silence_dbfs,
                )
        except RuntimeError as exc:
            compression_error = str(exc)

    if strategy in {"auto", "compress"}:
        try:
            compressed_path = compress_to_mp3(input_path, work_dir, sample_rate, mp3_bitrate)
            if is_within_api_limit(compressed_path):
                compressed_duration = duration_for_media(compressed_path)
                if compressed_duration > max_single_seconds:
                    return split_to_mp3_with_pyav(
                        input_path,
                        work_dir,
                        sample_rate,
                        mp3_bitrate,
                        max_single_seconds,
                        silence_search_seconds,
                        silence_dbfs,
                    )
                return [
                    AudioChunk(
                        path=compressed_path,
                        mime_type=SUPPORTED_INPUTS[".mp3"],
                        index=1,
                        total=1,
                        start_seconds=0,
                        end_seconds=0,
                    )
                ]
            compression_error = (
                f"压缩后的 MP3 Base64 大小为 {base64_size_for_file(compressed_path) / 1024 / 1024:.2f}MB，"
                "仍超过 10MB。"
            )
        except RuntimeError as exc:
            compression_error = str(exc)

        if strategy == "compress":
            raise RuntimeError(f"{compression_error} 请调低 --mp3-bitrate，或使用 --strategy split。")

    if strategy in {"auto", "split"} and shutil.which("ffmpeg"):
        return split_with_ffmpeg(input_path, work_dir, sample_rate, chunk_seconds)

    audio_path, mime_type = ensure_supported_audio(input_path, work_dir, sample_rate)
    if is_within_api_limit(audio_path):
        return [
            AudioChunk(
                path=audio_path,
                mime_type=mime_type,
                index=1,
                total=1,
                start_seconds=0,
                end_seconds=0,
            )
        ]

    if audio_path.suffix.lower() != ".wav":
        audio_path = convert_to_wav(audio_path, work_dir, sample_rate)

    try:
        return split_wav(audio_path, work_dir, chunk_seconds)
    except RuntimeError as exc:
        if compression_error:
            raise RuntimeError(f"{compression_error} 后备切片也失败：{exc}") from exc
        raise


def encode_audio(audio_path: Path) -> str:
    audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    encoded_size = len(audio_base64.encode("utf-8"))
    if encoded_size > MAX_BASE64_BYTES:
        raise RuntimeError(
            f"Base64 后音频大小为 {encoded_size / 1024 / 1024:.2f}MB，超过接口 10MB 限制。"
        )
    return audio_base64


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(extract_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        if isinstance(value.get("transcript"), str):
            return value["transcript"]
    return ""


def collect_response_text(value: Any) -> list[str]:
    text_keys = {"content", "text", "transcript", "result", "output_text"}
    skip_keys = {
        "id",
        "object",
        "created",
        "model",
        "role",
        "type",
        "finish_reason",
        "index",
        "usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }

    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(collect_response_text(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for key, item in value.items():
            if key in text_keys:
                extracted = extract_text(item)
                if extracted.strip():
                    texts.append(extracted)
                else:
                    texts.extend(collect_response_text(item))
            elif key not in skip_keys:
                texts.extend(collect_response_text(item))
        return texts
    return []


def build_payload(
    audio_path: Path,
    mime_type: str,
    language: str,
    stream: bool,
    max_tokens: int,
) -> dict[str, Any]:
    audio_base64 = encode_audio(audio_path)
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{mime_type};base64,{audio_base64}",
                        },
                    }
                ],
            }
        ],
        "asr_options": {"language": language},
        "max_tokens": max_tokens,
        "stream": stream,
    }


def create_request(api_key: str, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "mimo-asr-script/1.0",
        },
        method="POST",
    )


def parse_non_stream_response(response_body: str) -> str:
    event = json.loads(response_body)
    texts = collect_response_text(event.get("choices", event))
    return "".join(texts).strip()


def iter_stream_events(response: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue

        if line.startswith("data:"):
            line = line.removeprefix("data:").strip()
        if line == "[DONE]":
            break

        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def parse_stream_response(response: Any) -> str:
    fragments: list[str] = []
    for event in iter_stream_events(response):
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            message = choice.get("message", {})
            content = extract_text(delta.get("content")) or extract_text(message.get("content"))
            if content:
                fragments.append(content)
            else:
                fragments.extend(collect_response_text(choice))
    return "".join(fragments).strip()


def detect_repeated_transcript(text: str) -> str | None:
    normalized = re.sub(r"\s+", "", text)
    if len(normalized) < 240:
        return None

    sentences = [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", normalized)
        if len(part.strip()) >= 10
    ]
    if len(sentences) < 8:
        return None

    counts: dict[str, int] = {}
    for sentence in sentences:
        counts[sentence] = counts.get(sentence, 0) + 1
        if counts[sentence] >= 5:
            return f"检测到同一句话重复 {counts[sentence]} 次：{sentence[:40]}..."

    for width in range(2, 6):
        if len(sentences) < width * 3:
            continue
        sequence_counts: dict[tuple[str, ...], int] = {}
        for index in range(0, len(sentences) - width + 1):
            sequence = tuple(sentences[index : index + width])
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
            if sequence_counts[sequence] >= 3:
                preview = "。".join(sequence)[:60]
                return f"检测到连续句组重复 {sequence_counts[sequence]} 次：{preview}..."

    return None


def transcribe(audio_path: Path, mime_type: str, language: str, stream: bool, max_tokens: int) -> str:
    api_key = os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 MIMO_API_KEY，或在当前目录创建包含 MIMO_API_KEY=... 的 .env 文件。")

    request = create_request(api_key, build_payload(audio_path, mime_type, language, stream, max_tokens))
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if stream:
                transcript = parse_stream_response(response)
            else:
                response_body = response.read().decode("utf-8", errors="replace")
                transcript = parse_non_stream_response(response_body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise RateLimitError("接口返回 429 Too many requests。") from exc
        if exc.code == 400 and "context" in body.lower() and "8192" in body:
            raise RuntimeError(
                "接口返回 context 超限。请调小 --max-single-seconds，例如："
                "`--max-single-seconds 900`；如果仍超限可继续调到 600。"
            ) from exc
        raise RuntimeError(f"接口返回 HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 Mimo API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回内容不是有效 JSON: {exc}") from exc

    if not transcript:
        raise RuntimeError("接口调用完成，但没有从返回 JSON 中解析到转写文本；可尝试加 --stream 使用流式解析。")
    repetition_reason = detect_repeated_transcript(transcript)
    if repetition_reason:
        raise TranscriptionQualityError(
            f"模型输出疑似异常重复：{repetition_reason} "
            "将尝试自动二次切分当前音频段。"
        )
    return transcript


def chunk_display_name(chunk: AudioChunk) -> str:
    return chunk.label or f"{chunk.index}/{chunk.total}"


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_segment_turn_time_ranges(segment: MarkdownSegment, turns: list[AnnotatedTurn]) -> list[str]:
    segment_start = int(round(segment.start_seconds))
    segment_end = int(round(segment.end_seconds))
    cursor = segment_start
    ranges: list[str] = []

    for index, turn in enumerate(turns):
        remaining = len(turns) - index
        if segment_end - cursor >= remaining:
            latest_start = segment_end - remaining
            start_value = max(cursor, int(math.floor(turn.start_seconds)))
            start_value = min(start_value, latest_start)
            max_end = segment_end - (remaining - 1)
            end_value = max(start_value + 1, int(math.ceil(turn.end_seconds)))
            end_value = min(end_value, max_end)
        else:
            start_value = min(cursor, segment_end)
            end_value = max(start_value, min(segment_end, int(math.ceil(turn.end_seconds))))

        ranges.append(f"{format_timestamp(start_value)}-{format_timestamp(end_value)}")
        cursor = end_value
    return ranges


def chunk_cache_path(
    cache_dir: Path,
    source_path: Path,
    chunk: AudioChunk,
    language: str,
    stream: bool,
    max_tokens: int,
) -> Path:
    chunk_size = chunk.path.stat().st_size if chunk.path.exists() else 0
    cache_key = hashlib.sha256(
        f"{source_path.resolve()}:{source_path.stat().st_mtime_ns}:{chunk.index}:"
        f"{chunk.start_seconds:.3f}:{chunk.end_seconds:.3f}:{chunk_size}:"
        f"{MODEL}:{language}:{stream}:{max_tokens}".encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{source_path.stem}_part{chunk.index:03d}_{cache_key}.md"


def chunk_retry_marker_path(
    cache_dir: Path,
    source_path: Path,
    chunk: AudioChunk,
    language: str,
    stream: bool,
    max_tokens: int,
) -> Path:
    return chunk_cache_path(cache_dir, source_path, chunk, language, stream, max_tokens).with_suffix(".retry")


def transcribe_with_retries(
    chunk: AudioChunk,
    language: str,
    stream: bool,
    max_tokens: int,
    retry_attempts: int,
    retry_wait_seconds: int,
) -> str:
    attempt = 1
    while True:
        try:
            return transcribe(chunk.path, chunk.mime_type, language, stream, max_tokens)
        except RateLimitError:
            if attempt >= retry_attempts:
                raise RuntimeError(
                    f"第 {chunk.index}/{chunk.total} 段连续 {retry_attempts} 次触发 429。"
                    f"请稍后重试；已成功的分段会从缓存跳过。"
                )
            wait_seconds = retry_wait_seconds * attempt
            print(
                f"第 {chunk.index}/{chunk.total} 段触发 429，等待 {wait_seconds} 秒后重试 "
                f"({attempt + 1}/{retry_attempts})...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
            attempt += 1


def next_retry_chunk_seconds(duration_seconds: float, preferred_seconds: int, min_seconds: int) -> int:
    if duration_seconds - preferred_seconds >= min_seconds:
        target_seconds = preferred_seconds
    else:
        target_seconds = max(min_seconds, math.floor(duration_seconds / 2))
    if target_seconds >= duration_seconds - 1:
        target_seconds = max(min_seconds, math.floor(duration_seconds / 2))
    return int(target_seconds)


def transcribe_chunk_resumable(
    chunk: AudioChunk,
    source_path: Path,
    language: str,
    stream: bool,
    max_tokens: int,
    cache_dir: Path,
    retry_attempts: int,
    retry_wait_seconds: int,
    between_requests_seconds: int,
    retry_work_dir: Path,
    auto_subdivide: bool,
    subdivide_seconds: int,
    min_subdivide_seconds: int,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    retry_marker_path = chunk_retry_marker_path(cache_dir, source_path, chunk, language, stream, max_tokens)
    quality_error: TranscriptionQualityError | None = None

    if auto_subdivide and retry_marker_path.exists():
        quality_error = TranscriptionQualityError("此前该段已触发异常重复，直接自动二次切分。")
    else:
        try:
            return transcribe_with_retries(
                chunk,
                language,
                stream,
                max_tokens,
                retry_attempts,
                retry_wait_seconds,
            )
        except TranscriptionQualityError as exc:
            if not auto_subdivide:
                raise
            quality_error = exc
            retry_marker_path.write_text(str(exc) + "\n", encoding="utf-8")

    if quality_error is None:
        quality_error = TranscriptionQualityError("当前段需要自动二次切分。")

    duration_seconds = max(0.0, chunk.end_seconds - chunk.start_seconds)
    target_seconds = next_retry_chunk_seconds(
        duration_seconds,
        subdivide_seconds,
        min_subdivide_seconds,
    )
    if target_seconds < min_subdivide_seconds or target_seconds >= duration_seconds - 1:
        raise RuntimeError(
            f"{quality_error} 当前段约 {duration_seconds:.1f} 秒，已无法安全继续自动切短。"
        ) from quality_error

    print(
        f"第 {chunk_display_name(chunk)} 段输出异常，自动切成约 {target_seconds} 秒小段后续写...",
        file=sys.stderr,
    )
    subchunks = split_wav_chunk_for_retry(chunk, retry_work_dir, target_seconds)
    texts: list[str] = []
    for sub_position, subchunk in enumerate(subchunks, start=1):
        sub_cache_path = chunk_cache_path(cache_dir, source_path, subchunk, language, stream, max_tokens)
        if sub_cache_path.exists():
            transcript = sub_cache_path.read_text(encoding="utf-8").strip()
            print(f"读取缓存：第 {chunk_display_name(subchunk)} 段。", file=sys.stderr)
            texts.append(transcript)
            continue

        if sub_position > 1 and between_requests_seconds > 0:
            print(f"等待 {between_requests_seconds} 秒后继续下一小段...", file=sys.stderr)
            time.sleep(between_requests_seconds)

        print(
            f"正在转写第 {chunk_display_name(subchunk)} 段 "
            f"({format_timestamp(subchunk.start_seconds)}-{format_timestamp(subchunk.end_seconds)})...",
            file=sys.stderr,
        )
        transcript = transcribe_chunk_resumable(
            subchunk,
            source_path,
            language,
            stream,
            max_tokens,
            cache_dir,
            retry_attempts,
            retry_wait_seconds,
            between_requests_seconds,
            retry_work_dir,
            auto_subdivide,
            max(min_subdivide_seconds, target_seconds // 2),
            min_subdivide_seconds,
        )
        sub_cache_path.write_text(transcript + "\n", encoding="utf-8")
        texts.append(transcript)

    return "\n\n".join(texts).strip()


def transcribe_chunks(
    chunks: list[AudioChunk],
    source_path: Path,
    language: str,
    stream: bool,
    max_tokens: int,
    cache_dir: Path,
    retry_attempts: int,
    retry_wait_seconds: int,
    between_requests_seconds: int,
    retry_work_dir: Path,
    auto_subdivide: bool,
    subdivide_seconds: int,
    min_subdivide_seconds: int,
) -> list[tuple[AudioChunk, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[AudioChunk, str]] = []
    for position, chunk in enumerate(chunks, start=1):
        cache_path = chunk_cache_path(cache_dir, source_path, chunk, language, stream, max_tokens)
        if cache_path.exists():
            transcript = cache_path.read_text(encoding="utf-8").strip()
            print(f"读取缓存：第 {chunk_display_name(chunk)} 段。", file=sys.stderr)
            results.append((chunk, transcript))
            continue

        if position > 1 and between_requests_seconds > 0:
            print(f"等待 {between_requests_seconds} 秒后继续下一段...", file=sys.stderr)
            time.sleep(between_requests_seconds)

        if chunk.total > 1:
            print(
                f"正在转写第 {chunk_display_name(chunk)} 段 "
                f"({format_timestamp(chunk.start_seconds)}-{format_timestamp(chunk.end_seconds)})...",
                file=sys.stderr,
            )
        transcript = transcribe_chunk_resumable(
            chunk,
            source_path,
            language,
            stream,
            max_tokens,
            cache_dir,
            retry_attempts,
            retry_wait_seconds,
            between_requests_seconds,
            retry_work_dir,
            auto_subdivide,
            subdivide_seconds,
            min_subdivide_seconds,
        )
        cache_path.write_text(transcript + "\n", encoding="utf-8")
        results.append((chunk, transcript))
    return results


def write_markdown(output_path: Path, source_path: Path, results: list[tuple[AudioChunk, str]]) -> None:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    transcript = "\n\n".join(text for _, text in results)
    chunk_count = len(results)
    content = (
        f"# {source_path.stem} 转写\n\n"
        f"- 源文件：`{source_path.name}`\n"
        f"- 模型：`{MODEL}`\n"
        f"- 分段数：`{chunk_count}`\n"
        f"- 生成时间：`{created_at}`\n\n"
        "## 转写内容\n\n"
        f"{transcript}\n"
    )

    if chunk_count > 1:
        sections = []
        for chunk, text in results:
            sections.append(
                f"### 第 {chunk.index} 段 "
                f"({format_timestamp(chunk.start_seconds)}-{format_timestamp(chunk.end_seconds)})\n\n"
                f"{text}"
            )
        content = (
            f"# {source_path.stem} 转写\n\n"
            f"- 源文件：`{source_path.name}`\n"
            f"- 模型：`{MODEL}`\n"
            f"- 分段数：`{chunk_count}`\n"
            f"- 生成时间：`{created_at}`\n\n"
            "## 转写内容\n\n"
            + "\n\n".join(sections)
            + "\n"
        )

    output_path.write_text(content, encoding="utf-8")


def parse_timestamp_to_seconds(value: str) -> float:
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"无法解析时间戳：{value}")


def default_annotated_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_annotated{input_path.suffix}")


def default_diarized_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_diarized{input_path.suffix}")


def parse_transcript_markdown(markdown_path: Path) -> tuple[str, list[MarkdownSegment]]:
    content = markdown_path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else markdown_path.stem
    heading_pattern = re.compile(
        r"^###\s+第\s+(\d+)\s+段\s+\((\d{2}:\d{2}(?::\d{2})?)-(\d{2}:\d{2}(?::\d{2})?)\)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(content))
    if not matches:
        raise RuntimeError("未在 Markdown 中找到形如 `### 第 N 段 (00:00-01:30)` 的分段标题。")

    segments: list[MarkdownSegment] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[match.end() : next_start].strip()
        segments.append(
            MarkdownSegment(
                index=int(match.group(1)),
                start_seconds=parse_timestamp_to_seconds(match.group(2)),
                end_seconds=parse_timestamp_to_seconds(match.group(3)),
                text=body,
            )
        )
    return title, segments


def normalize_for_turn_detection(text: str) -> str:
    return re.sub(r"[\s，,。.!！?？；;、：:]+", "", text).lower()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [sentence.strip() for sentence in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)]


SHORT_REPLY_WORDS = {
    "嗯",
    "嗯嗯",
    "对",
    "对对",
    "对对对",
    "是",
    "是的",
    "好",
    "好的",
    "可以",
    "明白",
    "明白明白",
    "ok",
    "okay",
    "okok",
    "啊",
    "哦",
    "行",
}


def is_short_reply(sentence: str) -> bool:
    normalized = normalize_for_turn_detection(sentence)
    if normalized in SHORT_REPLY_WORDS:
        return True
    if len(normalized) <= 6 and any(word in normalized for word in SHORT_REPLY_WORDS):
        return True
    return False


def is_question_like(sentence: str) -> bool:
    if not sentence.endswith(("？", "?")):
        return False
    normalized = normalize_for_turn_detection(sentence)
    if normalized in {"对吧", "是吧", "嗯"}:
        return False
    cues = ("您", "你们", "贵司", "能不能", "有没有", "是什么", "为什么", "哪些", "怎么", "吗")
    return len(normalized) <= 90 and any(cue in sentence for cue in cues)


def next_speaker(speaker: int, max_speakers: int = 4) -> int:
    if speaker == 1:
        return 2
    return 1 if max_speakers <= 2 else 1


def make_turns_without_timestamps(segment: MarkdownSegment, state: SpeakerState) -> list[AnnotatedTurn]:
    sentences = split_sentences(segment.text)
    turns: list[AnnotatedTurn] = []
    buffer: list[str] = []
    buffer_speaker = state.current_speaker

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        turns.append(
            AnnotatedTurn(
                segment_index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                speaker=buffer_speaker,
                text="".join(buffer).strip(),
            )
        )
        buffer = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if is_short_reply(sentence):
            flush_buffer()
            reply_speaker = next_speaker(state.current_speaker, state.max_speakers)
            turns.append(
                AnnotatedTurn(
                    segment_index=segment.index,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    speaker=reply_speaker,
                    text=sentence,
                )
            )
            state.pending_question_switch = False
            continue

        if state.pending_question_switch and not buffer:
            state.current_speaker = next_speaker(state.current_speaker, state.max_speakers)
            buffer_speaker = state.current_speaker
            state.pending_question_switch = False

        if not buffer:
            buffer_speaker = state.current_speaker
        buffer.append(sentence)

        if is_question_like(sentence):
            flush_buffer()
            state.pending_question_switch = True

    flush_buffer()
    if turns:
        long_turns = [turn for turn in turns if not is_short_reply(turn.text)]
        if long_turns:
            state.current_speaker = long_turns[-1].speaker
    return turns


def assign_turn_timestamps(segment: MarkdownSegment, turns: list[AnnotatedTurn]) -> list[AnnotatedTurn]:
    if not turns:
        return []
    duration = max(0.0, segment.end_seconds - segment.start_seconds)
    weights = [max(3, len(normalize_for_turn_detection(turn.text))) for turn in turns]
    total_weight = sum(weights) or len(turns)
    elapsed_weight = 0
    timed_turns: list[AnnotatedTurn] = []

    for index, (turn, weight) in enumerate(zip(turns, weights)):
        start_seconds = segment.start_seconds + duration * elapsed_weight / total_weight
        elapsed_weight += weight
        if index == len(turns) - 1:
            end_seconds = segment.end_seconds
        else:
            end_seconds = segment.start_seconds + duration * elapsed_weight / total_weight
        timed_turns.append(
            AnnotatedTurn(
                segment_index=turn.segment_index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                speaker=turn.speaker,
                text=turn.text,
            )
        )
    return timed_turns


def annotate_segments(segments: list[MarkdownSegment]) -> list[AnnotatedTurn]:
    state = SpeakerState()
    annotated_turns: list[AnnotatedTurn] = []
    for segment in segments:
        raw_turns = make_turns_without_timestamps(segment, state)
        annotated_turns.extend(assign_turn_timestamps(segment, raw_turns))
    return annotated_turns


def write_annotated_markdown(
    output_path: Path,
    input_markdown_path: Path,
    title: str,
    segments: list[MarkdownSegment],
    turns: list[AnnotatedTurn],
    method_note: str | None = None,
) -> None:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    turns_by_segment: dict[int, list[AnnotatedTurn]] = {}
    for turn in turns:
        turns_by_segment.setdefault(turn.segment_index, []).append(turn)

    sections: list[str] = []
    for segment in segments:
        lines = [
            f"### 第 {segment.index} 段 "
            f"({format_timestamp(segment.start_seconds)}-{format_timestamp(segment.end_seconds)})"
        ]
        segment_turns = turns_by_segment.get(segment.index, [])
        time_ranges = format_segment_turn_time_ranges(segment, segment_turns)
        for turn, time_range in zip(segment_turns, time_ranges):
            lines.append(
                f"[{time_range}] 说话人 {turn.speaker}：{turn.text}"
            )
        sections.append("\n\n".join(lines))

    content = (
        f"# {title}（带说话人和时间戳）\n\n"
        f"- 原始 Markdown：`{input_markdown_path.name}`\n"
        f"- 分段数：`{len(segments)}`\n"
        f"- 发言轮次数：`{len(turns)}`\n"
        f"- 生成时间：`{created_at}`\n"
        f"- 标注说明：{method_note or '说话人和轮次级时间戳由轻量文本后处理估算生成，不是音频声纹级 diarization，也不是逐字精确对齐。'}\n\n"
        "## 带时间戳转写\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    output_path.write_text(content, encoding="utf-8")


def annotate_existing_markdown(input_path: Path, output_path: Path) -> None:
    title, segments = parse_transcript_markdown(input_path)
    turns = annotate_segments(segments)
    write_annotated_markdown(output_path, input_path, title, segments, turns)


def configure_local_model_caches(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["PYANNOTE_CACHE"] = str(cache_dir / "torch" / "pyannote")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "matplotlib")
    for key in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "PYANNOTE_CACHE",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
    ):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def diarization_cache_path(
    cache_dir: Path,
    audio_path: Path,
    model_name: str,
    sample_rate: int,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Path:
    stat = audio_path.stat()
    cache_key = hashlib.sha256(
        f"{audio_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
        f"{model_name}:{sample_rate}:{num_speakers}:{min_speakers}:{max_speakers}".encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"{audio_path.stem}_{cache_key}.json"


def cleanup_old_diarization_cache(cache_dir: Path, ttl_days: int) -> None:
    if ttl_days <= 0 or not cache_dir.exists():
        return
    cutoff = time.time() - (ttl_days * 24 * 60 * 60)
    removed_count = 0
    for cache_path in cache_dir.glob("*.json"):
        try:
            if cache_path.stat().st_mtime < cutoff:
                cache_path.unlink()
                removed_count += 1
        except OSError:
            continue
    if removed_count:
        print(f"已清理 {removed_count} 个超过 {ttl_days} 天的 pyannote 缓存文件。", file=sys.stderr)


def load_diarization_segments(cache_path: Path) -> list[DiarizationSegment]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data)
    segments: list[DiarizationSegment] = []
    for item in raw_segments:
        segments.append(
            DiarizationSegment(
                start_seconds=float(item["start"]),
                end_seconds=float(item["end"]),
                speaker=int(item["speaker"]),
            )
        )
    return segments


def write_diarization_segments(
    cache_path: Path,
    audio_path: Path,
    model_name: str,
    segments: list[DiarizationSegment],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": audio_path.name,
        "model": model_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "segments": [
            {
                "start": round(segment.start_seconds, 3),
                "end": round(segment.end_seconds, 3),
                "speaker": segment.speaker,
            }
            for segment in segments
        ],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def huggingface_token() -> str | None:
    return (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )


def allow_pyannote_torch_checkpoint_globals() -> None:
    try:
        import torch.serialization
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from torch.torch_version import TorchVersion
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([TorchVersion, Specifications, Problem, Resolution])


def load_pyannote_pipeline(model_name: str, token: str | None, cache_dir: Path) -> Any:
    configure_warning_filters()
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pyannote.audio。请先运行 `.venv/bin/python -m pip install -r requirements.txt`。"
        ) from exc

    kwargs: dict[str, str | Path] = {}
    parameters = inspect.signature(Pipeline.from_pretrained).parameters
    if token:
        if "use_auth_token" in parameters:
            kwargs["use_auth_token"] = token
        elif "token" in parameters:
            kwargs["token"] = token
        else:
            os.environ.setdefault("HF_TOKEN", token)
    if "cache_dir" in parameters:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = cache_dir

    allow_pyannote_torch_checkpoint_globals()
    pipeline = Pipeline.from_pretrained(model_name, **kwargs)

    if pipeline is None:
        raise RuntimeError(
            "pyannote 模型加载失败。请确认已在 Hugging Face 接受模型授权，并在 .env 中设置 "
            "HF_TOKEN=... 或 HUGGINGFACE_TOKEN=...。"
        )
    return pipeline


def move_pyannote_pipeline_to_device(pipeline: Any, device_name: str) -> None:
    if device_name == "cpu":
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 torch，无法设置 pyannote 推理设备。") from exc

    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    if device_name != "cpu":
        pipeline.to(torch.device(device_name))


def run_pyannote_diarization(
    audio_path: Path,
    work_dir: Path,
    model_cache_dir: Path,
    diarization_cache_dir: Path,
    model_name: str,
    sample_rate: int,
    device_name: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    segmentation_batch_size: int,
    embedding_batch_size: int,
    cache_ttl_days: int,
) -> list[DiarizationSegment]:
    configure_local_model_caches(model_cache_dir)
    cleanup_old_diarization_cache(diarization_cache_dir, cache_ttl_days)
    cache_path = diarization_cache_path(
        diarization_cache_dir,
        audio_path,
        model_name,
        sample_rate,
        num_speakers,
        min_speakers,
        max_speakers,
    )
    if cache_path.exists():
        print(f"读取 pyannote 缓存：{cache_path}", file=sys.stderr)
        return load_diarization_segments(cache_path)

    diarization_audio_path = work_dir / f"{audio_path.stem}_pyannote.wav"
    convert_to_mono_wav_with_pyav(audio_path, diarization_audio_path, sample_rate)

    pipeline = load_pyannote_pipeline(model_name, huggingface_token(), model_cache_dir / "pyannote")
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = segmentation_batch_size
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = embedding_batch_size
    move_pyannote_pipeline_to_device(pipeline, device_name)

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    print("正在运行 pyannote 说话人分离...", file=sys.stderr)
    diarization = pipeline(str(diarization_audio_path), **kwargs)
    speaker_map: dict[str, int] = {}
    segments: list[DiarizationSegment] = []
    for turn, _, raw_speaker in diarization.itertracks(yield_label=True):
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = len(speaker_map) + 1
        if turn.end <= turn.start:
            continue
        segments.append(
            DiarizationSegment(
                start_seconds=float(turn.start),
                end_seconds=float(turn.end),
                speaker=speaker_map[raw_speaker],
            )
        )

    segments.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    if not segments:
        raise RuntimeError("pyannote 未返回任何有效说话人时间段。")
    write_diarization_segments(cache_path, audio_path, model_name, segments)
    return segments


def speaker_for_time_range(
    start_seconds: float,
    end_seconds: float,
    diarization_segments: list[DiarizationSegment],
    fallback_speaker: int,
) -> int:
    if end_seconds <= start_seconds:
        end_seconds = start_seconds + 0.001
    scores: dict[int, float] = {}
    for segment in diarization_segments:
        overlap = max(
            0.0,
            min(end_seconds, segment.end_seconds) - max(start_seconds, segment.start_seconds),
        )
        if overlap > 0:
            scores[segment.speaker] = scores.get(segment.speaker, 0.0) + overlap
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]

    midpoint = (start_seconds + end_seconds) / 2
    nearest_segment = min(
        diarization_segments,
        key=lambda segment: min(abs(midpoint - segment.start_seconds), abs(midpoint - segment.end_seconds)),
        default=None,
    )
    if nearest_segment is not None:
        nearest_distance = min(
            abs(midpoint - nearest_segment.start_seconds),
            abs(midpoint - nearest_segment.end_seconds),
        )
        if nearest_distance <= 3.0:
            return nearest_segment.speaker
    return fallback_speaker


def annotate_segments_with_diarization(
    segments: list[MarkdownSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[AnnotatedTurn]:
    state = SpeakerState()
    annotated_turns: list[AnnotatedTurn] = []
    fallback_speaker = 1
    for segment in segments:
        raw_turns = make_turns_without_timestamps(segment, state)
        timed_turns = assign_turn_timestamps(segment, raw_turns)
        for turn in timed_turns:
            speaker = speaker_for_time_range(
                turn.start_seconds,
                turn.end_seconds,
                diarization_segments,
                fallback_speaker,
            )
            fallback_speaker = speaker
            annotated_turns.append(
                AnnotatedTurn(
                    segment_index=turn.segment_index,
                    start_seconds=turn.start_seconds,
                    end_seconds=turn.end_seconds,
                    speaker=speaker,
                    text=turn.text,
                )
            )
    return annotated_turns


def diarize_existing_markdown(
    input_markdown_path: Path,
    audio_path: Path,
    output_path: Path,
    model_cache_dir: Path,
    diarization_cache_dir: Path,
    model_name: str,
    sample_rate: int,
    device_name: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    segmentation_batch_size: int,
    embedding_batch_size: int,
    cache_ttl_days: int,
) -> None:
    title, segments = parse_transcript_markdown(input_markdown_path)
    with tempfile.TemporaryDirectory(prefix="mimo-pyannote-") as temp_dir:
        diarization_segments = run_pyannote_diarization(
            audio_path,
            Path(temp_dir),
            model_cache_dir,
            diarization_cache_dir,
            model_name,
            sample_rate,
            device_name,
            num_speakers,
            min_speakers,
            max_speakers,
            segmentation_batch_size,
            embedding_batch_size,
            cache_ttl_days,
        )
    turns = annotate_segments_with_diarization(segments, diarization_segments)
    write_annotated_markdown(
        output_path,
        input_markdown_path,
        title,
        segments,
        turns,
        "说话人标签由 pyannote.audio 基于原始音频做说话人分离后生成；"
        "轮次级时间戳仍由每个 Mimo 转写分段内的文本长度比例估算，"
        "不是逐字精确对齐。",
    )


def cleanup_process_files(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Mimo ASR 将音频文件转写为 Markdown。")
    parser.add_argument("audio", nargs="?", default="audio_file.m4a", help="待转写音频路径，默认 audio_file.m4a")
    parser.add_argument("-o", "--output", help="输出 Markdown 路径，默认与音频同名 .md")
    parser.add_argument(
        "--annotate-existing-md",
        metavar="MARKDOWN",
        help="直接为已有转写 Markdown 增加说话人和轮次级时间戳，不调用 ASR API。",
    )
    parser.add_argument(
        "--diarize-existing-md",
        metavar="MARKDOWN",
        help="直接用 pyannote 为已有转写 Markdown 增加音频级说话人标签，不调用 ASR API。",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="正常转写完成后，同时生成带说话人和轮次级时间戳的增强版 Markdown。",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="正常转写完成后，同时用 pyannote 基于原始音频生成说话人标签增强版 Markdown。",
    )
    parser.add_argument(
        "--annotated-output",
        help="增强版 Markdown 输出路径，默认在原文件名后追加 _annotated。",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-3.1",
        help="pyannote 说话人分离模型，默认 pyannote/speaker-diarization-3.1。",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=".model_cache",
        help="Hugging Face/Torch 模型缓存目录，默认 .model_cache，位于当前项目内。",
    )
    parser.add_argument(
        "--diarization-cache-dir",
        default=".diarization_cache",
        help="pyannote 说话人时间段缓存目录，默认 .diarization_cache。",
    )
    parser.add_argument(
        "--diarization-cache-ttl-days",
        type=int,
        default=DEFAULT_DIARIZATION_CACHE_TTL_DAYS,
        help=f"pyannote 说话人时间段缓存保留天数，默认 {DEFAULT_DIARIZATION_CACHE_TTL_DAYS}；设为 0 表示不自动清理。",
    )
    parser.add_argument(
        "--diarization-device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="auto",
        help="pyannote 推理设备，默认 auto；会优先尝试 NVIDIA CUDA，其次 Apple Silicon MPS，否则回退 cpu。",
    )
    parser.add_argument("--num-speakers", type=int, help="如果已知说话人数，可指定固定人数。")
    parser.add_argument("--min-speakers", type=int, help="说话人数下限，用于辅助 pyannote 判断。")
    parser.add_argument("--max-speakers", type=int, help="说话人数上限，用于辅助 pyannote 判断。")
    parser.add_argument(
        "--diarization-segmentation-batch-size",
        type=int,
        default=DEFAULT_DIARIZATION_BATCH_SIZE,
        help=f"pyannote 分割模型批量大小，默认 {DEFAULT_DIARIZATION_BATCH_SIZE}。",
    )
    parser.add_argument(
        "--diarization-embedding-batch-size",
        type=int,
        default=DEFAULT_DIARIZATION_BATCH_SIZE,
        help=f"pyannote 声纹嵌入模型批量大小，默认 {DEFAULT_DIARIZATION_BATCH_SIZE}。",
    )
    parser.add_argument("--language", default="auto", help="识别语言，默认 auto")
    parser.add_argument("--sample-rate", type=int, default=16000, help="转换为 wav 时使用的采样率，默认 16000")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"wav/ffmpeg 后备切片秒数，默认 {DEFAULT_CHUNK_SECONDS}。",
    )
    parser.add_argument(
        "--max-single-seconds",
        type=int,
        default=DEFAULT_MAX_SINGLE_SECONDS,
        help=f"单次提交给模型的最大音频秒数，默认 {DEFAULT_MAX_SINGLE_SECONDS}，提高长录音识别稳定性。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=f"每段转写允许的最大输出 token 数，默认 {DEFAULT_MAX_COMPLETION_TOKENS}。",
    )
    parser.add_argument(
        "--subdivide-seconds",
        type=int,
        default=DEFAULT_SUBDIVIDE_SECONDS,
        help=f"单段转写异常重复时自动二次切分的目标秒数，默认 {DEFAULT_SUBDIVIDE_SECONDS}。",
    )
    parser.add_argument(
        "--min-subdivide-seconds",
        type=int,
        default=DEFAULT_MIN_SUBDIVIDE_SECONDS,
        help=f"自动二次切分的最短秒数，默认 {DEFAULT_MIN_SUBDIVIDE_SECONDS}。",
    )
    parser.add_argument(
        "--no-auto-subdivide",
        action="store_true",
        help="关闭异常重复时的自动二次切分续写。",
    )
    parser.add_argument(
        "--silence-search-seconds",
        type=int,
        default=DEFAULT_SILENCE_SEARCH_SECONDS,
        help=f"在目标切点前寻找静音的窗口秒数，默认 {DEFAULT_SILENCE_SEARCH_SECONDS}。",
    )
    parser.add_argument(
        "--silence-dbfs",
        type=float,
        default=DEFAULT_SILENCE_DBFS,
        help=f"判定为静音/停顿的音量阈值 dBFS，默认 {DEFAULT_SILENCE_DBFS}。",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default=DEFAULT_STRATEGY,
        help="超限处理策略：wav 无损 PCM 分段；compress MP3 压缩分段；auto 优先 wav；split 使用后备切片。",
    )
    parser.add_argument("--mp3-bitrate", default=DEFAULT_MP3_BITRATE, help=f"MP3 压缩目标码率，默认 {DEFAULT_MP3_BITRATE}")
    parser.add_argument("--stream", action="store_true", help="使用流式接口解析；默认使用非流式接口。")
    parser.add_argument(
        "--between-requests",
        type=int,
        default=DEFAULT_BETWEEN_REQUESTS_SECONDS,
        help=f"分段请求之间的等待秒数，默认 {DEFAULT_BETWEEN_REQUESTS_SECONDS}。",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"429 限速时每段最多重试次数，默认 {DEFAULT_RETRY_ATTEMPTS}。",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=DEFAULT_RETRY_WAIT_SECONDS,
        help=f"429 限速时基础等待秒数，默认 {DEFAULT_RETRY_WAIT_SECONDS}，会按次数递增。",
    )
    parser.add_argument("--cache-dir", default="transcripts_cache", help="分段转写缓存目录，默认 transcripts_cache。")
    parser.add_argument("--keep-cache", action="store_true", help="转写全部成功后保留分段缓存，默认会清理。")
    return parser.parse_args()


def main() -> int:
    configure_warning_filters()
    args = parse_args()
    load_dotenv(Path(".env"))

    if args.annotate_existing_md and args.diarize_existing_md:
        print("--annotate-existing-md 和 --diarize-existing-md 不能同时使用。", file=sys.stderr)
        return 1
    if args.annotate and args.diarize:
        print("--annotate 和 --diarize 不能同时使用；请选择轻量文本标注或 pyannote 音频分离标注。", file=sys.stderr)
        return 1
    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        print("--num-speakers 不能和 --min-speakers/--max-speakers 同时使用。", file=sys.stderr)
        return 1
    for speaker_arg_name in ("num_speakers", "min_speakers", "max_speakers"):
        speaker_arg_value = getattr(args, speaker_arg_name)
        if speaker_arg_value is not None and speaker_arg_value <= 0:
            print(f"--{speaker_arg_name.replace('_', '-')} 必须大于 0。", file=sys.stderr)
            return 1
    if args.diarization_cache_ttl_days < 0:
        print("--diarization-cache-ttl-days 不能小于 0。", file=sys.stderr)
        return 1
    if args.diarization_segmentation_batch_size <= 0:
        print("--diarization-segmentation-batch-size 必须大于 0。", file=sys.stderr)
        return 1
    if args.diarization_embedding_batch_size <= 0:
        print("--diarization-embedding-batch-size 必须大于 0。", file=sys.stderr)
        return 1
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        print("--min-speakers 不能大于 --max-speakers。", file=sys.stderr)
        return 1

    if args.annotate_existing_md:
        input_markdown_path = Path(args.annotate_existing_md).expanduser().resolve()
        if not input_markdown_path.exists():
            print(f"找不到 Markdown 文件：{input_markdown_path}", file=sys.stderr)
            return 1
        output_path = (
            Path(args.annotated_output).expanduser().resolve()
            if args.annotated_output
            else Path(args.output).expanduser().resolve()
            if args.output
            else default_annotated_output_path(input_markdown_path)
        )
        try:
            annotate_existing_markdown(input_markdown_path, output_path)
        except Exception as exc:
            print(f"标注失败：{exc}", file=sys.stderr)
            return 1
        print(f"标注完成：{output_path}")
        return 0

    if args.diarize_existing_md:
        input_markdown_path = Path(args.diarize_existing_md).expanduser().resolve()
        input_path = Path(args.audio).expanduser().resolve()
        if not input_markdown_path.exists():
            print(f"找不到 Markdown 文件：{input_markdown_path}", file=sys.stderr)
            return 1
        if not input_path.exists():
            print(f"找不到音频文件：{input_path}", file=sys.stderr)
            return 1
        output_path = (
            Path(args.annotated_output).expanduser().resolve()
            if args.annotated_output
            else Path(args.output).expanduser().resolve()
            if args.output
            else default_diarized_output_path(input_markdown_path)
        )
        try:
            diarize_existing_markdown(
                input_markdown_path,
                input_path,
                output_path,
                Path(args.model_cache_dir).expanduser().resolve(),
                Path(args.diarization_cache_dir).expanduser().resolve(),
                args.diarization_model,
                args.sample_rate,
                args.diarization_device,
                args.num_speakers,
                args.min_speakers,
                args.max_speakers,
                args.diarization_segmentation_batch_size,
                args.diarization_embedding_batch_size,
                args.diarization_cache_ttl_days,
            )
        except Exception as exc:
            print(f"pyannote 标注失败：{exc}", file=sys.stderr)
            return 1
        print(f"pyannote 标注完成：{output_path}")
        return 0

    input_path = Path(args.audio).expanduser().resolve()
    if not input_path.exists():
        print(f"找不到音频文件：{input_path}", file=sys.stderr)
        return 1
    if args.sample_rate <= 0:
        print("--sample-rate 必须大于 0。", file=sys.stderr)
        return 1
    if args.chunk_seconds <= 0:
        print("--chunk-seconds 必须大于 0。", file=sys.stderr)
        return 1
    if args.max_single_seconds <= 0:
        print("--max-single-seconds 必须大于 0。", file=sys.stderr)
        return 1
    if args.max_tokens <= 0:
        print("--max-tokens 必须大于 0。", file=sys.stderr)
        return 1
    if args.subdivide_seconds <= 0:
        print("--subdivide-seconds 必须大于 0。", file=sys.stderr)
        return 1
    if args.min_subdivide_seconds <= 0:
        print("--min-subdivide-seconds 必须大于 0。", file=sys.stderr)
        return 1
    if args.min_subdivide_seconds > args.subdivide_seconds:
        print("--min-subdivide-seconds 不能大于 --subdivide-seconds。", file=sys.stderr)
        return 1
    if args.silence_search_seconds < 0:
        print("--silence-search-seconds 不能小于 0。", file=sys.stderr)
        return 1
    if args.between_requests < 0:
        print("--between-requests 不能小于 0。", file=sys.stderr)
        return 1
    if args.retry_attempts <= 0:
        print("--retry-attempts 必须大于 0。", file=sys.stderr)
        return 1
    if args.retry_wait < 0:
        print("--retry-wait 不能小于 0。", file=sys.stderr)
        return 1
    try:
        parse_bitrate(args.mp3_bitrate)
    except ValueError:
        print("--mp3-bitrate 格式无效，例如可使用 16k、24k、16000。", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".md")

    try:
        with tempfile.TemporaryDirectory(prefix="mimo-asr-") as temp_dir:
            chunks = prepare_audio_chunks(
                input_path,
                Path(temp_dir),
                args.sample_rate,
                args.chunk_seconds,
                args.strategy,
                args.mp3_bitrate,
                args.max_single_seconds,
                args.silence_search_seconds,
                args.silence_dbfs,
            )
            cache_dir = Path(args.cache_dir)
            results = transcribe_chunks(
                chunks,
                input_path,
                args.language,
                args.stream,
                args.max_tokens,
                cache_dir,
                args.retry_attempts,
                args.retry_wait,
                args.between_requests,
                Path(temp_dir) / "retry_chunks",
                not args.no_auto_subdivide,
                args.subdivide_seconds,
                args.min_subdivide_seconds,
            )
        write_markdown(output_path, input_path, results)
        if args.annotate:
            annotated_output_path = (
                Path(args.annotated_output).expanduser().resolve()
                if args.annotated_output
                else default_annotated_output_path(output_path)
            )
            annotate_existing_markdown(output_path, annotated_output_path)
        if args.diarize:
            annotated_output_path = (
                Path(args.annotated_output).expanduser().resolve()
                if args.annotated_output
                else default_diarized_output_path(output_path)
            )
            diarize_existing_markdown(
                output_path,
                input_path,
                annotated_output_path,
                Path(args.model_cache_dir).expanduser().resolve(),
                Path(args.diarization_cache_dir).expanduser().resolve(),
                args.diarization_model,
                args.sample_rate,
                args.diarization_device,
                args.num_speakers,
                args.min_speakers,
                args.max_speakers,
                args.diarization_segmentation_batch_size,
                args.diarization_embedding_batch_size,
                args.diarization_cache_ttl_days,
            )
        if not args.keep_cache:
            cleanup_process_files(cache_dir)
    except Exception as exc:
        print(f"转写失败：{exc}", file=sys.stderr)
        return 1

    print(f"转写完成：{output_path}")
    if args.annotate or args.diarize:
        print(f"标注完成：{annotated_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
