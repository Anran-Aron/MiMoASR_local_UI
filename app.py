from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory


ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "mimoasr_script.py"
ENV_PATH = ROOT / ".env"
UPLOAD_DIR = ROOT / ".ui_uploads"
OUTPUT_DIR = ROOT / "output"

DEFAULTS = {
    "language": "auto",
    "chunk_seconds": 90,
    "max_single_seconds": 90,
    "max_tokens": 1024,
    "subdivide_seconds": 60,
    "min_subdivide_seconds": 15,
    "silence_search_seconds": 45,
    "silence_dbfs": -40.0,
    "strategy": "wav",
    "between_requests": 2,
    "retry_attempts": 6,
    "retry_wait": 60,
    "diarization_device": "auto",
    "diarization_segmentation_batch_size": 32,
    "diarization_embedding_batch_size": 32,
}

SETTING_KEYS = ("MIMO_API_KEY", "HF_TOKEN")
HIDDEN_SUFFIX = "********"
LOCAL_URL = "http://127.0.0.1:7860"
DIARIZATION_CACHE_TTL_DAYS = 30


app = Flask(__name__)
state_lock = threading.Lock()
jobs: dict[str, "JobState"] = {}
active_job_id: str | None = None


@dataclass
class JobState:
    id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    logs: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    error: str | None = None
    command_preview: str = ""


def read_env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in SETTING_KEYS:
            values[key] = value
    return values


def write_env_values(updates: dict[str, str]) -> None:
    existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    new_lines: list[str] = []

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            new_lines.append(raw_line)
            continue
        key = raw_line.split("=", 1)[0].strip()
        if key in updates:
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(raw_line)

    for key in SETTING_KEYS:
        if key in updates and key not in seen:
            new_lines.append(f"{key}={updates[key]}")

    ENV_PATH.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def cleanup_old_diarization_cache() -> None:
    cache_dir = ROOT / ".diarization_cache"
    if not cache_dir.exists():
        return
    cutoff = datetime.now().timestamp() - (DIARIZATION_CACHE_TTL_DAYS * 24 * 60 * 60)
    for cache_path in cache_dir.glob("*.json"):
        try:
            if cache_path.stat().st_mtime < cutoff:
                cache_path.unlink()
        except OSError:
            continue


def settings_status() -> dict[str, Any]:
    values = read_env_values()
    return {
        "mimoApiKeySet": bool(values.get("MIMO_API_KEY")),
        "hfTokenSet": bool(values.get("HF_TOKEN")),
    }


def clean_filename(filename: str) -> str:
    name = Path(filename or "audio").name
    name = re.sub(r"[\x00-\x1f/\\:]+", "_", name).strip(" .")
    if not name:
        name = "audio"
    stem = Path(name).stem or "audio"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"\s+", "_", stem).strip("._") or "audio"
    return f"{stem}{suffix}"


def output_name_for(upload_name: str, suffix: str = ".md") -> Path:
    safe_name = clean_filename(upload_name)
    stem = Path(safe_name).stem or "audio"
    return OUTPUT_DIR / f"{stem}{suffix}"


def bool_form(name: str) -> bool:
    return request.form.get(name) in {"1", "true", "on", "yes"}


def int_form(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = request.form.get(name, "").strip()
    value = default if raw_value == "" else int(raw_value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    return value


def float_form(name: str, default: float) -> float:
    raw_value = request.form.get(name, "").strip()
    return default if raw_value == "" else float(raw_value)


def add_job_log(job_id: str, line: str) -> None:
    with state_lock:
        job = jobs.get(job_id)
        if job:
            job.logs.append(line.rstrip())


def set_job_status(job_id: str, status: str, error: str | None = None, results: list[str] | None = None) -> None:
    global active_job_id
    with state_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.status = status
        job.error = error
        if results is not None:
            job.results = results
        if status in {"done", "failed"} and active_job_id == job_id:
            active_job_id = None


def build_command(upload_path: Path, original_name: str) -> tuple[list[str], list[Path]]:
    annotation_mode = request.form.get("annotation_mode", "none")
    language = request.form.get("language", DEFAULTS["language"]).strip() or DEFAULTS["language"]
    strategy = request.form.get("strategy", DEFAULTS["strategy"]).strip() or DEFAULTS["strategy"]

    output_path = output_name_for(original_name)
    annotated_path: Path | None = None
    command = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        str(upload_path),
        "-o",
        str(output_path),
        "--language",
        language,
        "--strategy",
        strategy,
        "--chunk-seconds",
        str(int_form("chunk_seconds", DEFAULTS["chunk_seconds"], 1)),
        "--max-single-seconds",
        str(int_form("max_single_seconds", DEFAULTS["max_single_seconds"], 1)),
        "--max-tokens",
        str(int_form("max_tokens", DEFAULTS["max_tokens"], 1)),
        "--subdivide-seconds",
        str(int_form("subdivide_seconds", DEFAULTS["subdivide_seconds"], 1)),
        "--min-subdivide-seconds",
        str(int_form("min_subdivide_seconds", DEFAULTS["min_subdivide_seconds"], 1)),
        "--silence-search-seconds",
        str(int_form("silence_search_seconds", DEFAULTS["silence_search_seconds"], 0)),
        "--silence-dbfs",
        str(float_form("silence_dbfs", DEFAULTS["silence_dbfs"])),
        "--between-requests",
        str(int_form("between_requests", DEFAULTS["between_requests"], 0)),
        "--retry-attempts",
        str(int_form("retry_attempts", DEFAULTS["retry_attempts"], 1)),
        "--retry-wait",
        str(int_form("retry_wait", DEFAULTS["retry_wait"], 0)),
        "--cache-dir",
        str(ROOT / "transcripts_cache"),
    ]

    if bool_form("keep_cache"):
        command.append("--keep-cache")
    if bool_form("no_auto_subdivide"):
        command.append("--no-auto-subdivide")

    if annotation_mode == "light":
        annotated_path = output_path.with_name(f"{output_path.stem}_annotated.md")
        command.extend(["--annotate", "--annotated-output", str(annotated_path)])
    elif annotation_mode == "pyannote":
        annotated_path = output_path.with_name(f"{output_path.stem}_diarized.md")
        command.extend(
            [
                "--diarize",
                "--annotated-output",
                str(annotated_path),
                "--diarization-device",
                request.form.get("diarization_device", DEFAULTS["diarization_device"]),
                "--diarization-segmentation-batch-size",
                str(
                    int_form(
                        "diarization_segmentation_batch_size",
                        DEFAULTS["diarization_segmentation_batch_size"],
                        1,
                    )
                ),
                "--diarization-embedding-batch-size",
                str(
                    int_form(
                        "diarization_embedding_batch_size",
                        DEFAULTS["diarization_embedding_batch_size"],
                        1,
                    )
                ),
            ]
        )
        speaker_mode = request.form.get("speaker_mode", "auto")
        if speaker_mode == "fixed":
            command.extend(["--num-speakers", str(int_form("num_speakers", 2, 1))])
        elif speaker_mode == "range":
            min_speakers = int_form("min_speakers", 2, 1)
            max_speakers = int_form("max_speakers", 4, 1)
            if min_speakers > max_speakers:
                raise ValueError("说话人数下限不能大于上限")
            command.extend(["--min-speakers", str(min_speakers), "--max-speakers", str(max_speakers)])

    expected = [output_path]
    if annotated_path:
        expected.append(annotated_path)
    return command, expected


def run_job(job_id: str, command: list[str], expected_outputs: list[Path], upload_path: Path) -> None:
    set_job_status(job_id, "running")
    add_job_log(job_id, "开始处理音频...")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            add_job_log(job_id, line)
        return_code = process.wait()
        existing_results = [path.name for path in expected_outputs if path.exists()]
        if return_code == 0:
            add_job_log(job_id, "处理完成。")
            set_job_status(job_id, "done", results=existing_results)
        else:
            set_job_status(job_id, "failed", error=f"处理失败，退出码 {return_code}。", results=existing_results)
    except Exception as exc:
        add_job_log(job_id, f"任务异常：{exc}")
        set_job_status(job_id, "failed", error=str(exc))
    finally:
        try:
            upload_path.unlink(missing_ok=True)
            if upload_path.parent != UPLOAD_DIR and upload_path.parent.exists():
                shutil.rmtree(upload_path.parent, ignore_errors=True)
        except Exception as exc:
            add_job_log(job_id, f"清理上传临时文件失败：{exc}")


@app.get("/")
def index() -> str:
    return render_template("index.html", defaults=DEFAULTS)


@app.get("/api/settings")
def get_settings():
    return jsonify(settings_status())


@app.post("/api/settings")
def save_settings():
    current = read_env_values()
    updates: dict[str, str] = {}
    for key, field_name in (("MIMO_API_KEY", "mimo_api_key"), ("HF_TOKEN", "hf_token")):
        raw_value = request.form.get(field_name, "").strip()
        if raw_value == HIDDEN_SUFFIX:
            updates[key] = current.get(key, "")
        else:
            updates[key] = raw_value
    write_env_values(updates)
    return jsonify({"ok": True, **settings_status()})


@app.post("/api/start")
def start_job():
    global active_job_id
    settings = settings_status()
    annotation_mode = request.form.get("annotation_mode", "none")
    if not settings["mimoApiKeySet"]:
        return jsonify({"ok": False, "error": "请先在设置中填写 Xiaomi ASR API Key。"}), 400
    if annotation_mode == "pyannote" and not settings["hfTokenSet"]:
        return jsonify({"ok": False, "error": "选择 pyannote 方案时，请先在设置中填写 Hugging Face Token。"}), 400
    uploaded_file = request.files.get("audio")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "请先拖入或选择一个音频文件。"}), 400

    with state_lock:
        if active_job_id is not None:
            return jsonify({"ok": False, "error": "已有任务正在运行，请等待完成后再开始。"}), 409

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        job_upload_dir = UPLOAD_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = clean_filename(uploaded_file.filename)
        upload_path = job_upload_dir / safe_name
        uploaded_file.save(upload_path)
        command, expected_outputs = build_command(upload_path, uploaded_file.filename)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    command_preview = " ".join(command[:4] + ["..."])
    job = JobState(id=job_id, command_preview=command_preview)
    with state_lock:
        jobs[job_id] = job
        active_job_id = job_id

    thread = threading.Thread(target=run_job, args=(job_id, command, expected_outputs, upload_path), daemon=True)
    thread.start()
    return jsonify({"ok": True, "jobId": job_id})


@app.get("/api/status/<job_id>")
def job_status(job_id: str):
    with state_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "找不到任务。"}), 404
        return jsonify(
            {
                "ok": True,
                "id": job.id,
                "status": job.status,
                "createdAt": job.created_at,
                "logs": job.logs[-500:],
                "results": job.results,
                "error": job.error,
            }
        )


@app.get("/api/results")
def list_results():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (path for path in OUTPUT_DIR.glob("*.md") if path.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return jsonify(
        {
            "ok": True,
            "results": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "modifiedAt": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                }
                for path in files
            ],
        }
    )


@app.get("/download/<path:filename>")
def download_result(filename: str):
    safe_name = Path(filename).name
    return send_from_directory(OUTPUT_DIR, safe_name, as_attachment=True)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_diarization_cache()
    print(f"本地 UI：{LOCAL_URL}")
    print("关闭此终端窗口或按 Ctrl+C，即可停止服务并释放 7860 端口。")
    if os.environ.get("MIMO_ASR_OPEN_BROWSER") == "1":
        threading.Timer(1.5, lambda: webbrowser.open(LOCAL_URL)).start()
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True, load_dotenv=False)
