#!/usr/bin/env python3
"""Submit a fixed ACE-Step smoke test and record reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import soundfile as sf


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Send a JSON request and return the decoded response."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def unwrap(response: dict[str, Any]) -> Any:
    """Return API data or raise a useful error for a failed wrapper."""
    if response.get("code") not in (None, 200) or response.get("error"):
        raise RuntimeError(f"ACE-Step API error: {response}")
    return response.get("data", response)


def git_state(repo_path: str) -> dict[str, Any]:
    """Capture the engine commit and a digest of tracked modifications."""
    repo = Path(repo_path)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    porcelain = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
    )
    return {
        "head": head,
        "dirty": bool(porcelain.strip()),
        "changed_entry_count": len(porcelain.splitlines()),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


class GpuMonitor:
    """Sample NVIDIA GPU memory and utilization while a task runs."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.peak_memory_mib = 0
        self.peak_utilization_percent = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start background sampling."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait briefly for the thread."""
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.index}",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                memory, utilization = (int(value.strip()) for value in output.split(","))
                self.peak_memory_mib = max(self.peak_memory_mib, memory)
                self.peak_utilization_percent = max(
                    self.peak_utilization_percent, utilization
                )
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(1)


def wav_metadata(path: Path) -> dict[str, Any]:
    """Read basic WAV properties and calculate a content hash."""
    info = sf.info(str(path))
    metadata = {
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate_hz": info.samplerate,
        "frame_count": info.frames,
        "duration_seconds": info.duration,
    }
    metadata["bytes"] = path.stat().st_size
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata


def download_audio(api_base: str, file_value: str, destination: Path) -> None:
    """Download one generated audio result to the requested destination."""
    url = urllib.parse.urljoin(f"{api_base}/", file_value.lstrip("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def parse_result(raw_result: Any) -> list[dict[str, Any]]:
    """Normalize the API's JSON-string result representation."""
    if isinstance(raw_result, str):
        raw_result = json.loads(raw_result)
    if isinstance(raw_result, dict):
        return [raw_result]
    if not isinstance(raw_result, list):
        raise RuntimeError(f"Unexpected task result: {raw_result!r}")
    return raw_result


def main() -> int:
    """Run one manifest-defined smoke test against a local ACE-Step API."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "run-result.json"

    started = time.time()
    monitor = GpuMonitor(int(manifest["hardware"]["gpu_index"]))
    run: dict[str, Any] = {
        "test_id": manifest["test_id"],
        "started_unix": started,
        "manifest": str(args.manifest.resolve()),
        "api_base": args.api_base,
        "engine_git": git_state(manifest["engine"]["repo_path"]),
        "request": manifest["request"],
    }

    try:
        run["health"] = unwrap(request_json(f"{args.api_base}/health"))
        run["models"] = unwrap(request_json(f"{args.api_base}/v1/models"))
        monitor.start()
        release = unwrap(
            request_json(f"{args.api_base}/release_task", manifest["request"])
        )
        task_id = release["task_id"]
        run["task_id"] = task_id
        deadline = time.time() + args.timeout_seconds

        while time.time() < deadline:
            query = unwrap(
                request_json(
                    f"{args.api_base}/query_result", {"task_id_list": [task_id]}
                )
            )
            task = query[0] if isinstance(query, list) else query
            status = int(task.get("status", 0))
            if status == 1:
                outputs = parse_result(task.get("result"))
                output = outputs[0]
                audio_path = args.output_dir / manifest["output_name"]
                download_audio(args.api_base, output["file"], audio_path)
                run["api_output"] = output
                run["audio_path"] = str(audio_path.resolve())
                run["audio"] = wav_metadata(audio_path)
                run["status"] = "succeeded"
                break
            if status == 2:
                raise RuntimeError(f"ACE-Step generation failed: {task}")
            time.sleep(2)
        else:
            raise TimeoutError(f"Task {task_id} exceeded {args.timeout_seconds}s")
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as error:
        run["status"] = "failed"
        run["error"] = repr(error)
        raise
    finally:
        monitor.stop()
        run["elapsed_seconds"] = time.time() - started
        run["gpu"] = {
            "peak_memory_used_mib": monitor.peak_memory_mib,
            "peak_utilization_percent": monitor.peak_utilization_percent,
            "sample_count": monitor.samples,
        }
        result_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(result_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
