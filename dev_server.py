#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml


REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = REPO_ROOT / ".ui_runtime"
CONFIG_DIR = RUNTIME_DIR / "configs"
LOG_DIR = RUNTIME_DIR / "logs"


def compact_paper_id(url: str) -> str:
    match = re.search(
        r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)", str(url), re.IGNORECASE
    )
    if match:
        return match.group(1)
    return "paper.root"


@dataclass
class RunRecord:
    run_id: str
    root_id: str
    snapshot_path: str
    log_path: str
    config_path: str
    command: list[str]
    process: subprocess.Popen[str] | None = None
    status: str = "starting"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    exit_code: int | None = None
    lines: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_line(self, text: str) -> None:
        clean = text.rstrip("\n")
        if not clean:
            return
        with self.lock:
            self.lines.append({"cursor": self.next_cursor, "text": clean})
            self.next_cursor += 1
            if len(self.lines) > 2000:
                self.lines = self.lines[-2000:]

    def snapshot(self, cursor: int) -> dict[str, Any]:
        with self.lock:
            lines = [line for line in self.lines if line["cursor"] >= cursor]
            return {
                "run_id": self.run_id,
                "root_id": self.root_id,
                "status": self.status,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "exit_code": self.exit_code,
                "snapshot_path": self.snapshot_path,
                "log_path": self.log_path,
                "next_cursor": self.next_cursor,
                "lines": lines,
            }


class RunManager:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create_run(self, payload: dict[str, Any]) -> RunRecord:
        arxiv_url = str(payload.get("arxivUrl", "")).strip()
        if not arxiv_url:
            raise ValueError("Missing arXiv URL")

        max_rounds = int(payload.get("maxRounds", 3))
        top_k = int(payload.get("topK", 3))
        config = payload.get("config") or {}
        root_id = compact_paper_id(arxiv_url)
        run_id = uuid.uuid4().hex[:12]

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        config_path = CONFIG_DIR / f"{run_id}.yaml"
        log_path = LOG_DIR / f"{run_id}.log"
        snapshot_path = str(
            Path(str(config.get("dataDir") or "data"))
            / f"{root_id.replace('/', '_')}_graph.json"
        )

        runtime_config = {
            "agent": {
                "max_rounds": max_rounds,
                "top_k": top_k,
            },
            "llm": {
                "protocol": str(config.get("protocol") or "openai"),
                "base_url": str(config.get("baseUrl") or "https://api.openai.com/v1"),
                "api_key": str(config.get("apiKey") or ""),
                "model_id": str(config.get("modelId") or "gpt-4o"),
            },
            "paths": {
                "outputs_dir": str(config.get("outputsDir") or "outputs"),
                "data_dir": str(config.get("dataDir") or "data"),
            },
            "arxiv": {
                "enable_rate_limit": bool(config.get("enableRateLimit", True)),
                "request_timeout_seconds": int(config.get("requestTimeout") or 120),
            },
        }

        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(runtime_config, handle, sort_keys=False, allow_unicode=False)

        command = [
            sys.executable,
            "main.py",
            arxiv_url,
            "--config",
            str(config_path),
            "--max-rounds",
            str(max_rounds),
            "--top-k",
            str(top_k),
        ]

        run = RunRecord(
            run_id=run_id,
            root_id=root_id,
            snapshot_path=snapshot_path,
            log_path=str(log_path.relative_to(REPO_ROOT)),
            config_path=str(config_path.relative_to(REPO_ROOT)),
            command=command,
        )
        run.append_line(f"[ui] Starting trace for {arxiv_url}")
        run.append_line(f"[ui] Command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        run.process = process
        run.status = "running"

        with self._lock:
            self._runs[run_id] = run

        threading.Thread(target=self._capture_output, args=(run,), daemon=True).start()
        threading.Thread(target=self._wait_for_exit, args=(run,), daemon=True).start()
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def _capture_output(self, run: RunRecord) -> None:
        assert run.process is not None
        assert run.process.stdout is not None
        log_file = REPO_ROOT / run.log_path
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as handle:
            for line in run.process.stdout:
                run.append_line(line)
                handle.write(line)
                handle.flush()

    def _wait_for_exit(self, run: RunRecord) -> None:
        assert run.process is not None
        code = run.process.wait()
        run.completed_at = time.time()
        run.exit_code = code
        if code == 0:
            run.status = "completed"
            run.append_line("[ui] Trace process completed successfully")
        else:
            run.status = "failed"
            run.append_line(f"[ui] Trace process exited with code {code}")


RUN_MANAGER = RunManager()


class PaperAgentHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self._handle_create_run()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/logs"):
            self._handle_run_logs(parsed)
            return
        super().do_GET()

    def _handle_create_run(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            run = RUN_MANAGER.create_run(payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json(
                {"error": f"Failed to start run: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {
                "run_id": run.run_id,
                "root_id": run.root_id,
                "status": run.status,
                "snapshot_path": run.snapshot_path,
                "log_path": run.log_path,
            },
            status=HTTPStatus.CREATED,
        )

    def _handle_run_logs(self, parsed: Any) -> None:
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4:
            self.send_error(HTTPStatus.NOT_FOUND, "Invalid logs endpoint")
            return
        run_id = parts[2]
        run = RUN_MANAGER.get_run(run_id)
        if run is None:
            self._send_json(
                {"error": f"Run {run_id} not found"}, status=HTTPStatus.NOT_FOUND
            )
            return
        query = parse_qs(parsed.query)
        cursor = int(query.get("cursor", ["0"])[0])
        self._send_json(run.snapshot(cursor))

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(os.environ.get("PAPER_AGENT_UI_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), PaperAgentHandler)
    print(f"[*] Paper Agent UI server running at http://127.0.0.1:{port}/ui/")
    server.serve_forever()


if __name__ == "__main__":
    main()
