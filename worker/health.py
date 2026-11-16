from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Callable

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthState:
    service: str
    worker_type: str
    ready: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class HealthServer:
    def __init__(
        self,
        host: str,
        port: int,
        state: HealthState,
        readiness_check: Callable[[], bool] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._state = state
        self._readiness_check = readiness_check
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return

        state = self._state
        readiness_check = self._readiness_check

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path not in ("/health", "/ready"):
                    self.send_error(404)
                    return

                ready = state.ready and (readiness_check() if readiness_check else True)
                status_code = 200 if self.path == "/health" or ready else 503
                payload = {
                    "status": "ok" if ready else "not_ready",
                    "service": state.service,
                    "worker_type": state.worker_type,
                    **state.metadata,
                }
                encoded = json.dumps(payload).encode("utf-8")

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self):  # noqa: N802
                # Hai endpoint dưới chỉ do BFF web gọi từ phía server, không qua trình duyệt,
                # nên không mở CORS. Chỉ worker video micro-content có chúng: mọi worker dùng
                # chung server health này và đều publish cổng ra host.
                if state.worker_type != "micro_content_video":
                    self.send_error(404)
                    return

                if self.path == "/script-from-pdf":
                    try:
                        content_length = int(self.headers.get("Content-Length", 0))
                        if content_length <= 0:
                            self._send_json({"error": "Dữ liệu yêu cầu trống."}, status_code=400)
                            return

                        body = self.rfile.read(content_length)
                        data = json.loads(body.decode("utf-8", errors="replace"))

                        pdf_base64 = data.get("pdf_base64")
                        if not pdf_base64:
                            self._send_json({"error": "Thiếu dữ liệu file PDF (pdf_base64)."}, status_code=400)
                            return

                        try:
                            pdf_bytes = base64.b64decode(pdf_base64)
                        except Exception as exc:
                            self._send_json({"error": f"Dữ liệu base64 không hợp lệ: {exc}"}, status_code=400)
                            return

                        start_page = int(data.get("start_page") or 1)
                        end_page = int(data.get("end_page") or 20)
                        target_scenes = int(data.get("target_scenes") or 5)
                        topic = data.get("topic")

                        from worker.services.pdf_script_extractor import extract_script_from_pdf

                        result = extract_script_from_pdf(
                            pdf_bytes=pdf_bytes,
                            start_page=start_page,
                            end_page=end_page,
                            target_scenes=target_scenes,
                            topic=topic,
                        )
                        self._send_json(result, status_code=200)

                    except ValueError as val_err:
                        self._send_json({"error": str(val_err)}, status_code=400)
                    except Exception as exc:
                        logger.error("health_server.script_from_pdf_failed", error=str(exc), exc_info=exc)
                        self._send_json({"error": f"Lỗi xử lý file PDF: {exc}"}, status_code=500)
                    return

                if self.path == "/tts-preview":
                    try:
                        content_length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                        data = json.loads(body.decode("utf-8", errors="replace")) if body else {}

                        voice = data.get("voice") or "vi-VN-HoaiMyNeural"
                        default_sample = (
                            "Xin chào các bạn, đây là giọng đọc thử nghiệm cho bài giảng của bạn."
                            if voice.startswith("vi")
                            else "Hello, this is a sample audio preview for your lecture."
                        )
                        text = data.get("text") or default_sample

                        import asyncio
                        from worker.services.tts_client import get_default_tts_client

                        tts = get_default_tts_client()
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                            tmp_path = tmp.name

                        try:
                            duration = asyncio.run(tts.synthesize(text=text, output_path=tmp_path, voice=voice))
                            with open(tmp_path, "rb") as f:
                                mp3_bytes = f.read()

                            encoded = base64.b64encode(mp3_bytes).decode("ascii")
                            self._send_json({
                                "status": "ok",
                                "voice": voice,
                                "duration": duration,
                                "audio_base64": encoded,
                            }, status_code=200)
                        finally:
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass

                    except Exception as exc:
                        logger.error("health_server.tts_preview_failed", error=str(exc), exc_info=exc)
                        self._send_json({"error": f"Lỗi tạo audio nghe thử: {exc}"}, status_code=500)
                    return

                self.send_error(404)

            def _send_json(self, data: dict, status_code: int = 200) -> None:
                encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, fmt: str, *args) -> None:
                return

        self._server = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("worker.health.started", host=self._host, port=self._port)

    def close(self) -> None:
        if self._server is None:
            return

        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("worker.health.stopped", host=self._host, port=self._port)
        self._server = None
        self._thread = None

    def set_ready(self, ready: bool) -> None:
        self._state.ready = ready


def start_health_server(
    host: str,
    port: int,
    state: HealthState,
    readiness_check: Callable[[], bool] | None = None,
) -> HealthServer:
    server = HealthServer(host=host, port=port, state=state, readiness_check=readiness_check)
    server.start()
    return server
