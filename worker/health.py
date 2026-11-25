from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Semaphore, Thread
from typing import Callable

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)

# Tuần tự hóa preview Manim để tránh 2 tiến trình Cairo/Manim chạy cùng lúc gây OOM trong WSL2
_MANIM_PREVIEW_SEM = Semaphore(1)
_MANIM_PREVIEW_CACHE_LOCK = Lock()
_MANIM_PREVIEW_CACHE: OrderedDict[str, dict] = OrderedDict()
_MANIM_PREVIEW_CACHE_MAX = 24
_MANIM_PREVIEW_DISK_DIR = os.getenv("MANIM_PREVIEW_CACHE_DIR", "/tmp/manim_preview_cache")

# 5 mẫu chuẩn mặc định để pre-warm nền, giúp giảng viên bấm "Xem trước" mở tức thì (<50ms)
_DEFAULT_PREWARM_TEMPLATES: list[tuple[str, list[int]]] = [
    ("insertion_sort", [23, 78, 45, 8, 32, 56]),
    ("merge_sort", [38, 27, 43, 3, 9, 82]),
    ("selection_sort", [29, 10, 14, 37, 13, 8]),
    ("bubble_sort", [52, 18, 41, 9, 33, 24]),
    ("quicksort", [38, 12, 45, 7, 29, 21]),
]


def _load_disk_preview_cache(cache_key: str) -> dict | None:
    path = os.path.join(_MANIM_PREVIEW_DISK_DIR, f"{cache_key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("video_base64"):
            return data
    except Exception:
        pass
    return None


def _save_disk_preview_cache(cache_key: str, payload: dict) -> None:
    try:
        os.makedirs(_MANIM_PREVIEW_DISK_DIR, exist_ok=True)
        path = os.path.join(_MANIM_PREVIEW_DISK_DIR, f"{cache_key}.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        pass


def _probe_video_duration(mp4_path: str) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                mp4_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return round(float(completed.stdout.strip()), 2)
    except Exception:
        pass
    return None


def render_manim_preview_payload(data: dict) -> tuple[dict, int]:
    """
    Render nhanh 1 cảnh Manim (360p @ 12fps + --disable_caching + cache RAM/đĩa)
    phục vụ nút "Xem trước cảnh Manim" trên giao diện biên tập web.
    """
    import asyncio
    from worker.services.manim_generator import (
        ManimGenerationRequest,
        ManimGenerator,
        build_default_manim_llm_client,
    )

    visual_mode = str(data.get("visual_mode") or "MANIM_TEMPLATE").strip().upper()
    if visual_mode not in ("MANIM_TEMPLATE", "MANIM_AUTO", "MANIM_CUSTOM"):
        return (
            {
                "error": (
                    f"Chế độ '{visual_mode}' không hỗ trợ xem trước Manim. "
                    "Hãy chọn MANIM_TEMPLATE, MANIM_AUTO hoặc MANIM_CUSTOM."
                )
            },
            400,
        )

    title = str(data.get("title") or "Algorithm Preview").strip()[:120]
    narration = str(data.get("narration") or "").strip()
    raw_bullets = data.get("bullets")
    bullets = [str(b).strip() for b in raw_bullets if str(b).strip()] if isinstance(raw_bullets, list) else []
    template_name = str(data.get("manim_template") or "").strip().lower() or None
    template_params = data.get("manim_params") if isinstance(data.get("manim_params"), dict) else None
    custom_source_code = str(data.get("manim_source_code") or "").strip() or None
    no_cache = bool(data.get("no_cache", False))

    if visual_mode == "MANIM_CUSTOM" and not custom_source_code:
        return ({"error": "Vui lòng nhập mã nguồn Python Manim trước khi xem trước."}, 400)
    if visual_mode == "MANIM_TEMPLATE" and not template_name:
        template_name = "insertion_sort"

    # Kiểm tra bộ nhớ đệm RAM & đĩa cho chế độ tất định (TEMPLATE & CUSTOM).
    # Lưu ý: với MANIM_TEMPLATE, không đưa tiêu đề slide (title) vào khóa cache để đổi tên cảnh
    # hay đổi sang cảnh khác dùng cùng mẫu + mảng vẫn hit cache tức thì!
    cache_key: str | None = None
    if visual_mode in ("MANIM_TEMPLATE", "MANIM_CUSTOM") and not no_cache:
        cache_obj: dict = {
            "v": 3,
            "visual_mode": visual_mode,
        }
        if visual_mode == "MANIM_TEMPLATE":
            norm_params = dict(template_params or {})
            if "pace" not in norm_params:
                norm_params["pace"] = 1.0
            else:
                try:
                    norm_params["pace"] = round(float(norm_params["pace"]), 2)
                except (TypeError, ValueError):
                    norm_params["pace"] = 1.0
            cache_obj["template_name"] = template_name
            cache_obj["template_params"] = norm_params
        else:
            cache_obj["custom_source_code"] = custom_source_code

        cache_material = json.dumps(cache_obj, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
        with _MANIM_PREVIEW_CACHE_LOCK:
            cached = _MANIM_PREVIEW_CACHE.get(cache_key)
            if cached is not None:
                _MANIM_PREVIEW_CACHE.move_to_end(cache_key)
                return ({**cached, "cached": True}, 200)
        disk_cached = _load_disk_preview_cache(cache_key)
        if disk_cached is not None:
            with _MANIM_PREVIEW_CACHE_LOCK:
                _MANIM_PREVIEW_CACHE[cache_key] = disk_cached
                _MANIM_PREVIEW_CACHE.move_to_end(cache_key)
                while len(_MANIM_PREVIEW_CACHE) > _MANIM_PREVIEW_CACHE_MAX:
                    _MANIM_PREVIEW_CACHE.popitem(last=False)
            return ({**disk_cached, "cached": True}, 200)

    raw_dur = data.get("target_duration_seconds")
    try:
        target_dur = float(raw_dur) if raw_dur is not None else 18.0
    except (TypeError, ValueError):
        target_dur = 18.0
    target_dur = max(6.0, min(60.0, target_dur))

    req = ManimGenerationRequest(
        title=title,
        narration=narration or title,
        bullets=bullets,
        target_duration_seconds=target_dur,
        visual_mode=visual_mode,
        template_name=template_name,
        template_params=template_params,
        custom_source_code=custom_source_code,
        max_attempts=2 if visual_mode == "MANIM_AUTO" else 1,
        timeout_seconds=95,
        resolution=(640, 360),
        fps=12,
    )

    llm_client = build_default_manim_llm_client() if visual_mode == "MANIM_AUTO" else None
    generator = ManimGenerator(llm_client=llm_client)

    acquired = _MANIM_PREVIEW_SEM.acquire(timeout=90)
    if not acquired:
        return ({"error": "Hệ thống đang bận render cảnh xem trước khác, vui lòng thử lại sau ít giây."}, 503)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_mp4 = tmp.name

    try:
        res = asyncio.run(generator.generate_scene_clip(req, tmp_mp4))
        if not res.success or not res.clip_path or not os.path.exists(res.clip_path):
            return (
                {
                    "error": res.error or "Không thể render cảnh Manim xem trước.",
                    "error_category": res.error_category,
                    "tier_used": res.tier_used,
                    "attempts": res.attempts,
                    "source_code": res.source_code,
                },
                422,
            )

        with open(res.clip_path, "rb") as f:
            mp4_bytes = f.read()

        duration = _probe_video_duration(res.clip_path) or res.duration_seconds
        payload = {
            "status": "ok",
            "mime_type": "video/mp4",
            "duration": duration,
            "size_bytes": len(mp4_bytes),
            "tier_used": res.tier_used,
            "attempts": res.attempts,
            "source_code": res.source_code,
            "video_base64": base64.b64encode(mp4_bytes).decode("ascii"),
            "cached": False,
        }

        if cache_key is not None:
            with _MANIM_PREVIEW_CACHE_LOCK:
                _MANIM_PREVIEW_CACHE[cache_key] = payload
                _MANIM_PREVIEW_CACHE.move_to_end(cache_key)
                while len(_MANIM_PREVIEW_CACHE) > _MANIM_PREVIEW_CACHE_MAX:
                    _MANIM_PREVIEW_CACHE.popitem(last=False)
            _save_disk_preview_cache(cache_key, payload)

        return (payload, 200)
    finally:
        _MANIM_PREVIEW_SEM.release()
        if os.path.exists(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except Exception:
                pass


def _prewarm_default_manim_previews() -> None:
    """Chạy nền sau khi worker khởi động để dựng sẵn cache đĩa cho 5 mẫu thuật toán mặc định."""
    for tpl, arr in _DEFAULT_PREWARM_TEMPLATES:
        try:
            payload, status = render_manim_preview_payload(
                {
                    "visual_mode": "MANIM_TEMPLATE",
                    "manim_template": tpl,
                    "manim_params": {"array": arr, "pace": 1.0},
                }
            )
            logger.info(
                "manim_preview.prewarm_done",
                template=tpl,
                status=status,
                cached=payload.get("cached"),
                duration=payload.get("duration"),
            )
        except Exception as exc:
            logger.warning("manim_preview.prewarm_failed", template=tpl, error=str(exc))


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

                if self.path == "/manim-preview":
                    try:
                        content_length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                        data = json.loads(body.decode("utf-8", errors="replace")) if body else {}
                        payload, status_code = render_manim_preview_payload(data)
                        self._send_json(payload, status_code=status_code)
                    except Exception as exc:
                        logger.error("health_server.manim_preview_failed", error=str(exc), exc_info=exc)
                        self._send_json({"error": f"Lỗi render cảnh Manim xem trước: {exc}"}, status_code=500)
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

        if (
            self._state.worker_type == "micro_content_video"
            and os.getenv("MANIM_PREVIEW_PREWARM", "true").strip().lower() not in ("0", "false", "no")
        ):
            Thread(target=_prewarm_default_manim_previews, daemon=True).start()

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
