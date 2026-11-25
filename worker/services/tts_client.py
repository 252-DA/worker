from __future__ import annotations

import asyncio
import os
import re
import urllib.parse
import urllib.request
import wave
from abc import ABC, abstractmethod

from dataclasses import dataclass

from document_chunk.shared.logger import get_logger
from worker.services.micro_content_subtitles import SubtitleCue

logger = get_logger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
    duration_seconds: float
    cues: list[SubtitleCue]


class TtsClient(ABC):
    @abstractmethod
    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        """Sinh file âm thanh từ văn bản và trả về thời lượng (giây)."""
        pass


class MockTtsClient(TtsClient):
    """
    Sinh file audio giả lập (im lặng) để chạy offline hoặc test mà không cần mạng.
    Tốc độ đọc trung bình tiếng Việt khoảng ~3 từ/giây (hoặc 12 ký tự/giây).
    """

    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        words = text.strip().split()
        duration_seconds = max(2.0, len(words) * 0.38)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if output_path.endswith(".mp3"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=24000:cl=mono",
                    "-t", str(duration_seconds),
                    "-c:a", "libmp3lame",
                    "-b:a", "64k",
                    output_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except Exception:
                pass
        else:
            sample_rate = 24000
            num_samples = int(duration_seconds * sample_rate)
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(bytearray(num_samples * 2))

        logger.info(
            "mock_tts.synthesized",
            path=output_path,
            duration=duration_seconds,
            words=len(words),
        )
        return duration_seconds


def _split_tts_chunks(text: str, max_chars: int = 180) -> list[str]:
    """Chia văn bản dài theo ranh giới câu/dấu phẩy để mỗi đoạn <= max_chars."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
        parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sent) if p.strip()]
        buf = ""
        for part in parts:
            if not buf:
                buf = part
            elif len(buf) + 1 + len(part) <= max_chars:
                buf = f"{buf} {part}"
            else:
                chunks.append(buf)
                buf = part
        if buf:
            # Nếu vẫn vượt max_chars thì cắt theo từ
            while len(buf) > max_chars:
                cut = buf.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                chunks.append(buf[:cut].strip())
                buf = buf[cut:].strip()
            if buf:
                chunks.append(buf)
    return chunks or [text.strip()[:max_chars] or "Nội dung bài giảng."]


def _fetch_google_tts_mp3(text: str, lang: str = "vi") -> bytes:
    """Tải luồng MP3 giọng đọc thật từ Google Translate TTS cho từng đoạn <=180 ký tự."""
    chunks = _split_tts_chunks(text, max_chars=180)
    out = bytearray()
    for chunk in chunks:
        q = urllib.parse.quote(chunk)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl={lang}&q={q}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Referer": "https://translate.google.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if data:
                out.extend(data)
    if not out:
        raise RuntimeError("Google Translate TTS returned empty audio")
    return bytes(out)


class EdgeTtsClient(TtsClient):
    """Sử dụng edge-tts miễn phí với các giọng đọc tiếng Việt chất lượng cao (Hoài My / Nam Minh)."""

    def __init__(
        self,
        fallback: TtsClient | None = None,
        *,
        allow_fallback: bool | None = None,
        allow_google_fallback: bool | None = None,
        max_retries_per_voice: int = 2,
    ) -> None:
        self._max_retries_per_voice = max(1, max_retries_per_voice)
        self._allow_google_fallback = (
            os.getenv("MICRO_CONTENT_GOOGLE_TTS_FALLBACK", "").strip().lower()
            in ("1", "true", "yes")
            if allow_google_fallback is None
            else allow_google_fallback
        )
        if fallback is not None:
            self._fallback: TtsClient | None = fallback
        else:
            enabled = (
                os.getenv("MICRO_CONTENT_ALLOW_MOCK_TTS", "").strip().lower()
                in ("1", "true", "yes")
                if allow_fallback is None
                else allow_fallback
            )
            self._fallback = MockTtsClient() if enabled else None

    async def _measure_duration(self, audio_path: str, fallback_duration: float) -> float:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-i", audio_path,
                "-show_entries", "format=duration",
                "-v", "quiet",
                "-of", "csv=p=0",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                val = float(stdout.decode().strip())
                if val > 0:
                    return round(val, 2)
        except Exception:
            pass
        return fallback_duration

    async def synthesize_with_cues(
        self,
        text: str,
        output_path: str,
        voice: str = "vi-VN-HoaiMyNeural",
    ) -> SynthesisResult:
        clean_text = text.strip()
        if not clean_text:
            clean_text = "Nội dung bài giảng."

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            import edge_tts  # type: ignore

            # Danh sách giọng ưu tiên: giọng chính -> giọng phụ
            candidate_voices = [voice]
            if voice == "vi-VN-HoaiMyNeural":
                candidate_voices.append("vi-VN-NamMinhNeural")
            elif voice == "vi-VN-NamMinhNeural":
                candidate_voices.append("vi-VN-HoaiMyNeural")

            last_error: Exception | None = None
            for v in candidate_voices:
                for attempt in range(1, self._max_retries_per_voice + 1):
                    try:
                        communicate = edge_tts.Communicate(
                            clean_text,
                            v,
                            boundary="SentenceBoundary",
                        )
                        audio_buf = bytearray()
                        cues: list[SubtitleCue] = []
                        async for chunk in communicate.stream():
                            ctype = chunk.get("type")
                            if ctype == "audio":
                                data = chunk.get("data")
                                if data:
                                    audio_buf.extend(data)
                            elif ctype in ("SentenceBoundary", "WordBoundary"):
                                offset_ms = round(float(chunk.get("offset", 0)) / 10_000)
                                dur_ms = round(float(chunk.get("duration", 0)) / 10_000)
                                cue_text = str(chunk.get("text") or "").strip()
                                if cue_text:
                                    cues.append(
                                        SubtitleCue(
                                            start_ms=max(0, offset_ms),
                                            end_ms=max(offset_ms + 50, offset_ms + dur_ms),
                                            text=cue_text,
                                        )
                                    )

                        if not audio_buf:
                            raise RuntimeError("Edge TTS returned empty audio stream")

                        with open(output_path, "wb") as f:
                            f.write(audio_buf)

                        words = clean_text.split()
                        fallback_duration = max(2.0, len(words) * 0.38)
                        duration = await self._measure_duration(output_path, fallback_duration)

                        logger.info(
                            "edge_tts.synthesized",
                            path=output_path,
                            voice=v,
                            attempt=attempt,
                            duration=duration,
                            cues=len(cues),
                        )
                        return SynthesisResult(duration_seconds=duration, cues=cues)
                    except Exception as voice_err:
                        last_error = voice_err
                        logger.warning(
                            "edge_tts.voice_failed",
                            voice=v,
                            attempt=attempt,
                            error=str(voice_err),
                        )
                        if attempt < self._max_retries_per_voice:
                            await asyncio.sleep(0.35 * attempt)

            if self._allow_google_fallback:
                lang = "en" if voice.lower().startswith("en-") else "vi"
                try:
                    mp3_bytes = await asyncio.to_thread(_fetch_google_tts_mp3, clean_text, lang)
                    with open(output_path, "wb") as f:
                        f.write(mp3_bytes)
                    words = clean_text.split()
                    fallback_duration = max(2.0, len(words) * 0.38)
                    duration = await self._measure_duration(output_path, fallback_duration)
                    logger.warning(
                        "edge_tts.google_tts_fallback_used",
                        voice=voice,
                        lang=lang,
                        duration=duration,
                        edge_error=str(last_error),
                    )
                    return SynthesisResult(duration_seconds=duration, cues=[])
                except Exception as g_err:
                    logger.warning("edge_tts.google_tts_fallback_failed", error=str(g_err))

            raise RuntimeError(f"All Edge TTS voices failed: {last_error}")
        except Exception as exc:
            if self._fallback is None:
                logger.error("edge_tts.failed", voice=voice, error=str(exc))
                raise RuntimeError(f"Edge TTS failed ({voice}): {exc}") from exc
            logger.warning("edge_tts.failed_fallback_to_mock", voice=voice, error=str(exc))

        if self._fallback is None:
            raise RuntimeError(f"Edge TTS failed ({voice})")
        dur = await self._fallback.synthesize(clean_text, output_path, voice)
        return SynthesisResult(duration_seconds=dur, cues=[])

    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
        result = await self.synthesize_with_cues(text, output_path, voice=voice)
        return result.duration_seconds


def get_default_tts_client() -> TtsClient:
    return EdgeTtsClient()
