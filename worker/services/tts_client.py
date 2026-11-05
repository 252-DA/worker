from __future__ import annotations

import asyncio
import os
import wave
from abc import ABC, abstractmethod

from document_chunk.shared.logger import get_logger

logger = get_logger(__name__)


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


class EdgeTtsClient(TtsClient):
    """Sử dụng edge-tts miễn phí với các giọng đọc tiếng Việt chất lượng cao (Hoài My / Nam Minh)."""

    def __init__(self, fallback: TtsClient | None = None) -> None:
        self._fallback = fallback or MockTtsClient()

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

    async def synthesize(self, text: str, output_path: str, voice: str = "vi-VN-HoaiMyNeural") -> float:
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

            last_error = None
            for v in candidate_voices:
                try:
                    communicate = edge_tts.Communicate(clean_text, v)
                    await communicate.save(output_path)

                    words = clean_text.split()
                    fallback_duration = max(2.0, len(words) * 0.38)
                    duration = await self._measure_duration(output_path, fallback_duration)

                    logger.info("edge_tts.synthesized", path=output_path, voice=v, duration=duration)
                    return duration
                except Exception as voice_err:
                    last_error = voice_err
                    logger.warning("edge_tts.voice_failed", voice=v, error=str(voice_err))

            logger.error("edge_tts.all_voices_failed", error=str(last_error))
        except Exception as exc:
            logger.warning("edge_tts.failed_fallback_to_mock", error=str(exc))

        return await self._fallback.synthesize(clean_text, output_path, voice)


def get_default_tts_client() -> TtsClient:
    return EdgeTtsClient()
