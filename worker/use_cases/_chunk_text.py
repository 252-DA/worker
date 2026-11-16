"""
Helper dùng chung cho micro-content use cases:
- Ghép heading_path + content_text để không mất chữ khi bộ tách chunk đẩy tiêu đề lên heading.
- Lọc chunk theo đúng dải trang, tránh lấy nhầm các trang đầu tài liệu khi dải trang trống.
- Tra tên học phần từ metadata_store thay cho UUID.
"""
from __future__ import annotations

from typing import Sequence

from document_chunk.domain.ports.metadata_store import IMetadataStore, StoredChunkMetadata


def format_chunk_with_heading(chunk: StoredChunkMetadata) -> str:
    """Ghép heading_path còn thiếu vào trước content_text của một chunk đơn lẻ."""
    body = (chunk.content_text or "").strip()
    body_lower = body.lower()
    headings: list[str] = []
    for raw_heading in chunk.heading_path or ():
        heading = (raw_heading or "").strip()
        if heading and heading.lower() not in body_lower:
            headings.append(heading)
    if body:
        headings.append(body)
    return "\n".join(headings).strip()


def format_page_chunks_text(chunks: Sequence[StoredChunkMetadata]) -> str:
    """
    Ghép nội dung các chunk cùng trang theo thứ tự, chèn heading mới chưa có
    trong phần thân và khử heading lặp giữa các chunk kế tiếp.
    """
    seen_headings: set[str] = set()
    parts: list[str] = []
    for chunk in chunks:
        body = (chunk.content_text or "").strip()
        body_lower = body.lower()
        for raw_heading in chunk.heading_path or ():
            heading = (raw_heading or "").strip()
            if not heading:
                continue
            key = heading.lower()
            if key in seen_headings:
                continue
            seen_headings.add(key)
            if key not in body_lower:
                parts.append(heading)
        if body:
            parts.append(body)
    return "\n".join(parts).strip()


def filter_chunks_in_page_range(
    chunks: Sequence[StoredChunkMetadata],
    page_from: int,
    page_to: int,
) -> list[StoredChunkMetadata]:
    """
    Lọc chunk thuộc dải trang [page_from, page_to].
    Chỉ fallback sang chunk không có page_number nếu toàn tài liệu chưa đánh số trang;
    không bao giờ lấy nhầm chunks[:8] ở đầu tài liệu khi dải trang không có chữ.
    """
    in_range = [
        c for c in chunks
        if c.page_number is not None and page_from <= c.page_number <= page_to
    ]
    if in_range:
        return in_range
    if any(c.page_number is not None for c in chunks):
        return []
    return [c for c in chunks if c.page_number is None]


def resolve_course_title(metadata_store: IMetadataStore, course_id: str) -> str:
    """Lấy tên học phần thân thiện từ curriculum thay cho mã UUID."""
    cur = metadata_store.get_curriculum(course_id)
    if cur.is_ok() and cur.unwrap() is not None:
        course = cur.unwrap()[0]
        title = (course.title_vi or course.title_en or course.code or "").strip()
        if title:
            return title
    return course_id
