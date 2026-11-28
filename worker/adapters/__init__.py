"""
Client I/O riêng của worker: hàng đợi ``worker_results`` (BullMQ), đọc Postgres
chỉ SELECT, model runtime của ai-sdk.

Chỉ phụ thuộc ``document_chunk``/``ai_runtime`` và thư viện ngoài, không import
gì khác trong gói ``worker``. Adapter dùng chung với packages-ai (Qdrant, MinIO,
Neo4j…) nằm ở ``document_chunk.adapters``.
"""
