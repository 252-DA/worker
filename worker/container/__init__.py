"""
Dựng phụ thuộc cho từng runner, mỗi runner một module:

- ``base``: lớp cha có ``close()`` và các builder hạ tầng dùng chung
  (Postgres, Qdrant, MinIO, Neo4j, BullMQ, ``worker_results``, MCP)
- ``llm``: registry chọn model theo task và ``TaskRoles``
- ``document``, ``enrichment``, ``content_generation``, ``answer_grading``,
  ``outbox``, ``micro_content_video``: container của runner cùng tên

Runner chỉ import module của mình, nên không kéo phụ thuộc của runner khác
(parser, BGE, Manim…) vào process.
"""
