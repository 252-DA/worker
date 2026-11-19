"""Đối chiếu kết quả chấm worker gửi đi với JSON Schema trong contracts/ (bỏ qua khi image không có contracts/)."""
import json
import uuid
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from referencing import Registry, Resource

SCHEMAS = Path(__file__).resolve().parents[3] / "contracts" / "integration-events" / "v1"


class FakeBullJob:
    """Job BullMQ giả: ``updateData`` ghi lại dữ liệu như Redis, lượt sau đọc lại được."""

    def __init__(self, data: dict, *, attempts: int = 5, attempts_made: int = 0, fail_update: bool = False):
        self.id = data.get("event_id")
        self.data = data
        self.opts = {"attempts": attempts}
        self.attemptsMade = attempts_made
        self.fail_update = fail_update
        self.updates: list[dict] = []

    async def updateData(self, data: dict) -> None:
        if self.fail_update:
            raise ConnectionError("redis down")
        self.updates.append(data)
        self.data = data

    def retry(self) -> "FakeBullJob":
        """Lượt thử kế tiếp của cùng job (dữ liệu đã cập nhật, attemptsMade + 1)."""
        nxt = FakeBullJob(self.data, attempts=self.opts["attempts"], attempts_made=self.attemptsMade + 1)
        return nxt


def new_id() -> str:
    return str(uuid.uuid4())


def assert_matches_contract(envelope: dict[str, Any]) -> None:
    """Envelope + payload đúng schema của kind (vd QUIZ_GRADING_RESULT → quiz-grading-result.json)."""
    if not SCHEMAS.exists():
        pytest.skip("contracts/ không có trong image này")
    store = {}
    for path in SCHEMAS.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        store[schema["$id"]] = schema
    schema = json.loads((SCHEMAS / f"{envelope['kind'].lower().replace('_', '-')}.json").read_text(encoding="utf-8"))
    checker = jsonschema.FormatChecker()

    @checker.checks("uuid")
    def _uuid(value: object) -> bool:
        try:
            uuid.UUID(str(value))
            return True
        except ValueError:
            return False

    registry = Registry().with_resources(
        (schema_id, Resource.from_contents(contents)) for schema_id, contents in store.items()
    )
    validator = jsonschema.Draft7Validator(schema, registry=registry, format_checker=checker)
    errors = [e.message for e in validator.iter_errors(envelope)]
    assert not errors, errors
