"""
Tests for worker/use_cases/_llm_json.py — JSON extraction and repair helpers.
"""
import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.shared.result import Err, Ok
from worker.use_cases._llm_json import (
    extract_json_block,
    generate_structured_payload,
    parse_model_output,
)


class _TestModel(BaseModel):
    name: str
    age: int


class TestExtractJsonBlock:
    def test_pure_json(self):
        assert extract_json_block('{"name":"Alice","age":30}') == '{"name":"Alice","age":30}'

    def test_json_with_whitespace(self):
        assert extract_json_block('  \n{"name":"A"}\n  ') == '{"name":"A"}'

    def test_fenced_json(self):
        raw = '```json\n{"name":"Bob","age":25}\n```'
        assert extract_json_block(raw) == '{"name":"Bob","age":25}'

    def test_fenced_no_lang(self):
        raw = '```\n{"name":"C"}\n```'
        assert extract_json_block(raw) == '{"name":"C"}'

    def test_fenced_with_surrounding_text(self):
        raw = 'Here is the output:\n```json\n{"name":"D"}\n```\nHope this helps'
        assert extract_json_block(raw) == '{"name":"D"}'

    def test_multiline_json(self):
        raw = '{\n  "name": "E",\n  "age": 40\n}'
        assert extract_json_block(raw).startswith("{")
        assert "name" in extract_json_block(raw)

    def test_empty_input(self):
        assert extract_json_block("") == ""

    def test_no_fence_no_json(self):
        assert extract_json_block("just some text") == "just some text"


class TestParseModelOutput:
    def test_valid_json_matching_schema(self):
        result = parse_model_output('{"name":"Alice","age":30}', _TestModel)
        assert result.is_ok()
        model = result.unwrap()
        assert model.name == "Alice"
        assert model.age == 30

    def test_invalid_json_returns_err(self):
        result = parse_model_output("not json at all", _TestModel)
        assert result.is_err()
        assert isinstance(result.error, ProcessingError)

    def test_json_missing_field(self):
        result = parse_model_output('{"name":"Alice"}', _TestModel)
        assert result.is_err()

    def test_json_wrong_type(self):
        result = parse_model_output('{"name":"Alice","age":"thirty"}', _TestModel)
        assert result.is_err()

    def test_json_with_extra_fields_ok(self):
        result = parse_model_output('{"name":"Alice","age":30,"extra":"field"}', _TestModel)
        assert result.is_ok()
        model = result.unwrap()
        assert model.name == "Alice"

    def test_fenced_json(self):
        result = parse_model_output('```json\n{"name":"Bob","age":25}\n```', _TestModel)
        assert result.is_ok()
        assert result.unwrap().name == "Bob"


class TestGenerateStructuredPayload:
    def test_first_attempt_succeeds(self):
        llm = MagicMock()
        llm.generate.return_value = Ok('{"name":"Alice","age":30}')
        result = generate_structured_payload(
            llm_client=llm,
            prompt="Give me JSON",
            system="Be JSON",
            schema_model=_TestModel,
            repair_schema_hint='{"name":"...","age":0}',
        )
        assert result.is_ok()
        assert result.unwrap().name == "Alice"
        assert llm.generate.call_count == 1  # no repair needed

    def test_first_llm_fails(self):
        llm = MagicMock()
        llm.generate.return_value = Err(RuntimeError("llm down"))
        result = generate_structured_payload(
            llm_client=llm,
            prompt="Give me JSON",
            system="Be JSON",
            schema_model=_TestModel,
            repair_schema_hint="hint",
        )
        assert result.is_err()
        assert "llm down" in str(result.error)
        assert llm.generate.call_count == 1

    def test_first_invalid_json_triggers_repair_success(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            Ok("not valid json"),
            Ok('{"name":"Repaired","age":99}'),
        ]
        result = generate_structured_payload(
            llm_client=llm,
            prompt="Give me JSON",
            system="Be JSON",
            schema_model=_TestModel,
            repair_schema_hint='{"name":"...","age":0}',
        )
        assert result.is_ok()
        assert result.unwrap().name == "Repaired"
        assert result.unwrap().age == 99
        assert llm.generate.call_count == 2

    def test_repair_also_fails(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            Ok("invalid #1"),
            Ok("still not json"),
        ]
        result = generate_structured_payload(
            llm_client=llm,
            prompt="Give me JSON",
            system="Be JSON",
            schema_model=_TestModel,
            repair_schema_hint="hint",
        )
        assert result.is_err()
        assert isinstance(result.error, ProcessingError)
        assert "one repair attempt" in str(result.error)
        assert llm.generate.call_count == 2

    def test_repair_llm_fails(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            Ok("invalid"),
            Err(RuntimeError("repair llm down")),
        ]
        result = generate_structured_payload(
            llm_client=llm,
            prompt="Give me JSON",
            system="Be JSON",
            schema_model=_TestModel,
            repair_schema_hint="hint",
        )
        assert result.is_err()
        assert "repair llm down" in str(result.error)
