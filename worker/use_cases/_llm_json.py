"""
Shared JSON-parse + repair helpers — used by RunEnrichmentUseCase and GenerateCurriculumQuizUseCase.
"""
import json
import re

from pydantic import BaseModel, ValidationError

from document_chunk.domain.exceptions import ProcessingError
from document_chunk.domain.ports.llm_client import ILLMClient
from document_chunk.shared.result import Err, Ok, Result

_REPAIR_SYSTEM_PROMPT = "You repair malformed model outputs into strict JSON."


def extract_json_block(raw_text: str) -> str:
    trimmed = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", trimmed, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    return trimmed


def parse_model_output(
    raw_text: str,
    schema_model: type[BaseModel],
) -> Result[BaseModel, Exception]:
    try:
        payload = json.loads(extract_json_block(raw_text))
        return Ok(schema_model.model_validate(payload))
    except (json.JSONDecodeError, ValidationError) as exc:
        return Err(ProcessingError("Invalid structured output from LLM", cause=exc))


def generate_structured_payload(
    llm_client: ILLMClient,
    prompt: str,
    system: str,
    schema_model: type[BaseModel],
    repair_schema_hint: str,
) -> Result[BaseModel, Exception]:
    first_result = llm_client.generate(prompt, system=system)
    if first_result.is_err():
        return Err(first_result.error)

    parsed_result = parse_model_output(first_result.unwrap(), schema_model)
    if parsed_result.is_ok():
        return parsed_result

    repair_prompt = (
        "Rewrite the following content into strict JSON only.\n"
        f"Target schema: {repair_schema_hint}\n"
        "Do not add markdown fences or explanations.\n\n"
        f"{first_result.unwrap()}"
    )
    repair_result = llm_client.generate(repair_prompt, system=_REPAIR_SYSTEM_PROMPT)
    if repair_result.is_err():
        return Err(repair_result.error)

    repaired = parse_model_output(repair_result.unwrap(), schema_model)
    if repaired.is_ok():
        return repaired

    return Err(
        ProcessingError(
            "LLM returned invalid JSON after one repair attempt",
            cause=repaired.error,
        )
    )
