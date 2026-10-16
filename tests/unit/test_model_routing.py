"""
Chọn model theo task, không theo process — docs/model-routing-design.md.

Mỗi test khoá một tính chất mà thiết kế dựa vào: profile là đơn vị duy nhất
được gọi tên, key đến từ tên biến env trong catalog, và hai vai trò trong cùng
một process có thể chạy hai model khác nhau.
"""
from unittest.mock import MagicMock

import pytest
from ai_runtime import ModelProfile
from ai_runtime.errors import ModelConfigurationError

from document_chunk.application.dto.generation_dto import GenerateCurriculumQuizRequest
from document_chunk.domain.ports.llm_client import LLMUsage
from document_chunk.shared.result import Err, Ok
from tests.support import as_llm_client
from tests.unit.test_quizgen_phase0 import _all_supported, _curriculum, _questions_json
from worker.adapters.ai_runtime_llm_client import AIRuntimeLLMClient
from worker.config import WorkerSettings
from worker.container import build_llm_client, build_llm_client_for, build_model_registry
from worker.llm_role import LlmRole
from worker.use_cases.generate_curriculum_quiz import GenerateCurriculumQuizUseCase
from worker.use_cases.run_enrichment import RunEnrichmentRequest, RunEnrichmentUseCase

_LEGACY_ENV = ("LLM__PROVIDER", "LLM__MODEL", "LLM__BASE_URL", "LLM__API_KEY")
_PROFILE_ENV = (
    "LLM_PROFILE",
    "LLM_PROFILE__ENRICHMENT",
    "LLM_PROFILE__QUIZ_GENERATION",
    "LLM_PROFILE__QUIZ_VERIFICATION",
)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """Máy dev có .env thật; test phải quyết định env của chính nó."""
    for name in _LEGACY_ENV + _PROFILE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")


class TestRegistryWiring:
    def test_task_profile_env_selects_the_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROFILE__ENRICHMENT", "gemini-pro")
        resolved = build_model_registry(WorkerSettings()).resolve("enrichment")

        assert resolved.profile.name == "gemini-pro"
        assert resolved.binding.source == "env:LLM_PROFILE__ENRICHMENT"

    def test_without_any_env_each_task_takes_its_catalog_default(self):
        registry = build_model_registry(WorkerSettings())

        assert registry.resolve("quiz_generation").provider == "deepseek"
        assert registry.resolve("enrichment").provider == "gemini"

    def test_legacy_llm_env_still_wins(self, monkeypatch):
        # Một deployment chưa đổi sang LLM_PROFILE__* không được âm thầm nhảy
        # sang model khác chỉ vì code mới lên.
        monkeypatch.setenv("LLM__PROVIDER", "gemini")
        monkeypatch.setenv("LLM__MODEL", "gemini-1.5-legacy")
        monkeypatch.setenv("LLM__API_KEY", "legacy-key")
        resolved = build_model_registry(WorkerSettings()).resolve("quiz_generation")

        assert resolved.model_id == "gemini-1.5-legacy"
        assert resolved.binding.source == "legacy-env"

    def test_legacy_env_without_its_key_does_not_win(self, monkeypatch):
        # worker/Dockerfile từng gắn LLM__PROVIDER/LLM__MODEL vào image mà
        # compose không truyền LLM__API_KEY: profile cũ thắng rồi gọi model
        # không có key, mọi lần chấm tự luận đều hỏng.
        monkeypatch.setenv("LLM__PROVIDER", "gemini")
        monkeypatch.setenv("LLM__MODEL", "gemini-3-flash-preview")
        resolved = build_model_registry(WorkerSettings()).resolve("quiz_verification")

        assert resolved.profile.name == "gemini-flash"
        assert resolved.binding.source == "task-default"

    def test_profile_env_beats_legacy_env(self, monkeypatch):
        monkeypatch.setenv("LLM__PROVIDER", "gemini")
        monkeypatch.setenv("LLM__MODEL", "gemini-1.5-legacy")
        monkeypatch.setenv("LLM__API_KEY", "legacy-key")
        monkeypatch.setenv("LLM_PROFILE__QUIZ_GENERATION", "deepseek-chat")
        resolved = build_model_registry(WorkerSettings()).resolve("quiz_generation")

        assert resolved.provider == "deepseek"

    def test_unknown_provider_in_legacy_config_raises(self):
        settings = WorkerSettings()
        settings.core.llm.provider = "unknown"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_model_registry(settings)

    def test_unknown_profile_name_fails_with_the_valid_list(self, monkeypatch):
        monkeypatch.setenv("LLM_PROFILE__ENRICHMENT", "gpt-9")
        with pytest.raises(ModelConfigurationError, match="gpt-9"):
            build_llm_client(WorkerSettings(), "enrichment")

    def test_built_client_carries_its_profile_labels(self, monkeypatch):
        monkeypatch.setenv("LLM_PROFILE__QUIZ_GENERATION", "deepseek-chat")
        client = build_llm_client(WorkerSettings(), "quiz_generation")

        assert isinstance(client, AIRuntimeLLMClient)
        assert client.provider == "deepseek"
        assert client.profile_name == "deepseek-chat"

    def test_two_tasks_one_process_two_models(self, monkeypatch):
        monkeypatch.setenv("LLM_PROFILE__QUIZ_GENERATION", "deepseek-chat")
        monkeypatch.setenv("LLM_PROFILE__QUIZ_VERIFICATION", "gemini-flash")
        registry = build_model_registry(WorkerSettings())

        writer = build_llm_client_for(registry, "quiz_generation")
        judge = build_llm_client_for(registry, "quiz_verification")

        assert (writer.provider, judge.provider) == ("deepseek", "gemini")


class _Bindings:
    """Bảng llm_model_bindings giả, thay cho ModelBindingStore."""

    rows: dict[str, str] = {}

    def __init__(self, sql) -> None:
        del sql

    def fetch(self) -> dict[str, str]:
        return dict(_Bindings.rows)


@pytest.fixture
def db_bindings(monkeypatch):
    """Settings có SQL bật, bảng binding giả, TTL 0 để mỗi lần hỏi đọc lại bảng."""
    import worker.container as container_module
    import worker.services.model_binding_store as store_module

    _Bindings.rows = {}
    monkeypatch.setattr(store_module, "ModelBindingStore", _Bindings)
    monkeypatch.setattr(container_module, "_BINDING_TTL_SECONDS", 0.0)
    settings = WorkerSettings()
    settings.core.sql.enabled = True
    return settings


class TestBindingsFromDatabase:
    def test_table_binding_beats_env(self, monkeypatch, db_bindings):
        monkeypatch.setenv("LLM_PROFILE__QUIZ_GENERATION", "deepseek-chat")
        _Bindings.rows = {"quiz_generation": "gemini-pro"}
        resolved = build_model_registry(db_bindings).resolve("quiz_generation")

        assert resolved.profile.name == "gemini-pro"
        assert resolved.binding.source == "db:llm_model_bindings"

    def test_empty_table_keeps_env_and_defaults(self, db_bindings):
        resolved = build_model_registry(db_bindings).resolve("quiz_generation")

        assert resolved.profile.name == "deepseek-chat"

    def test_internal_legacy_profile_cannot_be_chosen_from_the_table(self, db_bindings):
        _Bindings.rows = {"enrichment": "env-llm"}
        resolved = build_model_registry(db_bindings).resolve("enrichment")

        assert resolved.profile.name == "gemini-flash"

    def test_sql_disabled_never_reads_the_table(self, monkeypatch):
        import worker.services.model_binding_store as store_module

        monkeypatch.setattr(store_module, "ModelBindingStore", MagicMock(side_effect=AssertionError))
        build_model_registry(WorkerSettings()).resolve("enrichment")


class TestModelChangeWithoutRestart:
    def test_task_roles_follow_the_table_between_jobs(self, db_bindings):
        from worker.container import TaskRoles

        roles = TaskRoles(build_model_registry(db_bindings))
        first = roles.role("enrichment")
        _Bindings.rows = {"enrichment": "gemini-pro"}
        second = roles.role("enrichment")

        assert (first.profile, second.profile) == ("gemini-flash", "gemini-pro")

    def test_same_profile_reuses_one_client(self, db_bindings):
        from worker.container import TaskRoles

        roles = TaskRoles(build_model_registry(db_bindings))

        assert roles.role("enrichment").client is roles.role("enrichment").client

    def test_grading_container_picks_up_a_new_judge(self, db_bindings):
        from worker.container import AnswerGradingContainer

        container = AnswerGradingContainer(db_bindings)
        before = container.grade_short_answers_use_case._grader.profile
        _Bindings.rows = {"quiz_verification": "gemini-pro"}
        after = container.grade_short_answers_use_case._grader.profile

        assert (before, after) == ("gemini-flash", "gemini-pro")
        container.close()

    def test_quiz_container_builds_each_job_with_current_writer_and_judge(self, db_bindings):
        from worker.container import CurriculumQuizContainer

        container = CurriculumQuizContainer(db_bindings)
        _Bindings.rows = {"quiz_generation": "gemini-pro", "quiz_verification": "deepseek-chat"}
        use_case = container.generate_curriculum_quiz_use_case

        assert use_case._writer.profile == "gemini-pro"
        assert use_case._verifier.profile == "deepseek-chat"
        container.close()

    def test_enrichment_container_rebuilds_with_the_new_model(self, db_bindings):
        from worker.container import EnrichmentWorkerContainer

        container = EnrichmentWorkerContainer(db_bindings)
        _Bindings.rows = {"enrichment": "deepseek-chat"}

        assert container.run_enrichment_use_case._role.profile == "deepseek-chat"
        container.close()


class TestCostFromCatalog:
    """cost_usd từng luôn là 0: adapter đọc TokenUsage.cost_usd, field không có."""

    def _client(self, profile, *, input_tokens=1000, output_tokens=2000):
        runtime = MagicMock()
        runtime.generate_text.return_value = MagicMock(
            text="hello",
            model=profile.model if profile else "m",
            usage=MagicMock(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        return AIRuntimeLLMClient(runtime, profile=profile)

    def test_priced_profile_turns_tokens_into_dollars(self):
        profile = ModelProfile(
            name="p", provider="gemini", model="m",
            price_in_per_1m=3.0, price_out_per_1m=15.0, priced_on="2026-09-28",
        )
        _, usage = self._client(profile).generate_with_usage("p").unwrap()

        assert usage.prompt_tokens == 1000
        assert usage.completion_tokens == 2000
        assert usage.cost_usd == pytest.approx(1000 / 1e6 * 3.0 + 2000 / 1e6 * 15.0)

    def test_unpriced_profile_records_zero_and_says_so_once(self):
        profile = ModelProfile(name="p", provider="gemini", model="m")
        client = self._client(profile)

        _, usage = client.generate_with_usage("p").unwrap()
        assert usage.cost_usd == 0.0
        assert client._warned_unpriced is True

    def test_tokens_are_recorded_even_without_a_profile(self):
        _, usage = self._client(None).generate_with_usage("p").unwrap()

        assert (usage.prompt_tokens, usage.completion_tokens) == (1000, 2000)
        assert usage.cost_usd == 0.0


class TestQuizRolesUseTheirOwnModel:
    def _run(self, store, writer, judge):
        _curriculum(store, bloom_level=4)
        writer.generate.side_effect = [Ok(_questions_json(1))]
        judge.generate.side_effect = [Ok(_all_supported(1))]
        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=store,
            llm_client=writer,
            llm_provider="deepseek",
            llm_profile="deepseek-chat",
            verifier=LlmRole(client=judge, provider="gemini", profile="gemini-flash"),
        )
        return use_case.execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
        ))

    @pytest.fixture
    def writer(self):
        client = as_llm_client(MagicMock())
        client.model_id = "deepseek-chat"
        return client

    @pytest.fixture
    def judge(self):
        client = as_llm_client(MagicMock())
        client.model_id = "gemini-3-flash-preview"
        return client

    def test_writer_writes_and_judge_judges(self, writer, judge):
        store = MagicMock()
        assert self._run(store, writer, judge).is_ok()

        assert writer.generate.call_count == 1
        assert judge.generate.call_count == 1

    def test_each_usage_row_records_the_model_that_did_the_work(self, writer, judge):
        store = MagicMock()
        assert self._run(store, writer, judge).is_ok()

        rows = {
            call.kwargs["use_case"]: (call.kwargs["provider"], call.kwargs["model"])
            for call in store.record_llm_usage.call_args_list
        }
        assert rows["quiz_generation"] == ("deepseek", "deepseek-chat")
        assert rows["quiz_verification"] == ("gemini", "gemini-3-flash-preview")

    def test_items_are_stamped_with_the_writer_model(self, writer, judge):
        store = MagicMock()
        assert self._run(store, writer, judge).is_ok()

        items = store.persist_curriculum_quiz_items.call_args.kwargs["quiz_items"]
        assert {item.model_id for item in items} == {"deepseek-chat"}

    def test_without_a_verifier_role_one_model_does_both(self, writer):
        store = MagicMock()
        _curriculum(store, bloom_level=4)
        writer.generate.side_effect = [Ok(_questions_json(1)), Ok(_all_supported(1))]
        use_case = GenerateCurriculumQuizUseCase(
            metadata_store=store, llm_client=writer, llm_provider="gemini",
        )
        result = use_case.execute(GenerateCurriculumQuizRequest(
            course_id="course-001", target_kind="lo", target_code="L.O.1.1", count=1,
        ))

        assert result.is_ok()
        assert writer.generate.call_count == 2
        providers = {c.kwargs["provider"] for c in store.record_llm_usage.call_args_list}
        assert providers == {"gemini"}


def test_usage_accumulator_still_sums_repair_calls():
    # Lần sửa JSON cũng tốn tiền; cộng dồn phải giữ nguyên sau khi thêm cost.
    total = LLMUsage(prompt_tokens=10, completion_tokens=5, cost_usd=0.001)
    total = total + LLMUsage(prompt_tokens=3, completion_tokens=1, cost_usd=0.0002)

    assert total.prompt_tokens == 13
    assert total.cost_usd == pytest.approx(0.0012)


class TestEnrichmentRecordsItsCost:
    """Đường enrichment từng không ghi dòng nào vào llm_usage_logs."""

    def _run(self, store, llm, document, context, provider="gemini"):
        from document_chunk.domain.ports.metadata_store import StoredChunkMetadata

        store.get_document_context.return_value = Ok(context)
        store.list_chunks.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-001", document_id=document.id, chunk_index=0,
                heading_path=("Giải tích", "Đạo hàm"), heading_level=2, language="vi",
                content_text="Đạo hàm mô tả tốc độ thay đổi tức thời.",
            )
        ])
        llm.generate.side_effect = [
            Ok('{"cards":[{"title":"Đạo hàm","bullets":["a","b"],"key_insight":"c"}]}'),
            Ok('{"questions":[{"question":"Để làm gì?","choices":["a","b","c","d"],'
               '"correct_index":1,"explanation":"vì vậy","difficulty":"medium"}]}'),
        ]
        use_case = RunEnrichmentUseCase(
            metadata_store=store,
            llm_client=llm,
            max_section_chars=500,
            llm_provider=provider,
            llm_profile="gemini-flash",
        )
        return use_case.execute(RunEnrichmentRequest(document_id=document.id))

    def test_cards_and_quiz_each_leave_a_usage_row(
        self, sample_document, sample_document_context, mock_metadata_store, mock_llm_client
    ):
        assert self._run(
            mock_metadata_store, mock_llm_client, sample_document, sample_document_context
        ).is_ok()

        rows = {
            call.kwargs["use_case"]: call.kwargs
            for call in mock_metadata_store.record_llm_usage.call_args_list
        }
        assert set(rows) == {"enrichment_cards", "enrichment_quiz"}
        for row in rows.values():
            assert row["provider"] == "gemini"
            assert row["status"] == "OK"  # CHECK của bảng phân biệt hoa thường
            assert row["latency_ms"] >= 0

    def test_a_failed_generation_is_still_recorded(
        self, sample_document, sample_document_context, mock_metadata_store, mock_llm_client
    ):
        from document_chunk.domain.ports.metadata_store import StoredChunkMetadata

        mock_metadata_store.get_document_context.return_value = Ok(sample_document_context)
        mock_metadata_store.list_chunks.return_value = Ok([
            StoredChunkMetadata(
                chunk_id="chunk-001", document_id=sample_document.id, chunk_index=0,
                heading_path=("Giải tích",), heading_level=1, language="vi",
                content_text="nội dung",
            )
        ])
        mock_llm_client.generate.side_effect = [Ok("không phải JSON"), Ok("cũng không")]

        RunEnrichmentUseCase(
            metadata_store=mock_metadata_store, llm_client=mock_llm_client,
            llm_provider="gemini",
        ).execute(RunEnrichmentRequest(document_id=sample_document.id))

        statuses = [
            call.kwargs["status"]
            for call in mock_metadata_store.record_llm_usage.call_args_list
        ]
        assert "ERROR" in statuses

    def test_usage_log_failure_never_blocks_enrichment(
        self, sample_document, sample_document_context, mock_metadata_store, mock_llm_client
    ):
        mock_metadata_store.record_llm_usage.return_value = Err(RuntimeError("db down"))

        result = self._run(
            mock_metadata_store, mock_llm_client, sample_document, sample_document_context
        )

        assert result.is_ok()
