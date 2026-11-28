"""Chọn model theo task: registry (DB → env → catalog ai-sdk) và ``TaskRoles``."""
import os
import threading
from typing import TYPE_CHECKING

from ai_runtime.catalog import ENRICHMENT
from document_chunk.shared.logger import get_logger

from worker.config import WorkerSettings

if TYPE_CHECKING:
    from ai_runtime import ModelProfile

logger = get_logger(__name__)


# Cấu hình LLM__* của thời "một model cho cả process". Còn bất kỳ biến nào
# trong số này được đặt thì nó vẫn thắng mọi default trong catalog, để một
# deployment chưa kịp đổi sang LLM_PROFILE__* không âm thầm chạy model khác.
_LEGACY_ENV_VARS = ("LLM__PROVIDER", "LLM__MODEL", "LLM__BASE_URL")
_LEGACY_PROFILE_NAME = "env-llm"
# Đổi model trên web có hiệu lực với job mới sau tối đa chừng này giây.
_BINDING_TTL_SECONDS = 30.0
_SUPPORTED_PROVIDERS = ("gemini", "openai-compatible", "deepseek")


def _legacy_profile(llm) -> "ModelProfile":
    """Dựng một profile từ LLM__* để giai đoạn chuyển tiếp không đứt."""
    from ai_runtime import ModelProfile

    if llm.provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {llm.provider}")
    return ModelProfile(
        name=_LEGACY_PROFILE_NAME,
        provider=llm.provider,
        model=llm.model,
        # Key của profile này đến từ settings (đọc cả .env), nên
        # build_model_registry bơm nó vào env mapping dưới tên dưới đây.
        key_env=("LLM__API_KEY",),
        base_url=llm.base_url or None,
        temperature=llm.temperature,
        timeout_seconds=float(llm.timeout_seconds),
        max_retries=llm.max_retries,
        internal=True,
    )


def build_model_registry(settings: WorkerSettings):
    """
    Registry phân giải "task nào dùng model nào", thay cho một client/process.

    Thứ tự: bảng llm_model_bindings (admin chọn trên web, đọc lại mỗi 30s) →
    LLM_PROFILE__<TASK> → LLM_PROFILE → LLM__* (nếu còn đặt kèm key) →
    TASK_DEFAULTS trong catalog của ai-sdk.
    """
    from ai_runtime import EnvConfigSource, ModelRegistry, OverrideConfigSource
    from ai_runtime.catalog import PROFILES

    legacy = _legacy_profile(settings.llm)  # luôn dựng để provider sai lộ ra ngay
    legacy_set = any(os.environ.get(name, "").strip() for name in _LEGACY_ENV_VARS)
    # Profile LLM__* chỉ lấy key từ LLM__API_KEY. Đặt provider/model mà không
    # có key thì nó chắc chắn hỏng — thường là ENV cũ còn nằm trong image —
    # nên không được thắng các default đang chạy được.
    prefer_legacy = legacy_set and bool(settings.llm.api_key)

    env = dict(os.environ)
    if settings.llm.api_key:
        env[legacy.key_env[0]] = settings.llm.api_key

    source = EnvConfigSource(
        env=env,
        fallback_profile=legacy.name if prefer_legacy else None,
    )
    if settings.sql.enabled:
        from worker.adapters.model_binding_store import ModelBindingStore

        source = OverrideConfigSource(
            ModelBindingStore(settings.sql).fetch,
            source,
            ttl_seconds=_BINDING_TTL_SECONDS,
            known_profiles=[name for name, p in PROFILES.items() if not p.internal],
            source_label="db:llm_model_bindings",
        )
    registry = ModelRegistry(
        source=source,
        profiles={**PROFILES, legacy.name: legacy},
        env=env,
    )
    if prefer_legacy:
        logger.warning(
            "worker_container.llm_legacy_env",
            provider=legacy.provider,
            model=legacy.model,
            hint="LLM__* vẫn đang ghi đè; chuyển sang LLM_PROFILE__<TASK> để tách model theo task",
        )
    elif legacy_set:
        logger.warning(
            "worker_container.llm_legacy_env_ignored",
            provider=legacy.provider,
            model=legacy.model,
            reason="LLM__PROVIDER/LLM__MODEL đặt nhưng thiếu LLM__API_KEY",
        )
    return registry


def build_llm_client(settings: WorkerSettings, task: str = ENRICHMENT):
    """Client cho một task, dựng registry riêng — dùng ở chỗ gọi lẻ và test."""
    return build_llm_client_for(build_model_registry(settings), task)


def build_llm_client_for(registry, task: str):
    from worker.adapters.ai_runtime_llm_client import AIRuntimeLLMClient

    resolved = registry.resolve(task)
    logger.info(
        "worker_container.llm_resolved",
        task=task,
        profile=resolved.profile.name,
        provider=resolved.provider,
        model=resolved.model_id,
        chosen_by=resolved.binding.source,
    )
    return AIRuntimeLLMClient(resolved.runtime(), profile=resolved.profile)


class TaskRoles:
    """
    Phân giải task → ``LlmRole`` mỗi lần được hỏi, để admin đổi model trên web
    mà không phải restart worker.

    Container hỏi một lần khi runner lấy use case đầu mỗi job, nên một job
    chạy trọn trên một model và sổ chi phí của nó không bị chia đôi. Client
    giữ theo profile: hỏi lại cùng profile trả lại đúng client cũ.
    """

    def __init__(self, registry) -> None:
        self._registry = registry
        self._clients: dict[tuple, object] = {}
        self._chosen: dict[str, str] = {}
        self._lock = threading.Lock()

    def role(self, task: str):
        from worker.adapters.ai_runtime_llm_client import AIRuntimeLLMClient
        from worker.use_cases.shared.llm_role import LlmRole

        resolved = self._registry.resolve(task)
        key = (resolved.profile.name, resolved.binding.temperature)
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = AIRuntimeLLMClient(resolved.runtime(), profile=resolved.profile)
                self._clients[key] = client
            if self._chosen.get(task) != resolved.profile.name:
                self._chosen[task] = resolved.profile.name
                logger.info(
                    "worker_container.llm_resolved",
                    task=task,
                    profile=resolved.profile.name,
                    provider=resolved.provider,
                    model=resolved.model_id,
                    chosen_by=resolved.binding.source,
                )
        return LlmRole(client=client, provider=client.provider, profile=client.profile_name)
