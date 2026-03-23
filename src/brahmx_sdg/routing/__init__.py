"""
Teacher Router — Model routing and inference orchestration.

Routes generation requests to the appropriate model pool based on:
- Role (teacher_a, teacher_b, teacher_c, dean, auditor, specialist)
- Workload type (bulk, frontier, audit)
- Model validation status on each runtime
- Cost/latency/quality tradeoffs
- Fallback strategy on pool failure

Placeholder mode: All endpoints point to OpenAI API.
Production mode: Endpoints point to vLLM running on TPU pods.
Both modes use the same OpenAI-compatible /v1/chat/completions interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
import structlog

from brahmx_sdg.schemas import InferenceRuntime

logger = structlog.get_logger()


# ── Model Registry ────────────────────────────────────────────────────────────


class ModelRole(str, Enum):
    TEACHER_A = "teacher_a"
    TEACHER_B = "teacher_b"
    TEACHER_C = "teacher_c"
    DEAN = "dean"
    AUDITOR = "auditor"
    TRANSLATION = "translation"
    LATEX_CODE = "latex_code"
    STRUCTURED_JSON = "structured_json"
    CONVERSATION = "conversation"


class WorkloadClass(str, Enum):
    BULK = "bulk"
    FRONTIER = "frontier"
    AUDIT = "audit"


@dataclass
class ModelEndpoint:
    """A registered model endpoint in the routing table."""
    model_id: str
    model_name: str
    runtime: InferenceRuntime
    base_url: str
    roles: list[ModelRole]
    workload_classes: list[WorkloadClass]
    max_context_length: int = 32768
    validated_on_tpu: bool = False
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    quality_score: float = 0.0
    temperature: float = 0.7
    enabled: bool = True
    # api_key_env: name of the environment variable holding the API key.
    # If set, an "Authorization: Bearer <value>" header is injected.
    # Leave empty for unauthenticated vLLM endpoints.
    api_key_env: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        """Read API key from the configured environment variable."""
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env, "")


@dataclass
class RoutingDecision:
    endpoint: ModelEndpoint
    fallback_endpoints: list[ModelEndpoint]
    routing_reason: str
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0


class ModelRegistry:
    """Registry of all available model endpoints with capability metadata."""

    def __init__(self) -> None:
        self._endpoints: dict[str, ModelEndpoint] = {}

    def register(self, endpoint: ModelEndpoint) -> None:
        self._endpoints[endpoint.model_id] = endpoint
        logger.debug(
            "model_registered",
            model_id=endpoint.model_id,
            runtime=endpoint.runtime.value,
        )

    def get(self, model_id: str) -> Optional[ModelEndpoint]:
        return self._endpoints.get(model_id)

    def get_by_role(self, role: ModelRole) -> list[ModelEndpoint]:
        return [e for e in self._endpoints.values() if role in e.roles and e.enabled]

    def get_by_workload(self, workload: WorkloadClass) -> list[ModelEndpoint]:
        return [
            e for e in self._endpoints.values()
            if workload in e.workload_classes and e.enabled
        ]

    def get_by_runtime(self, runtime: InferenceRuntime) -> list[ModelEndpoint]:
        return [e for e in self._endpoints.values() if e.runtime == runtime and e.enabled]

    @classmethod
    def from_config(cls, config_path: str) -> "ModelRegistry":
        import yaml
        registry = cls()
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for entry in config.get("models", []):
            endpoint = ModelEndpoint(
                model_id=entry["model_id"],
                model_name=entry["model_name"],
                runtime=InferenceRuntime(entry["runtime"]),
                base_url=entry["base_url"],
                roles=[ModelRole(r) for r in entry.get("roles", [])],
                workload_classes=[WorkloadClass(w) for w in entry.get("workload_classes", [])],
                max_context_length=entry.get("max_context_length", 32768),
                validated_on_tpu=entry.get("validated_on_tpu", False),
                cost_per_1k_tokens=entry.get("cost_per_1k_tokens", 0.0),
                avg_latency_ms=entry.get("avg_latency_ms", 0.0),
                quality_score=entry.get("quality_score", 0.0),
                temperature=entry.get("temperature", 0.7),
                enabled=entry.get("enabled", True),
                api_key_env=entry.get("api_key_env", ""),
            )
            registry.register(endpoint)
        return registry


# ── Routing Logic ─────────────────────────────────────────────────────────────


class RoutingStrategy(ABC):
    @abstractmethod
    def select(
        self,
        candidates: list[ModelEndpoint],
        context: dict[str, Any],
    ) -> RoutingDecision:
        ...


class DefaultRoutingStrategy(RoutingStrategy):
    """
    Default routing strategy:
    1. Bulk: prefer vLLM TPU validated → vLLM GPU
    2. Frontier: prefer vLLM GPU → TPU
    3. JetStream only as last resort
    """

    def select(
        self,
        candidates: list[ModelEndpoint],
        context: dict[str, Any],
    ) -> RoutingDecision:
        if not candidates:
            raise NoAvailableEndpointError("No candidate endpoints available")

        workload = WorkloadClass(context.get("workload_class", "bulk"))

        runtime_priority = {
            InferenceRuntime.VLLM_TPU: 0,
            InferenceRuntime.VLLM_GPU: 1,
            InferenceRuntime.HF_TRANSFORMERS: 2,
            InferenceRuntime.JETSTREAM_MAXTEXT: 3,
        }

        if workload == WorkloadClass.FRONTIER:
            runtime_priority[InferenceRuntime.VLLM_GPU] = 0
            runtime_priority[InferenceRuntime.VLLM_TPU] = 1

        sorted_candidates = sorted(
            candidates,
            key=lambda e: (
                runtime_priority.get(e.runtime, 99),
                -e.quality_score,
                e.cost_per_1k_tokens,
            ),
        )

        primary = sorted_candidates[0]
        fallbacks = sorted_candidates[1:3]

        return RoutingDecision(
            endpoint=primary,
            fallback_endpoints=fallbacks,
            routing_reason=(
                f"Selected {primary.model_id} ({primary.runtime.value}) "
                f"for {workload.value} workload"
            ),
            estimated_cost=primary.cost_per_1k_tokens,
            estimated_latency_ms=primary.avg_latency_ms,
        )


class NoAvailableEndpointError(Exception):
    pass


# ── Teacher Router ────────────────────────────────────────────────────────────


class TeacherRouter:
    """
    Main routing service. Resolves a generation request to an endpoint
    and executes inference with automatic fallback.

    Supports both authenticated (OpenAI API) and unauthenticated (local vLLM)
    endpoints via the api_key_env field on each ModelEndpoint.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        strategy: Optional[RoutingStrategy] = None,
    ) -> None:
        self.registry = registry
        self.strategy = strategy or DefaultRoutingStrategy()
        self._client = httpx.Client(timeout=180.0)

    def route(
        self,
        role: ModelRole,
        workload_class: WorkloadClass = WorkloadClass.BULK,
        context: Optional[dict[str, Any]] = None,
    ) -> RoutingDecision:
        candidates = self.registry.get_by_role(role)
        if not candidates:
            raise NoAvailableEndpointError(
                f"No endpoints registered for role {role.value}"
            )
        ctx = {"workload_class": workload_class.value, **(context or {})}
        return self.strategy.select(candidates, ctx)

    def generate(
        self,
        role: ModelRole,
        messages: list[dict[str, str]],
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        workload_class: WorkloadClass = WorkloadClass.BULK,
        context: Optional[dict[str, Any]] = None,
        response_format: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Route and execute a generation request with automatic fallback.

        Returns list of dicts with keys: content, model, runtime, role, finish_reason.
        """
        decision = self.route(role, workload_class, context)
        endpoints_to_try = [decision.endpoint] + decision.fallback_endpoints

        last_error: Optional[Exception] = None
        for endpoint in endpoints_to_try:
            try:
                # Use per-endpoint temperature if caller didn't specify
                actual_temp = temperature if temperature is not None else endpoint.temperature
                results = self._call_endpoint(
                    endpoint=endpoint,
                    messages=messages,
                    n=n,
                    temperature=actual_temp,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                for r in results:
                    r["role"] = role.value
                logger.info(
                    "generation_success",
                    model=endpoint.model_id,
                    runtime=endpoint.runtime.value,
                    role=role.value,
                    n_results=len(results),
                )
                return results
            except Exception as e:
                last_error = e
                logger.warning(
                    "endpoint_failed",
                    model=endpoint.model_id,
                    runtime=endpoint.runtime.value,
                    error=str(e),
                )
                continue

        raise NoAvailableEndpointError(
            f"All endpoints failed for role {role.value}: {last_error}"
        )

    def _call_endpoint(
        self,
        endpoint: ModelEndpoint,
        messages: list[dict[str, str]],
        n: int,
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Call an OpenAI-compatible /v1/chat/completions endpoint.
        Injects Authorization header when api_key_env is configured.
        """
        payload: dict[str, Any] = {
            "model": endpoint.model_name,
            "messages": messages,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = endpoint.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = self._client.post(
            f"{endpoint.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        return [
            {
                "content": choice["message"]["content"],
                "model": endpoint.model_id,
                "runtime": endpoint.runtime.value,
                "finish_reason": choice.get("finish_reason", ""),
                "model_name": endpoint.model_name,
            }
            for choice in data.get("choices", [])
        ]

    @classmethod
    def from_config(cls, config_path: str) -> "TeacherRouter":
        registry = ModelRegistry.from_config(config_path)
        return cls(registry=registry)


# ── Admission Controller ──────────────────────────────────────────────────────


class AdmissionController:
    def __init__(
        self,
        bulk_concurrency: int = 16,
        frontier_concurrency: int = 4,
        audit_concurrency: int = 4,
    ) -> None:
        self.limits = {
            WorkloadClass.BULK: bulk_concurrency,
            WorkloadClass.FRONTIER: frontier_concurrency,
            WorkloadClass.AUDIT: audit_concurrency,
        }
        self._active: dict[WorkloadClass, int] = {w: 0 for w in WorkloadClass}

    def admit(self, workload: WorkloadClass) -> bool:
        return self._active[workload] < self.limits[workload]

    def acquire(self, workload: WorkloadClass) -> bool:
        if self._active[workload] < self.limits[workload]:
            self._active[workload] += 1
            return True
        return False

    def release(self, workload: WorkloadClass) -> None:
        self._active[workload] = max(0, self._active[workload] - 1)
