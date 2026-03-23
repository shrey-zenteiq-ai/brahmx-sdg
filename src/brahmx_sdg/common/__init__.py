"""Common utilities — config, storage, retry, hashing."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import structlog
import yaml

logger = structlog.get_logger()


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class AppConfig:
    """Top-level application configuration."""
    env: str = "dev"  # dev, staging, prod
    gcs_bucket: str = ""
    gcs_prefix: str = "brahmx-sdg"
    provenance_url: str = "http://localhost:8080"
    teacher_router_url: str = "http://localhost:8081"
    log_level: str = "INFO"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # Overlay environment variables
        raw["env"] = os.getenv("BRAHMX_ENV", raw.get("env", "dev"))
        raw["gcs_bucket"] = os.getenv("BRAHMX_GCS_BUCKET", raw.get("gcs_bucket", ""))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            env=os.getenv("BRAHMX_ENV", "dev"),
            gcs_bucket=os.getenv("BRAHMX_GCS_BUCKET", ""),
            gcs_prefix=os.getenv("BRAHMX_GCS_PREFIX", "brahmx-sdg"),
            provenance_url=os.getenv("BRAHMX_PROVENANCE_URL", "http://localhost:8080"),
            teacher_router_url=os.getenv("BRAHMX_TEACHER_ROUTER_URL", "http://localhost:8081"),
        )


# ── Storage Abstraction ──────────────────────────────────────────────────────


class ArtifactStore(Protocol):
    """Protocol for artifact storage backends."""

    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> str: ...
    def get(self, key: str) -> tuple[bytes, dict]: ...
    def put_json(self, key: str, obj: dict) -> str: ...
    def get_json(self, key: str) -> dict: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def exists(self, key: str) -> bool: ...


class LocalArtifactStore:
    """Local filesystem artifact store for development."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> str:
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if metadata:
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            import json
            meta_path.write_text(json.dumps(metadata, indent=2))
        return str(path)

    def get(self, key: str) -> tuple[bytes, dict]:
        path = self.base_path / key
        data = path.read_bytes()
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = {}
        if meta_path.exists():
            import json
            metadata = json.loads(meta_path.read_text())
        return data, metadata

    def put_json(self, key: str, obj: dict) -> str:
        import json
        return self.put(key, json.dumps(obj, indent=2, ensure_ascii=False).encode())

    def get_json(self, key: str) -> dict:
        import json
        data, _ = self.get(key)
        return json.loads(data)

    def list_keys(self, prefix: str) -> list[str]:
        base = self.base_path / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self.base_path)) for p in base.rglob("*") if p.is_file()]

    def exists(self, key: str) -> bool:
        return (self.base_path / key).exists()


class GCSArtifactStore:
    """Google Cloud Storage artifact store for production."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        from google.cloud import storage
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> str:
        blob = self.bucket.blob(self._key(key))
        if metadata:
            blob.metadata = metadata
        blob.upload_from_string(data)
        return f"gs://{self.bucket.name}/{self._key(key)}"

    def get(self, key: str) -> tuple[bytes, dict]:
        blob = self.bucket.blob(self._key(key))
        data = blob.download_as_bytes()
        return data, dict(blob.metadata or {})

    def put_json(self, key: str, obj: dict) -> str:
        import json
        return self.put(key, json.dumps(obj, indent=2, ensure_ascii=False).encode())

    def get_json(self, key: str) -> dict:
        import json
        data, _ = self.get(key)
        return json.loads(data)

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._key(prefix)
        return [b.name for b in self.bucket.list_blobs(prefix=full_prefix)]

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._key(key)).exists()


def get_artifact_store(config: AppConfig) -> ArtifactStore:
    """Factory for artifact stores based on environment."""
    if config.env == "dev" or not config.gcs_bucket:
        return LocalArtifactStore(Path("data/artifacts"))
    return GCSArtifactStore(bucket=config.gcs_bucket, prefix=config.gcs_prefix)


# ── Retry ─────────────────────────────────────────────────────────────────────


def retry(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error("retry_exhausted", func=func.__name__, attempts=max_retries + 1, error=str(e))
                        raise
                    wait = backoff_base ** attempt
                    logger.warning("retry_attempt", func=func.__name__, attempt=attempt + 1, wait=wait, error=str(e))
                    time.sleep(wait)
        return wrapper
    return decorator


# ── Hashing ───────────────────────────────────────────────────────────────────


def deterministic_hash(content: str) -> str:
    """SHA-256 hash of normalized content."""
    normalized = " ".join(content.strip().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
