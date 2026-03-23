"""
Corpus Assembler — transforms accepted Gold bundles into static versioned corpora.

This is the Data Factory → Model Factory handoff boundary.
Output: JSONL files (one record per training example) + a release manifest.

Stages:
  1. Collect gold bundles from specified directories
  2. Emit training slices from each bundle
  3. Deduplicate (exact hash)
  4. Apply mixture/filter policy
  5. Export as JSONL (one file per slice type)
  6. Write release manifest
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

from brahmx_sdg.schemas import CorpusVersion, GoldRecordBundle, ReleaseManifest, TrainingStage

logger = structlog.get_logger()


@dataclass
class AssemblyResult:
    version: str
    total_examples: int
    slices_by_type: dict[str, int] = field(default_factory=dict)
    output_path: str = ""
    manifest_path: str = ""


class CorpusAssembler:
    """Assemble a static versioned training corpus from gold bundles."""

    def run(self, manifest_path: Path, output_path: Path) -> AssemblyResult:
        """Full corpus assembly: collect → slice → dedup → export → manifest."""
        import yaml
        with open(manifest_path) as f:
            config = yaml.safe_load(f)

        output_path.mkdir(parents=True, exist_ok=True)
        version = config.get("version", "v0.1")
        export_dir = output_path / version
        export_dir.mkdir(parents=True, exist_ok=True)

        # 1. Collect bundles
        gold_bundles = self._collect_bundles(config.get("gold_paths", []))
        logger.info("collected_bundles", count=len(gold_bundles))

        if not gold_bundles:
            logger.warning("no_gold_bundles_found")
            result = AssemblyResult(version=version, total_examples=0, output_path=str(export_dir))
            self._write_manifest(config, gold_bundles, [], version, output_path)
            return result

        # 2. Emit slices from each bundle
        all_slices = self._emit_all_slices(gold_bundles)
        logger.info("slices_emitted", count=len(all_slices))

        # 3. Deduplicate by content hash
        deduped = self._dedup_slices(all_slices)
        logger.info("slices_after_dedup", count=len(deduped))

        # 4. Apply mixture policy (gold/silver ratio etc.)
        filtered = self._apply_mixture(deduped, config.get("mixture", {}))

        # 5. Export
        slices_by_type = self._export_jsonl(filtered, export_dir)

        # 6. Manifest
        manifest_file = self._write_manifest(config, gold_bundles, filtered, version, output_path)

        total = sum(slices_by_type.values())
        logger.info(
            "corpus_assembled",
            version=version,
            total_examples=total,
            slice_types=list(slices_by_type.keys()),
        )

        return AssemblyResult(
            version=version,
            total_examples=total,
            slices_by_type=slices_by_type,
            output_path=str(export_dir),
            manifest_path=str(manifest_file),
        )

    # ── Collect ───────────────────────────────────────────────────────────────

    def _collect_bundles(self, paths: list[str]) -> list[GoldRecordBundle]:
        bundles: list[GoldRecordBundle] = []
        for p_str in paths:
            p = Path(p_str)
            pattern = "GOLD-*.json"
            files = sorted(p.glob(pattern)) if p.is_dir() else ([p] if p.is_file() else [])
            for f in files:
                try:
                    bundle = GoldRecordBundle.model_validate_json(f.read_text())
                    bundles.append(bundle)
                except Exception as e:
                    logger.warning("bundle_load_failed", file=str(f), error=str(e))
        return bundles

    # ── Emit slices ───────────────────────────────────────────────────────────

    def _emit_all_slices(self, bundles: list[GoldRecordBundle]) -> list[dict[str, Any]]:
        from brahmx_sdg.packaging.slice_emitter import SliceEmitter
        emitter = SliceEmitter()
        all_dicts: list[dict[str, Any]] = []
        for bundle in bundles:
            try:
                slices = emitter.emit(bundle)
                for sl in slices:
                    d = sl.model_dump()
                    # Add content hash for dedup
                    content_str = json.dumps(d["input_data"], sort_keys=True) + json.dumps(d["target_data"], sort_keys=True)
                    d["_content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()
                    all_dicts.append(d)
            except Exception as e:
                logger.warning("slice_emit_failed", bundle=bundle.record_id, error=str(e))
        return all_dicts

    # ── Dedup ─────────────────────────────────────────────────────────────────

    def _dedup_slices(self, slices: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for s in slices:
            h = s.get("_content_hash", s["slice_id"])
            if h not in seen:
                seen.add(h)
                result.append(s)
        return result

    # ── Mixture ───────────────────────────────────────────────────────────────

    def _apply_mixture(
        self, slices: list[dict[str, Any]], mixture: dict
    ) -> list[dict[str, Any]]:
        if not mixture:
            return slices
        # Filter by task type if specified
        allowed_types = mixture.get("task_types", [])
        if allowed_types:
            slices = [s for s in slices if s["task_type"] in allowed_types]
        # Cap per type if specified
        max_per_type = mixture.get("max_per_type", 0)
        if max_per_type > 0:
            by_type: dict[str, list] = {}
            for s in slices:
                by_type.setdefault(s["task_type"], []).append(s)
            slices = [s for group in by_type.values() for s in group[:max_per_type]]
        return slices

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_jsonl(
        self, slices: list[dict[str, Any]], export_dir: Path
    ) -> dict[str, int]:
        """Export one JSONL file per task type + one combined file."""
        by_type: dict[str, list] = {}
        for s in slices:
            # Remove internal hash key
            clean = {k: v for k, v in s.items() if not k.startswith("_")}
            by_type.setdefault(s["task_type"], []).append(clean)

        counts: dict[str, int] = {}
        for task_type, records in by_type.items():
            out_file = export_dir / f"{task_type}.jsonl"
            with open(out_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[task_type] = len(records)
            logger.info("exported_slice_type", task_type=task_type, count=len(records))

        # Combined file
        all_clean = [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in slices
        ]
        combined_file = export_dir / "all_slices.jsonl"
        with open(combined_file, "w") as f:
            for rec in all_clean:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return counts

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _write_manifest(
        self,
        config: dict,
        bundles: list[GoldRecordBundle],
        slices: list[dict],
        version: str,
        output_path: Path,
    ) -> Path:
        manifest = ReleaseManifest(
            version=version,
            gold_bundles_included=[b.record_id for b in bundles],
            notes=config.get("notes", ""),
        )
        manifest_file = output_path / f"manifest_{version}.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        return manifest_file
