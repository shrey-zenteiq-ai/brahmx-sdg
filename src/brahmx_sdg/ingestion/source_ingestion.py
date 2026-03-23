"""
Source Ingestion Pipeline.

Ingests authoritative scientific sources into normalized, policy-tagged
SourceAtoms stored in the KB as chunked JSON files.

Supported formats: .txt, .md, .json (doc or chunk list), .jsonl, .py, .pdf (text-only)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog

from brahmx_sdg.schemas import SourceAtom, SourceDocument
from brahmx_sdg.common import deterministic_hash

logger = structlog.get_logger()

# Token-level estimate: 1 token ≈ 4 characters for English
CHARS_PER_CHUNK = 2048     # ~512 tokens
CHUNK_OVERLAP_CHARS = 256  # ~64 tokens overlap


@dataclass
class IngestionResult:
    doc_count: int = 0
    atom_count: int = 0
    rejected_count: int = 0
    decontam_flagged: int = 0


class SourceIngestionPipeline:
    """
    Main ingestion pipeline.

    Stages:
      1. Discover files (txt, md, json, jsonl, py)
      2. Parse text content from each file
      3. Chunk into overlapping SourceAtoms
      4. Decontamination scan (n-gram based, optional)
      5. Write atoms to KB chunks directory
    """

    def __init__(
        self,
        chunk_size: int = CHARS_PER_CHUNK,
        chunk_overlap: int = CHUNK_OVERLAP_CHARS,
        decontam_manifests: Optional[list[Path]] = None,
        domain: str = "",
        license_tag: str = "unknown",
        redistribution_allowed: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.decontam_manifests = decontam_manifests or []
        self.domain = domain
        self.license_tag = license_tag
        self.redistribution_allowed = redistribution_allowed

    def run(self, source_path: Path, output_dir: Path) -> dict[str, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        result = IngestionResult()

        for file_path in self._discover_files(source_path):
            try:
                doc, text = self._parse_document(file_path)
                if not text.strip():
                    result.rejected_count += 1
                    continue
                atoms = self._chunk_document(doc, text)
                atoms = self._decontam_scan(atoms)
                result.doc_count += 1
                result.atom_count += len(atoms)
                result.decontam_flagged += sum(
                    1 for a in atoms if a.decontam_status == "flagged"
                )
                self._write_atoms(atoms, chunks_dir, file_path.stem)
                logger.info(
                    "doc_ingested",
                    file=file_path.name,
                    atoms=len(atoms),
                )
            except Exception as e:
                logger.error("ingestion_failed", file=str(file_path), error=str(e))
                result.rejected_count += 1

        logger.info(
            "ingestion_complete",
            docs=result.doc_count,
            atoms=result.atom_count,
            rejected=result.rejected_count,
        )
        return {
            "doc_count": result.doc_count,
            "atom_count": result.atom_count,
            "rejected_count": result.rejected_count,
            "decontam_flagged": result.decontam_flagged,
        }

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _discover_files(self, source_path: Path) -> list[Path]:
        extensions = {".pdf", ".tex", ".json", ".py", ".md", ".txt", ".jsonl", ".ndjson"}
        if source_path.is_file():
            return [source_path]
        return sorted(p for p in source_path.rglob("*") if p.suffix.lower() in extensions)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_document(self, file_path: Path) -> tuple[SourceDocument, str]:
        """Return (SourceDocument metadata, plain text content)."""
        suffix = file_path.suffix.lower()
        raw = file_path.read_bytes()
        text = ""

        if suffix in (".txt", ".md", ".py", ".tex"):
            text = raw.decode("utf-8", errors="replace")

        elif suffix in (".json",):
            try:
                obj = json.loads(raw)
                text = self._json_to_text(obj)
            except json.JSONDecodeError:
                text = raw.decode("utf-8", errors="replace")

        elif suffix in (".jsonl", ".ndjson"):
            lines = []
            for line in raw.decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lines.append(self._json_to_text(obj))
                except json.JSONDecodeError:
                    lines.append(line)
            text = "\n\n".join(lines)

        elif suffix == ".pdf":
            text = self._parse_pdf(raw, file_path)

        else:
            text = raw.decode("utf-8", errors="replace")

        # Light normalisation
        text = self._normalize_text(text)

        doc = SourceDocument(
            title=file_path.stem,
            source_type=suffix.lstrip("."),
            uri=str(file_path.resolve()),
            license=self.license_tag,
            redistribution_allowed=self.redistribution_allowed,
            domain=self.domain,
            raw_hash=deterministic_hash(text),
        )
        return doc, text

    def _json_to_text(self, obj: Any) -> str:
        """Flatten a JSON object to plain text for chunking."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, list):
            parts = []
            for item in obj:
                if isinstance(item, dict):
                    # Prefer common text fields
                    for field in ("text", "content", "body", "passage", "sentence"):
                        if field in item:
                            parts.append(str(item[field]))
                            break
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n\n".join(parts)
        if isinstance(obj, dict):
            parts = []
            for key in ("text", "content", "body", "passage", "abstract", "description"):
                if key in obj and obj[key]:
                    parts.append(str(obj[key]))
            if not parts:
                parts = [f"{k}: {v}" for k, v in obj.items() if isinstance(v, str)]
            return "\n".join(parts)
        return str(obj)

    def _parse_pdf(self, raw: bytes, path: Path) -> str:
        """Extract text from PDF. Falls back to empty string if no library available."""
        try:
            import io
            # Try pypdf (lightweight)
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except ImportError:
                pass
            # Try pdfminer
            try:
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                output = io.StringIO()
                extract_text_to_fp(io.BytesIO(raw), output, laparams=LAParams())
                return output.getvalue()
            except ImportError:
                pass
            logger.warning("pdf_parse_skipped", path=str(path), reason="no PDF library installed")
            return ""
        except Exception as e:
            logger.warning("pdf_parse_failed", path=str(path), error=str(e))
            return ""

    def _normalize_text(self, text: str) -> str:
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_document(self, doc: SourceDocument, text: str) -> list[SourceAtom]:
        """Sliding-window character-based chunking with paragraph-boundary preference."""
        atoms: list[SourceAtom] = []

        # Try paragraph-aware splitting first
        paragraphs = re.split(r"\n\n+", text)
        chunks = self._merge_paragraphs(paragraphs, self.chunk_size, self.chunk_overlap)

        for idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            atom = SourceAtom(
                doc_id=doc.doc_id,
                text=chunk_text,
                chunk_index=idx,
                domain=doc.domain,
                license=doc.license,
                redistribution_allowed=doc.redistribution_allowed,
                token_count=len(chunk_text) // 4,  # rough estimate
                metadata={
                    "source_uri": doc.uri,
                    "title": doc.title,
                    "source_type": doc.source_type,
                    "chunk_char_len": len(chunk_text),
                },
            )
            atoms.append(atom)

        return atoms

    def _merge_paragraphs(
        self,
        paragraphs: list[str],
        max_chars: int,
        overlap_chars: int,
    ) -> list[str]:
        """
        Greedily merge paragraphs into chunks not exceeding max_chars.
        Adjacent chunks share overlap_chars of trailing text.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)

            if current_len + para_len + 2 > max_chars and current_parts:
                # Flush current chunk
                chunk = "\n\n".join(current_parts)
                chunks.append(chunk)
                # Carry overlap: keep last N chars
                overlap_text = chunk[-overlap_chars:] if overlap_chars else ""
                current_parts = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            # If a single paragraph is bigger than max_chars, split it hard
            if para_len > max_chars:
                for start in range(0, para_len, max_chars - overlap_chars):
                    sub = para[start: start + max_chars]
                    chunks.append(sub)
                current_parts = []
                current_len = 0
            else:
                current_parts.append(para)
                current_len += para_len + 2  # +2 for "\n\n"

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    # ── Decontamination ───────────────────────────────────────────────────────

    def _decontam_scan(self, atoms: list[SourceAtom]) -> list[SourceAtom]:
        """Flag atoms whose text appears in decontamination manifests (n-gram match)."""
        if not self.decontam_manifests:
            for atom in atoms:
                atom.decontam_status = "clean"
            return atoms

        # Load reference n-grams lazily
        ref_ngrams: set[str] = set()
        for manifest_path in self.decontam_manifests:
            try:
                with open(manifest_path) as f:
                    for line in f:
                        ref_ngrams.add(line.strip().lower())
            except Exception:
                pass

        for atom in atoms:
            ngrams = _extract_ngrams(atom.text.lower(), n=13)
            if any(ng in ref_ngrams for ng in ngrams):
                atom.decontam_status = "flagged"
            else:
                atom.decontam_status = "clean"
        return atoms

    # ── Write ─────────────────────────────────────────────────────────────────

    def _write_atoms(
        self, atoms: list[SourceAtom], chunks_dir: Path, stem: str
    ) -> None:
        """Write atoms as a JSON array file in chunks_dir."""
        out_file = chunks_dir / f"{stem}.json"
        serialized = [
            {
                "atom_id": a.atom_id,
                "doc_id": a.doc_id,
                "text": a.text,
                "chunk_index": a.chunk_index,
                "domain": a.domain,
                "license": a.license,
                "redistribution_allowed": a.redistribution_allowed,
                "token_count": a.token_count,
                "decontam_status": a.decontam_status,
                "source": a.metadata.get("source_uri", ""),
                "title": a.metadata.get("title", ""),
            }
            for a in atoms
        ]
        out_file.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False)
        )

    @classmethod
    def from_config(cls, config_path: Path) -> "SourceIngestionPipeline":
        import yaml
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            config = {}
        return cls(
            chunk_size=config.get("chunk_size", CHARS_PER_CHUNK),
            chunk_overlap=config.get("chunk_overlap", CHUNK_OVERLAP_CHARS),
            domain=config.get("domain", ""),
            license_tag=config.get("license", "unknown"),
            redistribution_allowed=config.get("redistribution_allowed", True),
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def _extract_ngrams(text: str, n: int = 13) -> list[str]:
    words = text.split()
    return [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]
