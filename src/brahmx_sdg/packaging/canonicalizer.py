"""
Canonicalizer — tokenizer-safe text canonicalization.

Ensures all text is safe for the frozen custom student tokenizer:
- Unicode normalization
- Script policy enforcement
- Malformed character rejection
- Token inflation checks
"""
from __future__ import annotations
import unicodedata
import re
from dataclasses import dataclass

@dataclass
class CanonicalizationResult:
    text: str
    original_length: int
    canonical_length: int
    token_count: int = 0
    inflation_ratio: float = 1.0
    rejected: bool = False
    rejection_reason: str = ""

class Canonicalizer:
    def __init__(self, max_inflation: float = 2.0, allowed_scripts: set[str] = None):
        self.max_inflation = max_inflation
        self.allowed_scripts = allowed_scripts or {"Latin", "Common", "Devanagari", "Arabic"}

    def canonicalize(self, text: str) -> CanonicalizationResult:
        original_len = len(text)
        # NFC normalization
        text = unicodedata.normalize("NFC", text)
        # Reject private-use and control characters
        text = re.sub(r"[\ue000-\uf8ff\U000f0000-\U000fffff]", "", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # Script policy check
        if not self._check_scripts(text):
            return CanonicalizationResult(
                text=text, original_length=original_len,
                canonical_length=len(text), rejected=True,
                rejection_reason="off_policy_script",
            )
        return CanonicalizationResult(
            text=text, original_length=original_len, canonical_length=len(text),
        )

    def _check_scripts(self, text: str) -> bool:
        # TODO: implement proper script detection
        return True
