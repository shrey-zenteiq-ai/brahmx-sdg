"""LaTeX Compile Validator — compile loop with error feedback retry."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess, tempfile
import structlog

logger = structlog.get_logger()
MAX_COMPILE_RETRIES = 5

@dataclass
class LaTeXCompileResult:
    check_id: str
    passed: bool
    compile_attempts: int = 0
    errors: list[str] = None
    pdf_path: str = ""

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class LaTeXCompileValidator:
    """Compile LaTeX artifacts and retry with error feedback."""

    def __init__(self, max_retries: int = MAX_COMPILE_RETRIES):
        self.max_retries = max_retries

    def compile(self, latex_source: str, check_id: str = "") -> LaTeXCompileResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "doc.tex"
            tex_path.write_text(latex_source)
            for attempt in range(self.max_retries):
                try:
                    result = subprocess.run(
                        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                        cwd=tmpdir, capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        return LaTeXCompileResult(
                            check_id=check_id, passed=True,
                            compile_attempts=attempt + 1,
                            pdf_path=str(Path(tmpdir) / "doc.pdf"),
                        )
                except subprocess.TimeoutExpired:
                    pass
            return LaTeXCompileResult(
                check_id=check_id, passed=False,
                compile_attempts=self.max_retries,
                errors=["Compilation failed after max retries"],
            )
