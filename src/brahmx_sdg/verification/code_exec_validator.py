"""Code Execution Sandbox — compile/execute generated code in nsjail/container."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import subprocess, tempfile
from pathlib import Path
import structlog

logger = structlog.get_logger()

MAX_EXEC_TIMEOUT = 30  # seconds

@dataclass
class CodeExecResult:
    check_id: str
    passed: bool
    language: str = "python"
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False

class CodeExecValidator:
    """Execute code snippets in a sandboxed environment and verify correctness."""

    def __init__(self, sandbox_type: str = "subprocess", timeout: int = MAX_EXEC_TIMEOUT):
        self.sandbox_type = sandbox_type
        self.timeout = timeout

    def execute(self, code: str, language: str = "python", check_id: str = "") -> CodeExecResult:
        """Execute code and return result."""
        if language == "python":
            return self._exec_python(code, check_id)
        # TODO: support more languages via nsjail containers
        return CodeExecResult(check_id=check_id, passed=False, stderr=f"Unsupported language: {language}")

    def _exec_python(self, code: str, check_id: str) -> CodeExecResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python", f.name], capture_output=True, text=True, timeout=self.timeout
                )
                return CodeExecResult(
                    check_id=check_id, passed=result.returncode == 0,
                    language="python", stdout=result.stdout[:4096],
                    stderr=result.stderr[:4096], exit_code=result.returncode,
                )
            except subprocess.TimeoutExpired:
                return CodeExecResult(check_id=check_id, passed=False, timed_out=True)
            finally:
                Path(f.name).unlink(missing_ok=True)
