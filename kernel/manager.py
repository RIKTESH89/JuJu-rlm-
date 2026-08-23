"""A persistent IPython kernel, run in its own virtualenv.

The kernel is started lazily on the first execute() call, keeps its namespace
between calls, and is driven over the Jupyter protocol via jupyter_client.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import AsyncKernelManager

AGENT_DIR = Path.home() / ".myagent"
VENV_DIR = AGENT_DIR / "kernel-venv"
SPEC_ROOT = AGENT_DIR / "kernels"
MARKER_PATH = VENV_DIR / "myagent-kernel.json"

KERNEL_NAME = "myagent"
PACKAGES = ["ipykernel", "rlm"]
MARKER_VERSION = 2

# The in-kernel half of the host bridge, installed from source into the venv.
RLM_SOURCE = Path(__file__).resolve().parent.parent / "rlm_package"

SOCKET_ENV = "MYAGENT_HOST_SOCKET"

# Run silently once the kernel is up, so these names are always present.
BOOTSTRAP = """
import os, json, re
from pathlib import Path
import rlm
"""

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Printed by the kernel venv's own interpreter to report what is installed.
PROBE = """
import json, sys
from importlib import metadata

packages = {}
for name in sys.argv[1:]:
    try:
        packages[name] = metadata.version(name)
    except Exception:
        packages[name] = None

print(json.dumps({
    "python": ".".join(str(n) for n in sys.version_info[:3]),
    "packages": packages,
}))
"""


@dataclass
class ExecResult:
    """Everything one cell produced."""

    stdout: str = ""
    stderr: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    execution_count: Optional[int] = None
    duration_ms: float = 0.0


def rlm_fingerprint() -> str:
    """Hash the rlm sources, so editing them invalidates the built venv."""
    digest = hashlib.sha256()
    for path in sorted(RLM_SOURCE.rglob("*")):
        if path.is_file() and ".egg-info" not in str(path):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def venv_python(venv: Path) -> Path:
    """Path to the interpreter inside a virtualenv."""
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


async def run_command(*cmd: str) -> "tuple[int, str]":
    """Run a command to completion, returning (exit code, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")


async def probe(python: Path) -> Optional[dict]:
    """Ask an interpreter for its version and installed package versions."""
    code, out = await run_command(str(python), "-c", PROBE, *PACKAGES)
    if code != 0:
        return None
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


class KernelManager:
    """Owns one kernel process and serializes execution against it."""

    def __init__(
        self,
        host_socket: Optional[Path] = None,
        env_extra: Optional[dict] = None,
    ) -> None:
        self.host_socket = host_socket
        self.env_extra = env_extra or {}
        self._km: Optional[AsyncKernelManager] = None
        self._kc = None
        self._exec_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._kc is not None

    # -- venv ------------------------------------------------------------

    async def _is_stale(self) -> bool:
        """True if the kernel venv is missing, incomplete, or has drifted."""
        python = venv_python(VENV_DIR)
        if not python.exists() or not MARKER_PATH.exists():
            return True

        try:
            marker = json.loads(MARKER_PATH.read_text())
        except (OSError, ValueError):
            return True

        if marker.get("marker_version") != MARKER_VERSION:
            return True

        actual = await probe(python)
        if actual is None:
            return True
        if any(version is None for version in actual["packages"].values()):
            return True

        if marker.get("rlm_source") != rlm_fingerprint():
            return True

        return (
            actual["python"] != marker.get("python")
            or actual["packages"] != marker.get("packages")
        )

    async def _build_venv(self) -> None:
        """Create the kernel venv from scratch and record a marker."""
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)

        python = venv_python(VENV_DIR)
        uv = shutil.which("uv")

        # rlm is installed from local source, not from an index.
        installs = ["ipykernel", str(RLM_SOURCE)]

        if uv:
            steps: List[List[str]] = [
                [uv, "venv", str(VENV_DIR)],
                [uv, "pip", "install", "--python", str(python)] + installs,
            ]
        else:
            steps = [
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                [str(python), "-m", "pip", "install", "--quiet"] + installs,
            ]

        for step in steps:
            code, out = await run_command(*step)
            if code != 0:
                raise RuntimeError(
                    "kernel venv setup failed: %s\n%s" % (" ".join(step), out)
                )

        installed = await probe(python)
        if installed is None:
            raise RuntimeError("kernel venv built but could not be inspected")

        installed["marker_version"] = MARKER_VERSION
        installed["rlm_source"] = rlm_fingerprint()
        MARKER_PATH.write_text(json.dumps(installed, indent=2))

    async def _kernel_python(self) -> Path:
        """The interpreter the kernel will run under."""
        override = os.environ.get("MYAGENT_KERNEL_PYTHON")
        if override:
            return Path(override)

        if await self._is_stale():
            await self._build_venv()
        return venv_python(VENV_DIR)

    def _write_kernelspec(self, python: Path) -> Path:
        """Write a kernelspec pointing at our interpreter; return its search root."""
        spec_dir = SPEC_ROOT / KERNEL_NAME
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        str(python),
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": KERNEL_NAME,
                    "language": "python",
                },
                indent=2,
            )
        )
        return SPEC_ROOT

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Boot the kernel. Idempotent; called lazily by execute()."""
        async with self._start_lock:
            if self._kc is not None:
                return

            python = await self._kernel_python()
            spec_root = self._write_kernelspec(python)

            specs = KernelSpecManager()
            specs.kernel_dirs.insert(0, str(spec_root))

            environment = os.environ.copy()
            environment.update(self.env_extra)
            if self.host_socket is not None:
                environment[SOCKET_ENV] = str(self.host_socket)

            km = AsyncKernelManager(kernel_name=KERNEL_NAME, kernel_spec_manager=specs)
            await km.start_kernel(env=environment)

            kc = km.client()
            kc.start_channels()
            await kc.wait_for_ready(timeout=60)

            self._km, self._kc = km, kc
            await self._bootstrap()

    async def _bootstrap(self) -> None:
        """Seed the namespace. Not shown to the model, but failures are loud."""
        result = ExecResult()
        msg_id = self._kc.execute(BOOTSTRAP, store_history=False)
        await self._collect(msg_id, result)
        if result.error:
            raise RuntimeError("kernel bootstrap failed:\n" + result.error)

    async def interrupt(self) -> None:
        """Interrupt the running cell, leaving the namespace intact."""
        if self._km is not None:
            await self._km.interrupt_kernel()

    async def shutdown(self) -> None:
        """Stop the kernel and release its channels."""
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            await self._km.shutdown_kernel(now=True)
        self._km = self._kc = None

    # -- execution -------------------------------------------------------

    async def _collect(self, msg_id: str, result: ExecResult) -> None:
        """Drain iopub for one request, until that request goes idle."""
        while True:
            message = await self._kc.get_iopub_msg()
            if message.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            kind = message["msg_type"]
            content = message["content"]

            if kind == "stream":
                if content.get("name") == "stderr":
                    result.stderr += content.get("text", "")
                else:
                    result.stdout += content.get("text", "")

            elif kind in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain")
                if text is not None:
                    result.result = (
                        text if result.result is None else result.result + "\n" + text
                    )
                if content.get("execution_count") is not None:
                    result.execution_count = content["execution_count"]

            elif kind == "error":
                traceback = "\n".join(content.get("traceback", []))
                result.error = ANSI.sub("", traceback) or "{}: {}".format(
                    content.get("ename"), content.get("evalue")
                )

            elif kind == "execute_input":
                if content.get("execution_count") is not None:
                    result.execution_count = content["execution_count"]

            elif kind == "status" and content.get("execution_state") == "idle":
                return

    async def execute(self, code: str, timeout: Optional[float] = None) -> ExecResult:
        """Run one cell. timeout=None waits forever."""
        async with self._exec_lock:
            if self._kc is None:
                await self.start()

            result = ExecResult()
            started = time.monotonic()
            msg_id = self._kc.execute(code)

            try:
                await asyncio.wait_for(self._collect(msg_id, result), timeout)
            except asyncio.TimeoutError:
                await self.interrupt()
                try:
                    await asyncio.wait_for(self._collect(msg_id, result), 10)
                except asyncio.TimeoutError:
                    pass
                result.error = result.error or (
                    "TimeoutError: execution exceeded %ss" % timeout
                )

            result.duration_ms = (time.monotonic() - started) * 1000
            return result
