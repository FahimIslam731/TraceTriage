"""Docker-based Python code executor — ported from CausalFlow experiments/humaneval."""
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import docker
from docker.errors import DockerException
from requests.exceptions import ReadTimeout


class DockerCodeExecutor:
    """Runs Python code inside a constrained Docker container."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 10,
        mem_limit: str = "512m",
        workdir: str = "/workspace",
        env: Optional[Dict[str, str]] = None,
    ):
        self.image = image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.workdir = workdir
        self.env = {"PYTHONDONTWRITEBYTECODE": "1"}
        if env:
            self.env.update(env)
        self.client = docker.from_env()

    def run_code(
        self,
        code: str,
        command: Optional[List[str]] = None,
        filename: str = "solution.py",
    ) -> Tuple[bool, str]:
        """Execute code in Docker. Returns (success, combined_logs)."""
        tmp_dir = tempfile.mkdtemp(prefix="tracetriage_mbpp_")
        file_path = os.path.join(tmp_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        container = None
        logs = ""
        try:
            container = self.client.containers.run(
                self.image,
                command or ["python", filename],
                detach=True,
                volumes={tmp_dir: {"bind": self.workdir, "mode": "rw"}},
                working_dir=self.workdir,
                network_disabled=True,
                mem_limit=self.mem_limit,
                stdout=True,
                stderr=True,
                environment=self.env,
            )

            result = container.wait(timeout=self.timeout)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            status_code = result.get("StatusCode", 1)
            success = status_code == 0

            if not logs.strip() and status_code != 0:
                try:
                    logs = container.logs(stdout=True, stderr=True, tail=100).decode("utf-8", errors="replace")
                except Exception:
                    logs = f"Container exited with status code {status_code} but no logs captured"
        except ReadTimeout:
            if container:
                container.kill()
            success = False
            logs = logs + "\nTimeout while running code."
        except DockerException as e:
            success = False
            logs = logs + f"\nDocker error: {e}"
        finally:
            if container:
                try:
                    container.remove(force=True)
                except DockerException:
                    pass
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return success, logs
