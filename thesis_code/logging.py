from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import os, requests

_session = requests.Session()

def send_to_log_stream(run_name: str, seed: str, *content: str, timeout: float = 5.0) -> None:
    """Send one or more lines to the dashboard. Returns True on success."""
    try:
        r = _session.post(
            os.environ["DASH_URL"].rstrip("/") + "/log",
            headers={"X-Token": os.environ["DASH_TOKEN"]},
            json={"seed": seed, "lines": list(content)},
            timeout=timeout,
        )
        return r.ok
    except requests.RequestException:
        return False

class LoggerInstance:
    def __init__(self, seed: int, name: str) -> None:
        self.seed = seed
        self.name = name
        self.content = f"Output logs for {name}:\n"

    def write(self, content: str) -> None:
        send_to_log_stream(self.name, self.seed, content)
        self.content += "\n" + content
        print(content)

@contextmanager
def get_loggers(output_dir: Path, run_name: str, seeds: list[int]) -> Generator[list[LoggerInstance]]:
    loggers = [LoggerInstance(seed=i, name=f"{run_name}--seed_{i}") for i in seeds]
    try:
        yield loggers
    finally:
        with open(output_dir / "logs.log", "w") as f:
            f.write("\n\n".join([l.content for l in loggers]))