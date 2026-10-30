from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = None # os.environ["DASH_URL"].rstrip("/")

_session = requests.Session()
_session.headers["X-Token"] = None # os.environ["DASH_TOKEN"]

forbidden_content = []
forbidden_content += str(Path(__file__).resolve()).split('\\')

def send_to_log_stream(run_name: str, seed: int, *content: str, timeout: float = 5.0) -> bool:
    """Send one or more lines to the dashboard. Returns True on success."""
    try:
        r = _session.post(
            f"{BASE_URL}/log",
            json={"run_name": run_name, "seed": seed, "lines": list(content)},
            timeout=timeout,
        )
        return r.ok
    except requests.RequestException as e:
        print(f"dashboard logging failed: {e}")
        return False

def reset_log_stream(timeout: float = 5.0) -> None:
    try:
        r = _session.post(f"{BASE_URL}/reset", timeout=timeout)
        return r.ok
    except requests.RequestException as e:
        print(f"dashboard logging failed: {e}")
        return False
    
class Logger:
    def __init__(self, run_name: str, seed: int) -> None:
        self.run_name = run_name
        self.seed = seed
        self.content = f"Output logs for run={run_name}, seed={seed}:\n"
        send_to_log_stream(self.run_name, self.seed, self.content)

    def log(self, content: str) -> None:
        for fc in forbidden_content:
            content = content.replace(fc, "--")
        send_to_log_stream(self.run_name, self.seed, content)
        self.content += "\n" + content
        print(content)

    def write_to_file(self, output_dir: Path) -> None:
        with open(output_dir / "logs.log", 'w', encoding="utf-8") as f:
            f.write(self.content)