import torch
import subprocess, shutil, datetime, tomllib, os, base64, requests
from pathlib import Path

from .srnn import SRNN

REPO = "Emiel-Nagel/Thesis-Code"
CODE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CODE_DIR / "outputs"
RUNS_DIR = CODE_DIR / "runs"
RUNS_TEMP_DIR = RUNS_DIR / "_runs_tmp"
CONFIG_PATH = RUNS_DIR / "config.toml"

def load_config() -> dict:
    with open(CONFIG_PATH, 'rb') as cfg_file:
        return tomllib.load(cfg_file)

def generate_run_id() -> str:
    return str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def get_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, cwd=CODE_DIR,
    ).stdout.strip()

def create_output_dir(run_name: str) -> Path:
    output_dir = OUTPUT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, output_dir / "config.toml")
    shutil.copy2(CODE_DIR / "scripts" / "analysis.ipynb", output_dir / "analysis.ipynb")
    return output_dir

def save_net(net: SRNN, optimizer: torch.optim.Optimizer, epoch: int, save_path: Path) -> None:
    try:
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, save_path)
        print(f"Successfully saved net as {save_path.name}")
    except Exception as e:
        print(f"Failed to save net as {save_path.name}, caused by {e!r}")

def push_output(output_dir: Path, branch: str = "main") -> None:
    token = os.environ["GITHUB_TOKEN"]
    message = f"Successfully completed run, added results for run {output_dir.name}"
    repo_url = f"https://api.github.com/repos/{REPO}"
    rel = output_dir.resolve().relative_to(CODE_DIR.resolve())

    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    def call(method: str, path: str, **kw) -> dict:
        r = s.request(method, f"{repo_url}{path}", timeout=120, **kw)
        r.raise_for_status()
        return r.json()

    files = [p for p in output_dir.rglob("*") if p.is_file()]
    too_big = [p for p in files if p.stat().st_size > 95 * 1024**2]
    if too_big:
        raise RuntimeError(f"Files too large for GitHub: {too_big}")
    if not files:
        print("Nothing to push")
        return

    # Current head of the branch
    head_sha = call("GET", f"/git/ref/heads/{branch}")["object"]["sha"]
    base_tree = call("GET", f"/git/commits/{head_sha}")["tree"]["sha"]

    # One blob per file
    entries = []
    for p in files:
        blob = call("POST", "/git/blobs", json={
            "content": base64.b64encode(p.read_bytes()).decode(),
            "encoding": "base64",
        })
        repo_path = (rel / p.relative_to(output_dir)).as_posix()
        entries.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    # New tree layered on top of the existing one (other repo files untouched)
    tree = call("POST", "/git/trees", json={"base_tree": base_tree, "tree": entries})
    if tree["sha"] == base_tree:
        print("Nothing new to commit")
        return

    commit = call("POST", "/git/commits", json={
        "message": message, "tree": tree["sha"], "parents": [head_sha],
    })
    call("PATCH", f"/git/refs/heads/{branch}", json={"sha": commit["sha"]})
    print(f"Pushed {len(files)} files to {REPO}@{branch}")