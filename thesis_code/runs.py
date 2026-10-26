import torch
import subprocess, shutil, datetime, tomllib, os, base64, requests
from pathlib import Path
from collections.abc import Iterator
import logging

from .srnn import SRNN

REPO = "Emiel-Nagel/Thesis-Code"
CODE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = CODE_DIR / "outputs"
RUNS_DIR = CODE_DIR / "runs"
RUNS_TEMP_DIR = RUNS_DIR / "_runs_tmp"
CONFIG_DIR = RUNS_DIR / "scheduled"
DONE_DIR = CONFIG_DIR / "_done"

def load_config(config_path: Path) -> dict:
    with open(config_path, 'rb') as cfg_file:
        return tomllib.load(cfg_file)

def generate_run_id() -> str:
    return str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def get_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, cwd=CODE_DIR,
    ).stdout.strip()

def create_output_dir(config_path: Path, run_name: str) -> Path:
    output_dir = OUTPUT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.toml")
    shutil.copy2(CODE_DIR / "scripts" / "analysis.ipynb", output_dir / "analysis.ipynb")

    helpers_dir = output_dir / "helpers"
    helpers_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CODE_DIR / "thesis_code" / "plotting.py", helpers_dir / "plotting.py")
    shutil.copy2(CODE_DIR / "thesis_code" / "analysis.py", helpers_dir / "analysis.py")
    (helpers_dir / "__init__.py").touch()
    return output_dir

def iter_scheduled_runs() -> Iterator[tuple[Path, dict, str, Path]]:
    DONE_DIR.mkdir(exist_ok=True)
    print(sorted(CONFIG_DIR.glob("*.toml")))
    for config_path in sorted(CONFIG_DIR.glob("*.toml")):
        try:
            cfg = load_config(config_path)
            run_name = f"{generate_run_id()}-{cfg["run_name"]}"
            output_dir = create_output_dir(config_path, run_name)
        except (tomllib.TOMLDecodeError, KeyError, ValueError, OSError) as e:
            logging.error("Error: Skipping %s: %s", config_path.name, e)
            continue
        yield (config_path, cfg, run_name, output_dir)

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

def push_output(output_dir: Path) -> None:
    """
    Pushes the output to a new branch under 'outputs/'
    """
    token = os.environ["GITHUB_TOKEN"]
    message = f"Successfully completed run, added results for run {output_dir.name}"
    repo_url = f"https://api.github.com/repos/{REPO}"
    branch_name = f"outputs/{output_dir.name}"
    # rel = output_dir.resolve().relative_to(CODE_DIR.resolve())

    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })

    def call(method: str, path: str, **kw) -> dict:
        r = s.request(method, f"{repo_url}{path}", timeout=120, **kw)
        if not r.ok:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text}")
        return r.json()

    files = [p for p in output_dir.rglob("*") if p.is_file()]
    if not files:
        print("Nothing to push")
        return
    too_big = [p for p in files if p.stat().st_size > 95 * 1024**2]
    if too_big:
        raise RuntimeError(f"Files too large for GitHub: {too_big}")

    r = s.get(f"{repo_url}/git/ref/heads/{branch_name}", timeout=120)
    if r.status_code == 200:
        raise RuntimeError(f"Branch {branch_name} already exists")
    if r.status_code != 404:
        raise RuntimeError(f"Could not check branch: {r.status_code}: {r.text}")

    # One blob per file
    entries = []
    for p in files:
        size = p.stat().st_size
        try:
            blob = call("POST", "/git/blobs", json={
                "content": base64.b64encode(p.read_bytes()).decode(),
                "encoding": "base64",
            })
        except RuntimeError as e:
            raise RuntimeError(f"Failed on {p} ({size / 1024**2:.1f} MB): {e}") from e
        repo_path = p.relative_to(output_dir).as_posix()
        entries.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    # Tree with NO base_tree -> contains only these files
    tree = call("POST", "/git/trees", json={"tree": entries})

    try:
        commit = call("POST", "/git/commits", json={
            "message": message, "tree": tree["sha"], "parents": [],
        })
        call("POST", "/git/refs", json={"ref": f"refs/heads/{branch_name}", "sha": commit["sha"]})
    except Exception as e:
        print(f"Error pushing outputs to new branch {REPO}@{branch_name}: {e}")
        print(f"Keeping {output_dir} since the push failed")
        raise

    print(f"Created {REPO}@{branch_name} with {len(files)} files")
    shutil.rmtree(output_dir)
    print("Removed output directory after push")

def remove_config(config_path: Path) -> None:
    config_path.rename(DONE_DIR / f"{config_path.name}.toml")