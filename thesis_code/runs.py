import torch
import subprocess, shutil, datetime, tomllib
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

def push_output(output_dir: Path, message: str | None = None) -> None:
    """Commit and push only output_dir (not the rest of the repo)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=CODE_DIR, capture_output=True, text=True, check=True)

    # Raises ValueError if output_dir isn't inside the repo
    rel = str(output_dir.resolve().relative_to(CODE_DIR.resolve()))
    message = message or f"Add results for run {output_dir.name}"

    # GitHub rejects files over 100 MB
    too_big = [p for p in output_dir.rglob("*") if p.is_file() and p.stat().st_size > 95 * 1024**2]
    if too_big:
        raise RuntimeError(f"Files too large for GitHub: {too_big}")

    try:
        git("add", "--", rel)
        if not git("status", "--porcelain", "--", rel).stdout.strip():
            print("Nothing new to commit")
            return
        git("commit", "-m", message, "--", rel)   # pathspec limits the commit to output_dir
        git("push", "origin", "HEAD")
        print(f"Pushed {rel} to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Git push failed: {e.stderr}")



# def save_run(run_id: str, cfg: dict, net: SRNN, **recorders: Recorder) -> Path:
#     RUNS_TEMP_DIR.mkdir(exist_ok=True)
#     run_dir = Path(tempfile.mkdtemp(prefix=f"run-{run_id}-", dir=RUNS_TEMP_DIR))

#     net.reset()
#     net_cpu = copy.deepcopy(net).cpu()
#     for m in net_cpu.modules():
#         if hasattr(m, "spike_grad"):
#             m.spike_grad = None

#     state = {
#         "cfg": cfg,
#         "net": net_cpu,
#         **{name: rec.to_state() for name, rec in recorders.items() if rec is not None}
#     }
#     buf = io.BytesIO()
#     torch.save(state, buf)
#     (run_dir / "run.pt.gz").write_bytes(gzip.compress(buf.getvalue()))
#     return run_dir

# def push_run(run_dir: Path, run_id: str, cfg: dict) -> None:
#     subprocess.run(
#         ["gh", "release", "create", f"run-{run_id}", str(run_dir / "run.pt.gz"),
#          "-R", REPO, "--title", run_id, "--notes", yaml.safe_dump(cfg),
#          "--target", cfg["git_commit"]],
#         check=True,
#     )
#     # shutil.rmtree(run_dir)

# def load_run(path: Path) -> tuple[SRNN, dict, dict]:
#     with open(path, "rb") as f:
#         state = torch.load(io.BytesIO(gzip.decompress(f.read())), weights_only=True, map_location="cpu")
#     # recs = {name: sd for name, sd in state.items() if name not in ["cfg", "net"]}
#     cfg = state["cfg"]
#     net = state["net"]          # whatever args your constructor takes

#     # recs = {name: cls.from_state(state[name]) for name, cls in classes.items() if state[name] is not None}
#     return net, cfg, state