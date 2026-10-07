import torch
import subprocess, tempfile, shutil, yaml, io, gzip, datetime, copy, tomllib
from pathlib import Path

from .recording import Recorder
from .srnn import SRNN

REPO = "Emiel-Nagel/Thesis-Code"
CODE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = CODE_DIR / "runs"
RUNS_TEMP_DIR = RUNS_DIR / "_runs_tmp"

def load_config() -> dict:
    with open(RUNS_DIR / "config.toml", 'rb') as cfg_file:
        return tomllib.load(cfg_file)

def generate_run_id() -> str:
    return str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def get_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, cwd=CODE_DIR,
    ).stdout.strip()

def create_output_folder() -> ...:
    """
    Output folder will contain:
    - copy of config.toml
    - seed_(n) output data
        - all .npy datas
        - 
    """

def save_run(run_id: str, cfg: dict, net: SRNN, **recorders: Recorder) -> Path:
    RUNS_TEMP_DIR.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{run_id}-", dir=RUNS_TEMP_DIR))

    net.reset()
    net_cpu = copy.deepcopy(net).cpu()
    for m in net_cpu.modules():
        if hasattr(m, "spike_grad"):
            m.spike_grad = None

    state = {
        "cfg": cfg,
        "net": net_cpu,
        **{name: rec.to_state() for name, rec in recorders.items() if rec is not None}
    }
    buf = io.BytesIO()
    torch.save(state, buf)
    (run_dir / "run.pt.gz").write_bytes(gzip.compress(buf.getvalue()))
    return run_dir

def push_run(run_dir: Path, run_id: str, cfg: dict) -> None:
    subprocess.run(
        ["gh", "release", "create", f"run-{run_id}", str(run_dir / "run.pt.gz"),
         "-R", REPO, "--title", run_id, "--notes", yaml.safe_dump(cfg),
         "--target", cfg["git_commit"]],
        check=True,
    )
    # shutil.rmtree(run_dir)

def load_run(path: Path) -> tuple[SRNN, dict, dict]:
    with open(path, "rb") as f:
        state = torch.load(io.BytesIO(gzip.decompress(f.read())), weights_only=True, map_location="cpu")
    # recs = {name: sd for name, sd in state.items() if name not in ["cfg", "net"]}
    cfg = state["cfg"]
    net = state["net"]          # whatever args your constructor takes

    # recs = {name: cls.from_state(state[name]) for name, cls in classes.items() if state[name] is not None}
    return net, cfg, state