import torch
import subprocess, tempfile, shutil, yaml, io, gzip, uuid, time
from pathlib import Path

from .recording import Recorder
from .srnn import SRNN

REPO = "Emiel-Nagel/Thesis-Code"

def generate_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

def push_run(run_dir: Path, run_id: str, cfg: dict):
    subprocess.run(
        ["gh", "release", "create", f"run-{run_id}", str(run_dir / "run.pt.gz"),
         "-R", REPO, "--title", run_id, "--notes", yaml.safe_dump(cfg)],
        check=True,
    )
    shutil.rmtree(run_dir)

def save_run(run_id: str, cfg: dict, net: SRNN, **recorders: Recorder) -> Path:
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{run_id}-"))
    state = {
        "cfg": cfg,
        "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        **{name: rec.to_state() for name, rec in recorders.items()}
    }
    buf = io.BytesIO()
    torch.save(state, buf)
    (run_dir / "run.pt.gz").write_bytes(gzip.compress(buf.getvalue()))
    return run_dir

def load_run(path, classes: dict):
    with open(path, "rb") as f:
        state = torch.load(io.BytesIO(gzip.decompress(f.read())), weights_only=True, map_location="cpu")
    recs = {name: cls.from_state(state[name]) for name, cls in classes.items()}
    return state["cfg"], state["net"], recs