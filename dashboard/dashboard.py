import hmac, json, os, queue, threading
from collections import deque, defaultdict
from pathlib import Path
from flask import Flask, Response, request, send_file
from dotenv import load_dotenv
load_dotenv()   # for local dev

USER = os.environ["DASH_USER"]
PASSWORD = os.environ["DASH_PASS"]
TOKEN = os.environ["DASH_TOKEN"]
ALLOWED_FRAME = "https://emiel-nagel.github.io"     # your GitHub Pages origin

app = Flask(__name__)
listeners: list[queue.Queue] = []
lock = threading.Lock()
curdir = Path(__file__).resolve().parent

class History:
    def __init__(self) -> None:
        self.contents: dict[str, deque] = defaultdict(lambda: deque(maxlen=20000))        # run_name / seed -> lines

    def clear(self) -> None:
        self.contents.clear()

    def put(self, run_name: str, seed: int, text: str) -> None:
        key = f"{run_name}/{seed}"
        self.contents[key].append(text)

    def get(self, run_name: str, seed: int) -> deque | None:
        key = f"{run_name}/{seed}"
        return self.contents.get(key)

    def __iter__(self):
        for key, lines in self.contents.items():
            run_name, seed = key.split("/")
            yield run_name, seed, lines

history = History()

def command_reset() -> dict:
    return {"type": "reset"}

def command_push_text(run_name: str, seed: int, text: str) -> dict:
    return {"run_name": run_name, "seed": seed, "text": text}

def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())

def broadcast(event: dict) -> None:              # call with lock held
    for q in listeners:
        q.put(event)

@app.before_request
def auth():
    if request.path in ("/log", "/reset"):       # machine-to-machine: token
        if not same(request.headers.get("X-Token", ""), TOKEN):
            return "", 403
        return None
    a = request.authorization                    # browser: basic auth
    if not (a and same(a.username or "", USER) and same(a.password or "", PASSWORD)):
        return Response("Login required", 401,
                        {"WWW-Authenticate": 'Basic realm="dashboard"'})
    return None

@app.after_request
def frame_policy(resp):
    resp.headers["Content-Security-Policy"] = f"frame-ancestors {ALLOWED_FRAME}"
    return resp

@app.post("/log")
def log():
    e = request.get_json(force=True, silent=True)
    print(e)
    if not isinstance(e, dict):
        return "bad json", 400
    try:
        run_name = e["run_name"]
        seed = int(e["seed"])
    except (KeyError, ValueError, TypeError):
        return "run_name and integer seed required", 400
    lines = [str(t) for t in e.get("lines", [])]
    with lock:
        for t in lines:
            history.put(run_name, seed, t)
            broadcast(command_push_text(run_name, seed, t))
    return "", 204

@app.post("/reset")
def reset():
    with lock:
        history.clear()
        broadcast(command_reset())
    return "", 204

@app.get("/stream")
def stream():
    q: queue.Queue = queue.Queue()
    with lock:
        backlog = [command_reset()]
        backlog += [command_push_text(r, s, t) for r, s, lines in list(history) for t in lines]
        print(backlog)
        listeners.append(q)

    def generate():
        try:
            for e in backlog:
                yield f"data: {json.dumps(e)}\n\n"
            while True:
                try:
                    yield f"data: {json.dumps(q.get(timeout=15))}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with lock:
                listeners.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/")
def index():
    return send_file(curdir / "index.html")

if __name__ == "__main__":                       # local testing only; gunicorn ignores this
    app.run(host="127.0.0.1", port=8000, threaded=True)