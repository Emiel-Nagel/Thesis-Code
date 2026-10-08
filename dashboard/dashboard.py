import hmac, json, os, queue, threading
from collections import deque
from flask import Flask, Response, request

USER = os.environ["DASH_USER"]
PASSWORD = os.environ["DASH_PASS"]
TOKEN = os.environ["DASH_TOKEN"]
ALLOWED_FRAME = "https://USERNAME.github.io"     # your GitHub Pages origin

app = Flask(__name__)
history: dict[int, deque] = {}                   # seed -> lines
run_name = ""
listeners: list[queue.Queue] = []
lock = threading.Lock()


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
    e = request.get_json(force=True)             # {"seed": 0, "lines": ["...", ...]}
    seed = int(e["seed"])
    lines = [str(t) for t in e.get("lines", [])]
    with lock:
        dq = history.setdefault(seed, deque(maxlen=20000))
        for t in lines:
            dq.append(t)
            broadcast({"seed": seed, "text": t})
    return "", 204


@app.post("/reset")
def reset():
    global run_name
    run_name = str((request.get_json(silent=True) or {}).get("run", ""))
    with lock:
        history.clear()
        broadcast({"type": "reset", "run": run_name})
    return "", 204


@app.get("/stream")
def stream():
    q: queue.Queue = queue.Queue()
    with lock:
        backlog = [{"type": "reset", "run": run_name}]
        backlog += [{"seed": s, "text": t} for s, lines in history.items() for t in lines]
        listeners.append(q)

    def gen():
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

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return PAGE


PAGE = """
<body style="margin:0;font-family:monospace;background:#111;color:#ddd">
<div style="padding:6px"><b id="run">no run</b> &nbsp;
  seed: <select id="sel"></select></div>
<div id="box" style="height:calc(100vh - 40px);overflow:auto">
  <pre id="out" style="margin:0;padding:6px;white-space:pre-wrap"></pre>
</div>
<script>
const sel = document.getElementById("sel"), out = document.getElementById("out"),
      box = document.getElementById("box"), runEl = document.getElementById("run");
let buf = {}, current = null;

function render() {
  out.textContent = current === null ? "" : buf[current].join("\\n") + "\\n";
  box.scrollTop = box.scrollHeight;
}
sel.onchange = () => { current = sel.value; render(); };

new EventSource("/stream").onmessage = ev => {
  const e = JSON.parse(ev.data);
  if (e.type === "reset") {
    buf = {}; current = null; sel.innerHTML = ""; out.textContent = "";
    runEl.textContent = e.run || "no run";
    return;
  }
  if (!(e.seed in buf)) {
    buf[e.seed] = [];
    const o = document.createElement("option");
    o.value = o.textContent = e.seed;
    sel.appendChild(o);
    if (current === null) { current = String(e.seed); sel.value = current; }
  }
  buf[e.seed].push(e.text);
  if (String(e.seed) === current) {
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 50;
    out.textContent += e.text + "\\n";
    if (atBottom) box.scrollTop = box.scrollHeight;
  }
};
</script></body>"""

if __name__ == "__main__":                       # local testing only; gunicorn ignores this
    app.run(host="127.0.0.1", port=8000, threaded=True)