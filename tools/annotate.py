import argparse
import base64
import json
import sys
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sudoku_scan as ss

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LOUPE_RADIUS = 40
LOUPE_ZOOM = 3


class Store:
    def __init__(self, images_dir, annotations_dir):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.cache = OrderedDict()
        self.prefill = {}
        self.lock = threading.Lock()

    def check(self, name):
        if name not in self.names:
            raise KeyError(name)
        return name

    def image(self, name):
        with self.lock:
            if name in self.cache:
                self.cache.move_to_end(name)
                return self.cache[name]
            img = cv2.imread(str(self.images_dir / self.check(name)), cv2.IMREAD_COLOR)
            self.cache[name] = img
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            return img

    def annotation_path(self, name):
        return self.annotations_dir / f"{Path(name).stem}.json"

    def info(self, name):
        img = self.image(name)
        h, w = img.shape[:2]
        path = self.annotation_path(name)
        if path.exists():
            ann = json.loads(path.read_text())
            return {"width": w, "height": h, "source": "annotation", "corners": ann["corners"],
                    "grid": ann["grid"], "conditions": ann.get("conditions", [])}
        if name not in self.prefill:
            debug = {}
            try:
                result = ss.run_pipeline(img, debug)
                corners, grid = result["corners"], result["grid"]
            except Exception:
                fields = debug.get("fields", {})
                corners, grid = fields.get("corners"), fields.get("grid")
            self.prefill[name] = (corners, grid)
        corners, grid = self.prefill[name]
        return {"width": w, "height": h, "source": "pipeline" if corners else "none",
                "corners": corners, "grid": grid or [[0] * 9 for _ in range(9)], "conditions": []}

    def display(self, name, width):
        img = self.image(name)
        scale = width / img.shape[1]
        small = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def loupe(self, name, x, y):
        img = self.image(name)
        r = LOUPE_RADIUS
        x0, y0 = int(round(x)) - r, int(round(y)) - r
        padded = cv2.copyMakeBorder(img, r + 1, r + 1, r + 1, r + 1, cv2.BORDER_CONSTANT, value=(90, 90, 90))
        crop = padded[y0 + r + 1:y0 + 3 * r + 2, x0 + r + 1:x0 + 3 * r + 2]
        big = cv2.resize(crop, None, fx=LOUPE_ZOOM, fy=LOUPE_ZOOM, interpolation=cv2.INTER_NEAREST)
        png = cv2.imencode(".png", big)[1].tobytes()
        return {"x0": x0, "y0": y0, "zoom": LOUPE_ZOOM, "png": base64.b64encode(png).decode()}

    def rectified(self, name, corners):
        rect = ss.rectify(self.image(name), np.float32(corners))
        h4 = rect["H4_image"]
        grid = ss.read_grid(h4)
        png = cv2.imencode(".png", h4)[1].tobytes()
        return {"png": base64.b64encode(png).decode(), "grid": grid}

    def save(self, name, payload):
        corners = [[round(float(x), 2), round(float(y), 2)] for x, y in payload["corners"]]
        grid = [[int(v) for v in row] for row in payload["grid"]]
        if len(corners) != 4 or len(grid) != 9 or any(len(r) != 9 or any(v < 0 or v > 9 for v in r) for r in grid):
            raise ValueError("need 4 corners and a 9x9 grid of 0-9")
        conditions = sorted(c for c in payload.get("conditions", []) if c in ("C1", "C2", "C3", "C4", "C5"))
        text = "\n".join([
            "{",
            f'  "image": {json.dumps(name)},',
            f'  "conditions": {json.dumps(conditions)},',
            f'  "corners": {json.dumps(corners)},',
            '  "grid": [',
            ",\n".join(f"    {json.dumps(row)}" for row in grid),
            "  ]",
            "}",
        ])
        json.loads(text)
        self.annotation_path(name).write_text(text + "\n")
        return {"saved": str(self.annotation_path(name).name)}

    def listing(self):
        return [{"name": n, "done": self.annotation_path(n).exists()} for n in self.names]


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, ctype="application/json", status=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.send(PAGE, "text/html; charset=utf-8")
                elif url.path == "/api/list":
                    self.send(store.listing())
                elif url.path == "/api/info":
                    self.send(store.info(store.check(q["name"])))
                elif url.path == "/api/image":
                    self.send(store.display(store.check(q["name"]), int(q.get("w", 1000))), "image/jpeg")
                elif url.path == "/api/loupe":
                    self.send(store.loupe(store.check(q["name"]), float(q["x"]), float(q["y"])))
                elif url.path == "/api/rect":
                    corners = np.array([float(v) for v in q["c"].split(",")]).reshape(4, 2)
                    self.send(store.rectified(store.check(q["name"]), corners))
                else:
                    self.send({"error": "not found"}, status=404)
            except KeyError as exc:
                self.send({"error": f"unknown {exc}"}, status=404)
            except Exception as exc:
                self.send({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def do_POST(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path != "/api/save":
                    self.send({"error": "not found"}, status=404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                self.send(store.save(store.check(q["name"]), payload))
            except Exception as exc:
                self.send({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SudokuScan annotator</title>
<style>
:root { --bg:#f6f6f4; --panel:#fff; --ink:#1f1f1d; --muted:#6b6b66; --line:#dcdcd6; --accent:#2a78d6; --tl:#d6382a; --tr:#eb8a00; --br:#1baf7a; --bl:#8a4fd6; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.4 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--ink); display:flex; height:100vh; }
#side { width:190px; border-right:1px solid var(--line); background:var(--panel); overflow:auto; flex:none; }
#side h1 { font-size:13px; margin:12px; color:var(--muted); font-weight:600; }
#side button { display:block; width:100%; text-align:left; border:0; background:none; padding:7px 12px; font:inherit; cursor:pointer; }
#side button.active { background:#e8f0fb; font-weight:600; }
#side .ok { color:#1baf7a; }
#main { flex:1; overflow:auto; padding:14px; display:flex; gap:16px; align-items:flex-start; }
#left { flex:none; }
#photo { position:relative; cursor:crosshair; border:1px solid var(--line); background:#333; }
#photo img { display:block; }
#overlay { position:absolute; inset:0; }
.bar { display:flex; gap:6px; align-items:center; margin:8px 0; flex-wrap:wrap; }
.bar button, .bar label { font:inherit; }
button.c { border:2px solid transparent; border-radius:6px; padding:4px 10px; background:#eee; cursor:pointer; font-weight:600; }
button.c.sel { border-color:var(--ink); }
#loupes { display:grid; grid-template-columns:repeat(2, 243px); gap:8px; }
.loupe { position:relative; width:243px; height:243px; border:2px solid var(--line); cursor:crosshair; background:#555; }
.loupe.sel { border-color:var(--ink); }
.loupe img { width:243px; height:243px; image-rendering:pixelated; display:block; }
.loupe span { position:absolute; left:4px; top:2px; font-weight:700; text-shadow:0 0 3px #fff; }
.loupe svg { position:absolute; inset:0; pointer-events:none; }
#right { flex:none; }
#rectwrap { position:relative; width:450px; height:450px; border:1px solid var(--line); background:#ddd; }
#rectwrap img { position:absolute; inset:0; width:450px; height:450px; opacity:.55; }
#cells { position:absolute; inset:0; display:grid; grid-template-columns:repeat(9, 50px); }
#cells input { width:50px; height:50px; border:1px solid rgba(0,0,0,.12); background:transparent; text-align:center; font:700 20px system-ui; color:#c0262d; outline:none; padding:0 0 0 26px; }
#cells input:focus { background:rgba(42,120,214,.15); }
#cells input.b-r { border-right:2px solid #333; } #cells input.b-b { border-bottom:2px solid #333; }
#status { color:var(--muted); min-height:20px; }
.primary { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:7px 16px; font:600 14px system-ui; cursor:pointer; }
.ghost { background:#fff; border:1px solid var(--line); border-radius:6px; padding:6px 12px; font:inherit; cursor:pointer; }
kbd { background:#eee; border-radius:3px; padding:0 4px; font-size:12px; }
.help { color:var(--muted); font-size:12px; max-width:500px; }
</style></head><body>
<div id="side"><h1>IMAGES</h1><div id="list"></div></div>
<div id="main">
  <div id="left">
    <div class="bar" id="cornerbar"></div>
    <div id="photo"><img id="img"><svg id="overlay"></svg></div>
    <div class="help">Click the photo to place the selected corner (outer corner of the thick outer border), then click inside a zoom box to refine it to the exact pixel. <kbd>1</kbd>-<kbd>4</kbd> select corner, arrow keys nudge 0.5 px.</div>
  </div>
  <div id="right">
    <div id="loupes"></div>
    <div class="bar"><button class="ghost" id="reread">Rectify &amp; re-read from these corners</button></div>
    <div id="rectwrap"><img id="rect"><div id="cells"></div></div>
    <div class="help">Small grey digits = photo. Red = your annotation (empty = 0). Fix any wrong cell.</div>
    <div class="bar" id="conds"></div>
    <div class="bar"><button class="primary" id="save">Save &amp; next</button><span id="status"></span></div>
  </div>
</div>
<script>
const NAMES = ["TL", "TR", "BR", "BL"], COLORS = ["#d6382a", "#eb8a00", "#1baf7a", "#8a4fd6"];
let DISPLAY_W = 760;
let list = [], cur = null, info = null, corners = [null, null, null, null], sel = 0, scale = 1;
function fitWidth(w, h) {
  const byWidth = window.innerWidth - 190 - 500 - 60;
  const byHeight = (window.innerHeight - 110) * w / h;
  return Math.max(360, Math.round(Math.min(byWidth, byHeight, 1100)));
}
const $ = id => document.getElementById(id);
const api = async (url, opts) => { const r = await fetch(url, opts); const j = r.headers.get("content-type").includes("json") ? await r.json() : null; if (!r.ok) throw new Error(j && j.error || r.status); return j; };
const enc = encodeURIComponent;

function buildStatic() {
  $("cornerbar").innerHTML = NAMES.map((n, i) => `<button class="c" data-i="${i}" style="color:${COLORS[i]}">${i + 1} ${n}</button>`).join("");
  $("cornerbar").onclick = e => { const b = e.target.closest("button"); if (b) selectCorner(+b.dataset.i); };
  $("loupes").innerHTML = NAMES.map((n, i) => `<div class="loupe" data-i="${i}"><img><svg viewBox="0 0 243 243"></svg><span style="color:${COLORS[i]}">${n}</span></div>`).join("");
  document.querySelectorAll(".loupe").forEach(el => el.onclick = e => loupeClick(+el.dataset.i, e));
  $("conds").innerHTML = ["C1", "C2", "C3", "C4", "C5"].map(c => `<label><input type="checkbox" value="${c}"> ${c}</label>`).join(" ") +
    ` <span class="help">C1 frontal · C2 oblique · C3 rotated · C4 shadow/low light · C5 clutter</span>`;
  let cells = "";
  for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++)
    cells += `<input maxlength="1" data-r="${r}" data-c="${c}" class="${c % 3 == 2 && c < 8 ? "b-r" : ""} ${r % 3 == 2 && r < 8 ? "b-b" : ""}">`;
  $("cells").innerHTML = cells;
  $("cells").addEventListener("input", e => { e.target.value = e.target.value.replace(/[^1-9]/g, ""); if (e.target.value) moveCell(e.target, 1); });
  $("cells").addEventListener("keydown", e => {
    const map = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 9, ArrowUp: -9 };
    if (map[e.key]) { e.preventDefault(); moveCell(e.target, map[e.key]); }
    if (e.key === "Backspace" && !e.target.value) moveCell(e.target, -1);
    if (e.key === "0" || e.key === " ") { e.preventDefault(); e.target.value = ""; moveCell(e.target, 1); }
  });
  $("photo").onclick = photoClick;
  $("reread").onclick = reread;
  $("save").onclick = save;
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if ("1234".includes(e.key)) selectCorner(+e.key - 1);
    const d = { ArrowLeft: [-.5, 0], ArrowRight: [.5, 0], ArrowUp: [0, -.5], ArrowDown: [0, .5] }[e.key];
    if (d && corners[sel]) { e.preventDefault(); corners[sel] = [corners[sel][0] + d[0], corners[sel][1] + d[1]]; drawAll(); }
    if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); save(); }
  });
}

function moveCell(input, step) {
  const all = [...document.querySelectorAll("#cells input")];
  const i = all.indexOf(input) + step;
  if (i >= 0 && i < 81) all[i].focus(), all[i].select();
}

async function loadList() {
  list = await api("/api/list");
  $("list").innerHTML = list.map(it => `<button data-n="${enc(it.name)}" class="${it.name === cur ? "active" : ""}">${it.done ? '<span class="ok">✓</span> ' : "· "}${it.name}</button>`).join("");
  $("list").querySelectorAll("button").forEach(b => b.onclick = () => openImage(decodeURIComponent(b.dataset.n)));
}

async function openImage(name) {
  cur = name; await loadList();
  $("status").textContent = "Loading (first open runs the pipeline, a few seconds)…";
  info = await api(`/api/info?name=${enc(name)}`);
  DISPLAY_W = fitWidth(info.width, info.height);
  scale = DISPLAY_W / info.width;
  $("img").src = `/api/image?name=${enc(name)}&w=${DISPLAY_W}`;
  $("img").style.width = DISPLAY_W + "px";
  $("overlay").setAttribute("viewBox", `0 0 ${DISPLAY_W} ${Math.round(info.height * scale)}`);
  corners = info.corners ? info.corners.map(p => [p[0], p[1]]) : [null, null, null, null];
  setGrid(info.grid);
  document.querySelectorAll("#conds input").forEach(cb => cb.checked = info.conditions.includes(cb.value));
  selectCorner(corners.findIndex(c => !c) >= 0 ? corners.findIndex(c => !c) : 0);
  $("rect").removeAttribute("src");
  drawAll();
  $("status").textContent = { annotation: "Loaded your saved annotation.", pipeline: "Pre-filled from the pipeline: check every corner and cell.", none: "Pipeline found nothing: place the 4 corners." }[info.source];
  if (corners.every(c => c)) reread(false);
}

function selectCorner(i) {
  sel = i;
  document.querySelectorAll("button.c").forEach((b, k) => b.classList.toggle("sel", k === i));
  document.querySelectorAll(".loupe").forEach((b, k) => b.classList.toggle("sel", k === i));
}

function photoClick(e) {
  const r = $("img").getBoundingClientRect();
  const x = (e.clientX - r.left + 0.5) / scale - 0.5, y = (e.clientY - r.top + 0.5) / scale - 0.5;
  corners[sel] = [x, y];
  drawAll();
  if (sel < 3 && corners.slice(sel + 1).some(c => !c)) selectCorner(corners.findIndex((c, k) => k > sel && !c));
}

async function loupeClick(i, e) {
  const el = document.querySelector(`.loupe[data-i="${i}"]`);
  if (!el.dataset.x0) return;
  const r = el.getBoundingClientRect(), z = +el.dataset.zoom;
  const lx = (e.clientX - r.left) * 243 / r.width, ly = (e.clientY - r.top) * 243 / r.height;
  corners[i] = [+el.dataset.x0 + (lx - z / 2) / z, +el.dataset.y0 + (ly - z / 2) / z];
  selectCorner(i);
  drawAll();
}

function drawOverlay() {
  const pts = corners.map(c => c && [(c[0] + 0.5) * scale - 0.5, (c[1] + 0.5) * scale - 0.5]);
  let s = "";
  if (pts.every(p => p)) s += `<polygon points="${pts.map(p => p.join(",")).join(" ")}" fill="none" stroke="#2a78d6" stroke-width="1.5"/>`;
  pts.forEach((p, i) => { if (p) s += `<circle cx="${p[0]}" cy="${p[1]}" r="7" fill="none" stroke="${COLORS[i]}" stroke-width="2.5"/><text x="${p[0] + 9}" y="${p[1] - 8}" fill="#fff" stroke="#000" stroke-width="3" paint-order="stroke" font-weight="700" font-size="13">${NAMES[i]}</text>`; });
  $("overlay").innerHTML = s;
}

let loupeToken = 0;
async function drawLoupes() {
  const token = ++loupeToken;
  await Promise.all(corners.map(async (c, i) => {
    const el = document.querySelector(`.loupe[data-i="${i}"]`);
    if (!c) { el.querySelector("img").removeAttribute("src"); el.querySelector("svg").innerHTML = ""; delete el.dataset.x0; return; }
    const L = await api(`/api/loupe?name=${enc(cur)}&x=${c[0]}&y=${c[1]}`);
    if (token !== loupeToken) return;
    el.dataset.x0 = L.x0; el.dataset.y0 = L.y0; el.dataset.zoom = L.zoom;
    el.querySelector("img").src = "data:image/png;base64," + L.png;
    const cx = (c[0] - L.x0) * L.zoom + L.zoom / 2, cy = (c[1] - L.y0) * L.zoom + L.zoom / 2;
    el.querySelector("svg").innerHTML = `<line x1="${cx}" y1="0" x2="${cx}" y2="243" stroke="#00e0ff" stroke-width="1"/><line x1="0" y1="${cy}" x2="243" y2="${cy}" stroke="#00e0ff" stroke-width="1"/><circle cx="${cx}" cy="${cy}" r="5" fill="none" stroke="#ff2d55" stroke-width="1.5"/>`;
  }));
}

function drawAll() { drawOverlay(); drawLoupes(); }

function setGrid(g) {
  document.querySelectorAll("#cells input").forEach(inp => { const v = g[+inp.dataset.r][+inp.dataset.c]; inp.value = v ? String(v) : ""; });
}
function getGrid() {
  const g = Array.from({ length: 9 }, () => Array(9).fill(0));
  document.querySelectorAll("#cells input").forEach(inp => g[+inp.dataset.r][+inp.dataset.c] = inp.value ? +inp.value : 0);
  return g;
}

async function reread(replaceGrid = true) {
  if (!corners.every(c => c)) { $("status").textContent = "Place all 4 corners first."; return; }
  $("status").textContent = "Rectifying…";
  const R = await api(`/api/rect?name=${enc(cur)}&c=${corners.flat().join(",")}`);
  $("rect").src = "data:image/png;base64," + R.png;
  if (replaceGrid) setGrid(R.grid);
  $("status").textContent = replaceGrid ? "Grid re-read from these corners: check every cell." : "Check corners (zoom boxes) and every cell.";
}

async function save() {
  if (!corners.every(c => c)) { $("status").textContent = "Place all 4 corners first."; return; }
  const conditions = [...document.querySelectorAll("#conds input:checked")].map(cb => cb.value);
  if (!conditions.length && !confirm("No condition (C1–C5) ticked. Save anyway?")) return;
  await api(`/api/save?name=${enc(cur)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corners, grid: getGrid(), conditions }) });
  await loadList();
  const next = list.find(it => !it.done);
  $("status").textContent = "Saved.";
  if (next) openImage(next.name);
}

buildStatic();
loadList().then(() => openImage((list.find(it => !it.done) || list[0]).name));
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="Annotate photos in the Appendix A format (corners + grid + conditions).")
    parser.add_argument("--images", type=Path, required=True, help="directory with the photos")
    parser.add_argument("--annotations", type=Path, required=True, help="directory to write <name>.json files into")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    store = Store(args.images, args.annotations)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Annotating {len(store.names)} images -> {args.annotations}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
