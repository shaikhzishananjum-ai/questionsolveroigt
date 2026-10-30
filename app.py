#!/usr/bin/env python3
"""
=====================================================================
 ESP32-CAM Question Solver  -  Render web service
=====================================================================
 Flow

   ESP32-CAM  --POST /ask (raw JPEG)-->  this service  --> Gemini 2.5 Flash
                                              |
                                              +--> Telegram (photo + answer)
   ESP32-CAM  --GET /poll?since=N -->  "pending" / "done" + answer text

   Your phone --> Telegram bot --> this service (webhook) --> same pipeline
   so anything you photograph with your phone also appears on the TFT.

 Environment variables (set these in the Render dashboard)
   GEMINI_API_KEY   Google AI Studio key            (required)
   DEVICE_KEY       shared secret with the firmware (required)
   TG_TOKEN         Telegram bot token              (optional)
   TG_CHAT_ID       your Telegram numeric chat id   (optional)
   WEBHOOK_SECRET   secret path for the webhook     (auto-generated if unset)
   PUBLIC_URL       https://your-app.onrender.com   (needed to set the webhook)
   GEMINI_MODEL     default gemini-2.5-flash
=====================================================================
"""

import os
import re
import time
import base64
import logging
import secrets
import threading
import unicodedata

import requests
from flask import Flask, request, Response, abort

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qs")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------
TG_TOKEN       = os.environ.get("TG_TOKEN", "").strip()
TG_CHAT        = os.environ.get("TG_CHAT_ID", "").strip()
GEMINI_KEY     = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL          = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
DEVICE_KEY     = os.environ.get("DEVICE_KEY", "").strip()
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip() or secrets.token_hex(12)
PUBLIC_URL     = os.environ.get("PUBLIC_URL", "").rstrip("/")
MAX_CHARS      = int(os.environ.get("MAX_ANSWER_CHARS", "1900"))

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              f"{MODEL}:generateContent")
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

app = Flask(__name__)

# ---------------------------------------------------------------------
# tiny in-memory job store (single gunicorn worker -> safe)
# ---------------------------------------------------------------------
LOCK   = threading.Lock()
JOBS   = {}          # id -> {"status": pending|done|error, "text": str, "ts": float}
SEQ    = 0
KEEP   = 40


def new_job() -> int:
    global SEQ
    with LOCK:
        SEQ += 1
        JOBS[SEQ] = {"status": "pending", "text": "", "ts": time.time()}
        if len(JOBS) > KEEP:
            for k in sorted(JOBS)[:-KEEP]:
                JOBS.pop(k, None)
        return SEQ


def set_job(job: int, status: str, text: str) -> None:
    with LOCK:
        if job in JOBS:
            JOBS[job].update(status=status, text=text, ts=time.time())


# ---------------------------------------------------------------------
# the prompt - this is where answer quality actually comes from
# ---------------------------------------------------------------------
PROMPT = """You solve exam questions from photographs.

RULES
1. Read every question visible in the photo.
2. Answer at most the first TWO questions.
3. Output PLAIN TEXT only - no markdown, no asterisks, no #, no backticks, no LaTeX.
4. Write maths in plain ASCII: x^2, sqrt(3), 1/2, pi, theta, 45 deg, ->, <=, >=, +-, |x|, integral, sum.
5. Use exactly these labels:
Q: <the question, at most 90 characters>
SOL: <at most 4 short lines of working>
ANS: <the final answer on one line>
6. Total output under 1500 characters and under 55 lines.
7. If part of the photo is blurry or cut off, still answer your best reading and put (?) after the unclear part.
8. Never refuse, never describe the image, never mention that it is a photo.
"""

TEXT_PROMPT = """You solve exam questions.

RULES
1. Answer the question below.
2. Output PLAIN TEXT only - no markdown, no LaTeX.
3. Use exactly these labels:  Q: ...   SOL: ...   ANS: ...
4. Write maths in plain ASCII (x^2, sqrt(3), 1/2, pi, ->, <=).
5. Keep the whole answer under 1500 characters.

QUESTION:
"""

# ---------------------------------------------------------------------
# AI
# ---------------------------------------------------------------------
def gemini_call(parts, timeout=90) -> str:
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set on Render")

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.15,
            "topP": 0.95,
            "maxOutputTokens": 2048,
            # Gemini 2.5-Flash "thinks" by default which doubles latency.
            # For OCR + school maths, 0 is both faster and just as good.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    headers = {"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}

    r = None
    for attempt in range(3):
        try:
            r = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            log.warning("gemini transport error (attempt %d): %s", attempt + 1, e)
            if attempt == 2:
                raise RuntimeError("cannot reach the AI service")
            time.sleep(2)
            continue

        # some models reject thinkingBudget -> retry once without it
        if r.status_code == 400 and "thinkingConfig" in payload["generationConfig"]:
            log.warning("model rejects thinkingConfig, retrying without it")
            payload["generationConfig"].pop("thinkingConfig")
            continue

        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            log.warning("gemini %s, retrying", r.status_code)
            time.sleep(2 + attempt * 3)
            if r.status_code == 429:
                continue
            continue
        break

    if r is None:
        raise RuntimeError("no response from Gemini")
    if r.status_code == 429:
        raise RuntimeError("AI is rate limited, try again in a minute")
    if r.status_code != 200:
        log.error("gemini %s: %s", r.status_code, r.text[:400])
        raise RuntimeError(f"AI error {r.status_code}")

    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        fb = (data.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"AI blocked the request ({fb or 'no candidates'})")

    parts_out = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts_out).strip()
    if not text:
        raise RuntimeError("AI returned an empty answer")
    if cands[0].get("finishReason") == "MAX_TOKENS":
        text += "\n(truncated)"
    return text


def solve_image(image_bytes: bytes) -> str:
    parts = [
        {"text": PROMPT},
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(image_bytes).decode()}},
    ]
    return gemini_call(parts)


def solve_text(question: str) -> str:
    return gemini_call([{"text": TEXT_PROMPT + question}])


# ---------------------------------------------------------------------
# make the answer fit a 22-column, 8-pixel-tall screen
# ---------------------------------------------------------------------
ASCII_MAP = {
    "×": "x", "÷": "/", "−": "-", "–": "-", "—": "-", "‑": "-",
    "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
    "²": "^2", "³": "^3", "⁴": "^4", "½": "1/2", "¼": "1/4", "¾": "3/4",
    "√": "sqrt", "π": "pi", "θ": "theta", "φ": "phi", "ω": "omega",
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "λ": "lambda",
    "μ": "u", "Ω": "ohm", "∞": "inf", "≈": "~=", "≠": "!=",
    "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "⇒": "=>", "±": "+-",
    "°": " deg", "∫": "integral ", "∑": "sum ", "∈": " in ", "∴": "so ",
    "•": "-", "·": ".", "\u00a0": " ", "\t": "  ",
}

LATEX_SUBS = [
    (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)"),
    (r"\\times", "x"),
    (r"\\cdot", "*"),
    (r"\\pi", "pi"),
    (r"\\theta", "theta"),
    (r"\\alpha", "alpha"),
    (r"\\beta", "beta"),
    (r"\\rightarrow", "->"),
    (r"\\le", "<="),
    (r"\\ge", ">="),
    (r"\\neq", "!="),
    (r"\\pm", "+-"),
    (r"\\infty", "inf"),
    (r"\\hat\{([^{}]*)\}", r"\1"),
    (r"\\vec\{([^{}]*)\}", r"\1"),
    (r"\\bar\{([^{}]*)\}", r"\1"),
    (r"\\left|\\right", "|"),
    (r"\^\{([^{}]*)\}", r"^\1"),
    (r"_\{([^{}]*)\}", r"_\1"),
]


def sanitize(text: str) -> str:
    """Turn whatever the model said into something an ST7789 can draw."""
    if not text:
        return "No answer."

    # strip code fences / stray markdown
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    for pat, rep in LATEX_SUBS:
        text = re.sub(pat, rep, text)

    # anything LaTeX-ish we did not name explicitly: drop the command,
    # keep its argument   \mathbf{v} -> v ,  \hat{i} -> i
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\\\([()\[\]])", r"\1", text)
    text = text.replace("\\", "")
    # markdown horizontal rules
    text = re.sub(r"(?m)^\s*[-=_]{3,}\s*$", "", text)

    text = text.replace("$", "").replace("**", "").replace("`", "")
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)   # headings
    text = re.sub(r"(?m)^\s*[-*]\s+", "- ", text)               # bullets

    out = []
    for ch in text:
        if ch in ASCII_MAP:
            out.append(ASCII_MAP[ch])
        elif ch == "\n":
            out.append("\n")
        elif ord(ch) < 128:
            out.append(ch)
        else:
            folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
            out.append(folded if folded else "")
    text = "".join(out)

    text = text.replace("\r", "")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()

    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        nl = cut.rfind("\n")
        text = cut[:nl] if nl > MAX_CHARS * 0.6 else cut
        text += "\n(full answer on Telegram)"
    return text or "No answer."


def wrap22(text: str):
    """Same word-wrap rules as the firmware, used by the browser preview."""
    lines = []
    for para in text.split("\n"):
        para = para.rstrip()
        if not para:
            lines.append("")
            continue
        p = 0
        while p < len(para):
            if len(para) - p <= 22:
                lines.append(para[p:])
                break
            take = 22
            sp = para.rfind(" ", p + 5, p + 23)
            if sp != -1:
                take = sp - p
            lines.append(para[p:p + take].rstrip())
            p += take
            while p < len(para) and para[p] == " ":
                p += 1
    return lines


# ---------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------
def tg(method: str, files=None, **params):
    if not TG_TOKEN:
        return None
    try:
        r = requests.post(f"{TG_API}/{method}", data=params, files=files, timeout=45)
        if r.status_code != 200:
            log.warning("telegram %s -> %s %s", method, r.status_code, r.text[:200])
        return r.json()
    except Exception as e:
        log.warning("telegram %s failed: %s", method, e)
        return None


def tg_send_text(chat_id, text):
    if not chat_id:
        return
    for i in range(0, min(len(text), 4000), 3900):
        tg("sendMessage", chat_id=chat_id, text=text[i:i + 3900])


def tg_send_photo(chat_id, jpeg: bytes, caption="solving..."):
    if not chat_id:
        return
    tg("sendPhoto", files={"photo": ("question.jpg", jpeg, "image/jpeg")},
       chat_id=chat_id, caption=caption)


def tg_download(file_id: str) -> bytes:
    info = tg("getFile", file_id=file_id) or {}
    path = (info.get("result") or {}).get("file_path")
    if not path:
        raise RuntimeError("telegram getFile failed")
    r = requests.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{path}", timeout=60)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------
# the worker that actually does the work
# ---------------------------------------------------------------------
def run_job(job: int, image_bytes=None, chat_id=None, question=None):
    try:
        if image_bytes and chat_id:
            tg_send_photo(chat_id, image_bytes, "solving...")

        if image_bytes:
            raw = solve_image(image_bytes)
        else:
            raw = solve_text(question or "")

        answer = sanitize(raw)
        set_job(job, "done", answer)
        log.info("job %s done (%d chars)", job, len(answer))
        tg_send_text(chat_id, answer)

    except Exception as e:
        log.exception("job %s failed", job)
        msg = "AI ERROR\n" + str(e)[:180]
        set_job(job, "error", msg)
        tg_send_text(chat_id, msg)


def spawn(image_bytes=None, chat_id=None, question=None) -> int:
    job = new_job()
    threading.Thread(target=run_job,
                     args=(job, image_bytes, chat_id, question),
                     daemon=True).start()
    return job


# ---------------------------------------------------------------------
# auth helper
# ---------------------------------------------------------------------
def auth_ok() -> bool:
    if not DEVICE_KEY:
        return True                      # no key configured -> open (dev only)
    key = request.headers.get("X-Device-Key") or request.args.get("key", "")
    return key == DEVICE_KEY


# ---------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return Response("ok", content_type="text/plain")


@app.post("/ask")
def ask():
    """ESP32-CAM uploads a raw JPEG here. Returns the new job id as text."""
    if not auth_ok():
        abort(401)

    if request.files.get("photo"):
        img = request.files["photo"].read()
    else:
        img = request.get_data(cache=False)

    if not img or len(img) < 500:
        return Response("image too small", status=400, content_type="text/plain")
    if len(img) > 4 * 1024 * 1024:
        return Response("image too large", status=413, content_type="text/plain")

    log.info("ask: %d bytes", len(img))
    job = spawn(img, TG_CHAT or None)
    return Response(str(job), content_type="text/plain")


@app.get("/poll")
def poll():
    """
    GET /poll?since=N      -> body = answer, header X-Status = pending|done|error
                              header X-Job = the job id the body belongs to
    GET /poll?peek=1       -> body = newest job id (used once at boot)
    """
    if not auth_ok():
        abort(401)

    with LOCK:
        newest = SEQ

    if request.args.get("peek"):
        return Response(str(newest),
                        headers={"X-Status": "peek", "X-Job": str(newest)},
                        content_type="text/plain")

    try:
        since = int(request.args.get("since", "0"))
    except ValueError:
        since = 0

    job = newest if newest > since else since
    with LOCK:
        rec = dict(JOBS.get(job) or {})

    if not rec:
        return Response("", headers={"X-Status": "none", "X-Job": str(job)},
                        content_type="text/plain")

    body = rec["text"] if rec["status"] in ("done", "error") else ""
    return Response(body,
                    headers={"X-Status": rec["status"], "X-Job": str(job)},
                    content_type="text/plain; charset=utf-8")


@app.post("/tg/<secret>")
def telegram_webhook(secret):
    """Phone -> bot -> here. Same pipeline, answer lands on the TFT too."""
    if secret != WEBHOOK_SECRET:
        abort(403)

    upd = request.get_json(force=True, silent=True) or {}
    msg = upd.get("message") or upd.get("edited_message") or {}
    chat_id = str((msg.get("chat") or {}).get("id", ""))
    if not chat_id:
        return "ok"

    # remember the chat so ESP32-initiated jobs also reply to you
    global TG_CHAT
    if TG_CHAT and chat_id != TG_CHAT:
        return "ok"                      # ignore strangers

    try:
        if msg.get("photo"):
            file_id = msg["photo"][-1]["file_id"]
            img = tg_download(file_id)
            spawn(img, chat_id)
            tg_send_text(chat_id, "got it, solving...")
        elif msg.get("text"):
            spawn(None, chat_id, msg["text"])
    except Exception as e:
        log.exception("webhook")
        tg_send_text(chat_id, "failed: " + str(e)[:150])
    return "ok"


@app.post("/ui/solve")
def ui_solve():
    """Browser debug page: upload a photo, see the answer and the TFT layout."""
    if not auth_ok():
        abort(401)
    f = request.files.get("photo")
    if not f:
        return {"error": "no file"}, 400
    img = f.read()
    try:
        raw = solve_image(img)
        answer = sanitize(raw)
        return {"ok": True, "text": answer, "lines": wrap22(answer)}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 200


@app.get("/setwebhook")
def set_webhook():
    """Open this once in a browser after the first deploy."""
    if not (PUBLIC_URL and TG_TOKEN):
        return Response("PUBLIC_URL or TG_TOKEN missing", status=400)
    url = f"{PUBLIC_URL}/tg/{WEBHOOK_SECRET}"
    r = tg("setWebhook", url=url, drop_pending_updates=True)
    return Response(f"webhook -> {url}\n{r}", content_type="text/plain")


@app.get("/")
def index():
    return Response(INDEX_HTML, content_type="text/html")


# ---------------------------------------------------------------------
# register the webhook on boot (best effort, harmless if it fails)
# ---------------------------------------------------------------------
def _boot_webhook():
    time.sleep(3)
    if PUBLIC_URL and TG_TOKEN:
        try:
            tg("setWebhook", url=f"{PUBLIC_URL}/tg/{WEBHOOK_SECRET}",
               drop_pending_updates=True)
            log.info("telegram webhook registered")
        except Exception:
            pass


if PUBLIC_URL and TG_TOKEN:
    threading.Thread(target=_boot_webhook, daemon=True).start()


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Question Solver - test bench</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{background:#111;color:#ddd;font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;
      display:flex;gap:32px;flex-wrap:wrap;justify-content:center}
 h1{font-size:18px;margin:0 0 4px;color:#0f0}
 .card{background:#1a1a1a;border:1px solid #2c2c2c;border-radius:12px;padding:18px;max-width:420px}
 .screen{width:270px;height:480px;background:#000;border-radius:10px;
         border:6px solid #333;overflow:hidden;position:relative;box-shadow:0 8px 24px #0008}
 .bar{height:18px;color:#0f0;font:12px/18px monospace;padding-left:2px;background:#000;
      border-bottom:1px solid #222}
 .txt{color:#fff;font:16px/16px ui-monospace,monospace;white-space:pre-wrap;padding:2px;
      width:22ch;transition:transform .35s linear}
 button,input[type=file]{font:inherit;margin-top:10px}
 button{background:#0f0;color:#000;border:0;padding:8px 16px;border-radius:8px;font-weight:700;cursor:pointer}
 input[type=text]{background:#000;color:#0f0;border:1px solid #333;padding:6px;border-radius:6px;width:100%}
 a{color:#0f0}
</style></head><body>
<div class="card">
  <h1>Question Solver</h1>
  <p>Test the whole AI pipeline without touching the ESP32.</p>
  <p>Device key:<br><input id="key" type="text" placeholder="DEVICE_KEY"></p>
  <p><input id="f" type="file" accept="image/*" capture="environment"></p>
  <button onclick="go()">Solve it</button>
  <p id="st" style="color:#888"></p>
  <p style="font-size:12px;color:#666">Scroll speed looks the same on the real TFT.</p>
</div>
<div class="card">
  <div class="screen">
    <div class="bar" id="bar">ANSWER  &gt;SCROLL</div>
    <div class="txt" id="txt">READY.<br><br>Pick a photo of a question.</div>
  </div>
</div>
<script>
const keyEl=document.getElementById('key');
keyEl.value=localStorage.getItem('dk')||'';
keyEl.onchange=()=>localStorage.setItem('dk',keyEl.value);
let timer=null, off=0;
function setScreen(lines){
  const t=document.getElementById('txt');
  t.innerHTML='';
  lines.slice(0,400).forEach((l,i)=>{ if(i) t.appendChild(document.createElement('br'));
                                     t.appendChild(document.createTextNode(l||' ')); });
  document.getElementById('bar').textContent=lines.length>28?'ANSWER  &gt;SCROLL':'ANSWER';
  off=0;t.style.transform='translateY(0px)';
  clearInterval(timer);
  if(lines.length>28){
    timer=setInterval(()=>{
      off++;
      if(off>lines.length-28){clearInterval(timer);return;}
      t.style.transform='translateY(-'+(off*16)+'px)';
    },1500);
  }
}
async function go(){
  const f=document.getElementById('f').files[0];
  if(!f){return;}
  const st=document.getElementById('st'); st.textContent='solving... (10-20 s)';
  document.getElementById('bar').textContent='AI THINKING';
  const fd=new FormData(); fd.append('photo',f);
  const r=await fetch('/ui/solve',{method:'POST',headers:{'X-Device-Key':keyEl.value},body:fd});
  const j=await r.json();
  if(j.ok){ st.textContent='done - '+j.lines.length+' lines'; setScreen(j.lines); }
  else { st.textContent='error: '+j.error; setScreen(['ERROR', ''].concat(String(j.error).match(/.{1,22}/g)||[])); }
}
</script></body></html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    log.info("dev server on :%d   model=%s  telegram=%s",
             port, MODEL, "on" if TG_TOKEN else "off")
    app.run(host="0.0.0.0", port=port, threaded=True)
