"""Generative abilities — the ChatGPT/Gemini-style features, done locally.

What ChatGPT and Gemini *are* is a set of large neural models (a language
model, an image model, speech models). Their weights are proprietary and
cannot be copied into a repository. What *can* be reproduced is the
**product surface** around them, wired to open models that run on the
owner's own machine:

===============  ==========================================  ==============
capability       local engine (runs on your machine)         protocol here
===============  ==========================================  ==============
fluent talk      Ollama (llama3 / qwen / mistral / gemma…)   ``/api/chat``
images           Stable Diffusion WebUI started with --api   ``/sdapi/v1/txt2img``
listen (STT)     whisper.cpp ``whisper-server``              ``/inference``
speak (TTS)      Piper (http server *or* the CLI binary)     ``/`` or stdin
presentations    built right here — a .pptx is zipped XML    stdlib zipfile
===============  ==========================================  ==============

Rules kept from the rest of RealAI:

- **stdlib only** (``urllib``, ``zipfile``, ``subprocess``, ``base64``).
- **off by default** — every backend is disabled until the owner sets its
  URL in :class:`~realaiagent.config.Config`. With nothing configured the
  agent keeps its honest "I cannot do that" answers.
- **no external calls** — the URLs are meant to be localhost / LAN. The
  agent never ships a key anywhere because there is no key.
- **everything counted** — each generation lands in the usage ledger.

Every public method returns a :class:`GenResult` and never raises.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.sax.saxutils import escape as _xml

from .config import Config

Messages = List[Dict[str, str]]


@dataclass
class GenResult:
    ok: bool
    text: str = ""                 # reply text / transcript / description
    path: Optional[Path] = None    # generated file (image, wav, pptx)
    url: str = ""                  # where the web layer serves that file
    mime: str = ""
    error: str = ""
    ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "text": self.text, "url": self.url,
                "mime": self.mime, "error": self.error,
                "ms": round(self.ms, 1), "meta": self.meta}


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _post_multipart(url: str, fields: Dict[str, str], file_field: str,
                    filename: str, data: bytes, mime: str,
                    timeout: float) -> bytes:
    boundary = "----realai" + uuid.uuid4().hex
    parts: List[bytes] = []
    for k, v in fields.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                      f"name=\"{k}\"\r\n\r\n{v}\r\n").encode())
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                  f"name=\"{file_field}\"; filename=\"{filename}\"\r\n"
                  f"Content-Type: {mime}\r\n\r\n").encode())
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _explain(exc: BaseException, where: str) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"{where} answered HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return f"{where} is not reachable ({exc.reason})"
    return f"{where} failed: {type(exc).__name__}: {exc}"


class Generative:
    """All optional generative backends behind one object."""

    #: persona used when the local LLM answers free conversation
    PERSONA = (
        "You are {name}, a fully local AI assistant owned by {owner}. You run "
        "entirely on the owner's machine: no cloud, no external API, no keys. "
        "Be warm, direct and genuinely helpful, like a good conversational "
        "assistant. Use markdown when it helps (lists, code blocks). If asked "
        "to do something you cannot do, say so plainly. Keep answers "
        "focused; do not pad."
    )

    def __init__(self, cfg: Config, storage: Any = None,
                 agent_name: str = "REAL", owner_name: str = "Owner") -> None:
        self.cfg = cfg
        self.storage = storage
        self.agent_name = agent_name
        self.owner_name = owner_name
        self.media_dir = cfg.media_dir

    # ----------------------------------------------------------- status
    @property
    def can_talk(self) -> bool:
        return bool(self.cfg.llm_url)

    @property
    def can_draw(self) -> bool:
        return bool(self.cfg.image_url)

    @property
    def can_listen(self) -> bool:
        return bool(self.cfg.stt_url)

    @property
    def can_speak(self) -> bool:
        return bool(self.cfg.tts_url or self.cfg.tts_command)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "talk": {"enabled": self.can_talk, "engine": "ollama",
                     "model": self.cfg.llm_model if self.can_talk else None},
            "images": {"enabled": self.can_draw, "engine": "stable-diffusion-webui"},
            "listen": {"enabled": self.can_listen, "engine": "whisper.cpp"},
            "speak": {"enabled": self.can_speak, "engine": "piper"},
            "slides": {"enabled": True, "engine": "built-in pptx writer"},
        }

    def _count(self, event: str, **detail: Any) -> None:
        if self.storage is not None:
            try:
                self.storage.count("generative", event, detail=detail or None)
            except Exception:  # noqa: BLE001
                pass

    def _new_file(self, ext: str) -> Path:
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.prune()
        name = f"{int(time.time())}-{uuid.uuid4().hex[:10]}.{ext}"
        return self.media_dir / name

    def _url_for(self, path: Path) -> str:
        return f"/media/{path.name}"

    def prune(self) -> int:
        """Delete generated files older than ``media_ttl_hours``."""
        if not self.media_dir.exists():
            return 0
        cutoff = time.time() - self.cfg.media_ttl_hours * 3600
        n = 0
        for p in self.media_dir.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except OSError:
                pass
        return n

    _NAME_RE = re.compile(r"[0-9]+-[0-9a-f]{10}\.(png|jpg|wav|pptx|md)")
    _MIMES = {"png": "image/png", "jpg": "image/jpeg", "wav": "audio/wav",
              "md": "text/markdown; charset=utf-8",
              "pptx": "application/vnd.openxmlformats-officedocument."
                      "presentationml.presentation"}

    def resolve_media(self, name: str) -> Optional[Path]:
        """Safe lookup for ``/media/<name>`` on disk (no traversal)."""
        if not self._NAME_RE.fullmatch(name):
            return None
        p = self.media_dir / name
        return p if p.is_file() else None

    def fetch_media(self, name: str) -> Optional[Dict[str, Any]]:
        """``{"mime", "data"}`` for a generated file — from disk, or from
        the database when the deploy is serverless (ephemeral disk)."""
        p = self.resolve_media(name)
        if p is not None:
            return {"mime": self._MIMES[p.suffix[1:]], "data": p.read_bytes()}
        if self.storage is not None and self._NAME_RE.fullmatch(name):
            try:
                return self.storage.media_get(name)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _persist(self, path: Path) -> None:
        """Mirror a generated file into the database when storage is remote,
        and prune old rows, so /media works across serverless instances."""
        st = self.storage
        if st is None or not getattr(st, "remote", False):
            return
        try:
            st.media_put(path.name, self._MIMES[path.suffix[1:]], path.read_bytes())
            st.media_prune(self.cfg.media_ttl_hours * 3600)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------- talk
    def chat(self, messages: Messages, system: Optional[str] = None,
             temperature: float = 0.7) -> GenResult:
        """One turn of fluent conversation via the local LLM."""
        t0 = time.time()
        if not self.can_talk:
            return GenResult(False, error="no local language model configured "
                                          "(set REALAI_LLM_URL to your Ollama)")
        sys_prompt = system or self.PERSONA.format(
            name=self.agent_name, owner=self.owner_name)
        payload = {
            "model": self.cfg.llm_model,
            "messages": [{"role": "system", "content": sys_prompt}] + messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            data = _post_json(f"{self.cfg.llm_url.rstrip('/')}/api/chat",
                              payload, self.cfg.llm_timeout)
            text = str((data.get("message") or {}).get("content", "")).strip()
        except Exception as exc:  # noqa: BLE001
            self._count("talk_error")
            return GenResult(False, error=_explain(exc, "the local language model"),
                             ms=(time.time() - t0) * 1000)
        self._count("talk", model=self.cfg.llm_model)
        return GenResult(True, text=text, ms=(time.time() - t0) * 1000,
                         meta={"model": self.cfg.llm_model,
                               "eval_count": data.get("eval_count")})

    def stream_chat(self, messages: Messages,
                    system: Optional[str] = None) -> Iterable[str]:
        """Token stream (NDJSON from Ollama). Yields text deltas."""
        if not self.can_talk:
            return
        sys_prompt = system or self.PERSONA.format(
            name=self.agent_name, owner=self.owner_name)
        payload = json.dumps({
            "model": self.cfg.llm_model,
            "messages": [{"role": "system", "content": sys_prompt}] + messages,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{self.cfg.llm_url.rstrip('/')}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as resp:
                for line in resp:
                    try:
                        obj = json.loads(line.decode("utf-8", "replace"))
                    except ValueError:
                        continue
                    delta = (obj.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if obj.get("done"):
                        break
            self._count("talk_stream", model=self.cfg.llm_model)
        except Exception as exc:  # noqa: BLE001
            self._count("talk_error")
            yield f"\n\n[{_explain(exc, 'the local language model')}]"

    # ----------------------------------------------------------- images
    def image(self, prompt: str, width: int = 768, height: int = 768,
              steps: int = 25, negative: str = "") -> GenResult:
        """Text → image via Stable Diffusion WebUI's txt2img API."""
        t0 = time.time()
        prompt = (prompt or "").strip()
        if not prompt:
            return GenResult(False, error="empty image prompt")
        if not self.can_draw:
            return GenResult(False, error="no local image model configured "
                                          "(set REALAI_IMAGE_URL to your "
                                          "Stable Diffusion WebUI, started "
                                          "with --api)")
        payload = {
            "prompt": prompt[:1500],
            "negative_prompt": negative or "blurry, low quality, watermark, text",
            "width": max(256, min(1536, int(width))),
            "height": max(256, min(1536, int(height))),
            "steps": max(4, min(60, int(steps))),
            "cfg_scale": 7,
            "batch_size": 1, "n_iter": 1,
        }
        try:
            data = _post_json(f"{self.cfg.image_url.rstrip('/')}/sdapi/v1/txt2img",
                              payload, self.cfg.image_timeout)
            b64 = (data.get("images") or [""])[0]
            if not b64:
                raise ValueError("no image in response")
            raw = base64.b64decode(b64.split(",", 1)[-1])
        except Exception as exc:  # noqa: BLE001
            self._count("image_error")
            return GenResult(False, error=_explain(exc, "the local image model"),
                             ms=(time.time() - t0) * 1000)
        path = self._new_file("png")
        path.write_bytes(raw)
        self._persist(path)
        self._count("image", bytes=len(raw))
        return GenResult(True, text=prompt, path=path, url=self._url_for(path),
                         mime="image/png", ms=(time.time() - t0) * 1000,
                         meta={"width": payload["width"],
                               "height": payload["height"]})

    # ----------------------------------------------------------- listen
    def transcribe(self, audio: bytes, filename: str = "audio.wav",
                   mime: str = "audio/wav") -> GenResult:
        """Speech → text via whisper.cpp's ``/inference``."""
        t0 = time.time()
        if not audio:
            return GenResult(False, error="empty audio")
        if not self.can_listen:
            return GenResult(False, error="no local speech recogniser configured "
                                          "(set REALAI_STT_URL to whisper-server)")
        try:
            raw = _post_multipart(
                f"{self.cfg.stt_url.rstrip('/')}/inference",
                {"response_format": "json", "temperature": "0.0"},
                "file", filename, audio, mime, self.cfg.llm_timeout)
            data = json.loads(raw.decode("utf-8", "replace"))
            text = str(data.get("text", "")).strip()
        except Exception as exc:  # noqa: BLE001
            self._count("listen_error")
            return GenResult(False, error=_explain(exc, "the local speech recogniser"),
                             ms=(time.time() - t0) * 1000)
        self._count("listen", bytes=len(audio))
        return GenResult(True, text=text, ms=(time.time() - t0) * 1000)

    # ------------------------------------------------------------ speak
    def speak(self, text: str) -> GenResult:
        """Text → WAV via Piper (HTTP server, or the CLI on stdin)."""
        t0 = time.time()
        text = re.sub(r"[*_`#>]+", "", (text or "")).strip()
        if not text:
            return GenResult(False, error="nothing to say")
        if not self.can_speak:
            return GenResult(False, error="no local voice configured (set "
                                          "REALAI_TTS_URL or REALAI_TTS_COMMAND "
                                          "for Piper)")
        text = text[:2000]
        try:
            if self.cfg.tts_url:
                req = urllib.request.Request(
                    self.cfg.tts_url, data=json.dumps({"text": text}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout) as r:
                    wav = r.read()
            else:
                cmd = shlex.split(self.cfg.tts_command) + ["--output_file", "-"]
                proc = subprocess.run(cmd, input=text.encode(), capture_output=True,
                                      timeout=self.cfg.llm_timeout)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-300:])
                wav = proc.stdout
            if not wav[:4] == b"RIFF":
                raise ValueError("voice engine did not return a WAV file")
        except Exception as exc:  # noqa: BLE001
            self._count("speak_error")
            return GenResult(False, error=_explain(exc, "the local voice"),
                             ms=(time.time() - t0) * 1000)
        path = self._new_file("wav")
        path.write_bytes(wav)
        self._persist(path)
        self._count("speak", chars=len(text))
        return GenResult(True, text=text, path=path, url=self._url_for(path),
                         mime="audio/wav", ms=(time.time() - t0) * 1000)

    # ----------------------------------------------------------- slides
    def presentation(self, topic: str, slides: Optional[List[Dict[str, Any]]] = None,
                     count: int = 6) -> GenResult:
        """Build a real ``.pptx``.

        With a local LLM the outline is written by the model; without one
        the caller must pass ``slides`` (``[{"title", "bullets": [...]}]``)
        or a plain structured deck is produced from the topic text.
        """
        t0 = time.time()
        topic = (topic or "").strip()
        if not topic and not slides:
            return GenResult(False, error="give me a topic or slides")
        source = "provided"
        if slides is None:
            slides, source = self._outline(topic, count)
        slides = _clean_slides(slides)
        if not slides:
            return GenResult(False, error="could not produce an outline")
        path = self._new_file("pptx")
        try:
            write_pptx(path, topic or slides[0]["title"], slides,
                       author=self.agent_name)
            self._persist(path)
        except Exception as exc:  # noqa: BLE001
            self._count("slides_error")
            return GenResult(False, error=f"pptx writer failed: {exc}")
        self._count("slides", n=len(slides), outline=source)
        summary = "\n".join(f"{i + 1}. **{s['title']}**" for i, s in enumerate(slides))
        return GenResult(True, text=summary, path=path, url=self._url_for(path),
                         mime=("application/vnd.openxmlformats-officedocument."
                               "presentationml.presentation"),
                         ms=(time.time() - t0) * 1000,
                         meta={"slides": len(slides), "outline": source})

    def _outline(self, topic: str, count: int):
        count = max(3, min(15, int(count)))
        if self.can_talk:
            ask = (f"Write a {count}-slide presentation outline about: {topic}.\n"
                   "Return ONLY JSON: a list of objects with keys \"title\" "
                   "(string) and \"bullets\" (3-5 short strings). The first "
                   "slide is the title slide (bullets = a one-line subtitle); "
                   "the last slide is a summary.")
            res = self.chat([{"role": "user", "content": ask}],
                            system="You output strict JSON and nothing else.",
                            temperature=0.4)
            if res.ok:
                m = re.search(r"\[.*\]", res.text, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        if isinstance(parsed, list) and parsed:
                            return parsed, "llm"
                    except ValueError:
                        pass
        # deterministic fallback: a sensible skeleton the owner can fill in
        return [
            {"title": topic, "bullets": [f"Prepared by {self.agent_name}"]},
            {"title": "Why it matters", "bullets": [
                f"Context of {topic}", "Who is affected", "What changes"]},
            {"title": "Key points", "bullets": [
                "Point one", "Point two", "Point three"]},
            {"title": "Details", "bullets": [
                "Evidence / data", "Examples", "Risks"]},
            {"title": "Next steps", "bullets": [
                "Decision needed", "Owner & timeline", "Success measure"]},
            {"title": "Summary", "bullets": [
                f"{topic}: the one thing to remember", "Questions?"]},
        ][:count], "template"


def _clean_slides(slides: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(slides, list):
        return out
    for s in slides[:20]:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title", "")).strip()[:120]
        bullets = s.get("bullets") or s.get("points") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        bullets = [str(b).strip()[:220] for b in bullets if str(b).strip()][:8]
        if title:
            out.append({"title": title, "bullets": bullets})
    return out


# ---------------------------------------------------------------------------
# Minimal PPTX writer (Office Open XML, stdlib zipfile only)
# ---------------------------------------------------------------------------

_CT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
{slides}
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

_PRES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""

_PRES_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
{slides}
</Relationships>"""

_MASTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="0B0F17"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
<p:txStyles><p:titleStyle><a:lvl1pPr><a:defRPr sz="4000"/></a:lvl1pPr></p:titleStyle><p:bodyStyle><a:lvl1pPr><a:defRPr sz="2000"/></a:lvl1pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr></p:otherStyle></p:txStyles>
</p:sldMaster>"""

_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

_THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="RealAI">
<a:themeElements>
<a:clrScheme name="RealAI"><a:dk1><a:srgbClr val="E8ECF4"/></a:dk1><a:lt1><a:srgbClr val="0B0F17"/></a:lt1><a:dk2><a:srgbClr val="9AA6BC"/></a:dk2><a:lt2><a:srgbClr val="141A26"/></a:lt2><a:accent1><a:srgbClr val="4F8CFF"/></a:accent1><a:accent2><a:srgbClr val="34D399"/></a:accent2><a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="F472B6"/></a:accent4><a:accent5><a:srgbClr val="A78BFA"/></a:accent5><a:accent6><a:srgbClr val="22D3EE"/></a:accent6><a:hlink><a:srgbClr val="4F8CFF"/></a:hlink><a:folHlink><a:srgbClr val="A78BFA"/></a:folHlink></a:clrScheme>
<a:fontScheme name="RealAI"><a:majorFont><a:latin typeface="Calibri Light"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>
<a:fmtScheme name="RealAI"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
</a:themeElements>
</a:theme>"""

_SLIDE_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""

_CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator>
<dcterms:created xsi:type="dcterms:W3CDTF">{ts}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{ts}</dcterms:modified>
</cp:coreProperties>"""

_APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>RealAI</Application><Slides>{n}</Slides></Properties>"""


def _textbox(sp_id: int, name: str, x: int, y: int, cx: int, cy: int,
             paragraphs: List[str], size: int, bold: bool = False,
             color: str = "E8ECF4", bullets: bool = False) -> str:
    paras = []
    for text in paragraphs:
        ppr = '<a:pPr marL="342900" indent="-342900"><a:buChar char="•"/></a:pPr>' \
            if bullets else "<a:pPr/>"
        paras.append(
            f'<a:p>{ppr}<a:r><a:rPr lang="en-US" sz="{size}" '
            f'b="{1 if bold else 0}" dirty="0"><a:solidFill><a:srgbClr '
            f'val="{color}"/></a:solidFill></a:rPr><a:t>{_xml(text)}</a:t>'
            f'</a:r></a:p>')
    if not paras:
        paras.append("<a:p><a:endParaRPr/></a:p>")
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sp_id}" name="{_xml(name)}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm>'
            f'<a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" anchor="t"><a:normAutofit/>'
            f'</a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def _accent_bar(sp_id: int, y: int) -> str:
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sp_id}" name="bar"/><p:cNvSpPr/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="838200" y="{y}"/>'
            f'<a:ext cx="1371600" cy="45720"/></a:xfrm><a:prstGeom prst="rect">'
            f'<a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="4F8CFF"/>'
            f'</a:solidFill></p:spPr></p:sp>')


def _slide_xml(index: int, total: int, title: str, bullets: List[str],
               is_title: bool, author: str) -> str:
    shapes = []
    if is_title:
        shapes.append(_textbox(2, "Title", 838200, 2286000, 10515600, 1600000,
                               [title], 4400, bold=True))
        shapes.append(_accent_bar(3, 3960000))
        shapes.append(_textbox(4, "Subtitle", 838200, 4100000, 10515600, 900000,
                               bullets[:1] or [author], 2000, color="9AA6BC"))
    else:
        shapes.append(_textbox(2, "Title", 838200, 685800, 10515600, 1100000,
                               [title], 3600, bold=True))
        shapes.append(_accent_bar(3, 1750000))
        shapes.append(_textbox(4, "Body", 838200, 2000000, 10515600, 4000000,
                               bullets, 2200, bullets=True))
    shapes.append(_textbox(5, "Footer", 838200, 6300000, 10515600, 400000,
                           [f"{author}  ·  {index}/{total}"], 1100,
                           color="5B6577"))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/>'
        '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
        '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>'
        '</a:xfrm></p:grpSpPr>' + "".join(shapes) +
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def write_pptx(path: Path, title: str, slides: List[Dict[str, Any]],
               author: str = "RealAI") -> Path:
    """Write a valid, openable .pptx (PowerPoint / LibreOffice / Keynote)."""
    n = len(slides)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ct_slides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/'
        f'vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n + 1))
    ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{i + 2}"/>'
                  for i in range(1, n + 1))
    rels = "\n".join(
        f'<Relationship Id="rId{i + 2}" Type="http://schemas.openxmlformats.org/'
        f'officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, n + 1))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CT.format(slides=ct_slides))
        z.writestr("_rels/.rels", _RELS)
        z.writestr("docProps/core.xml", _CORE.format(
            title=_xml(title), author=_xml(author), ts=ts))
        z.writestr("docProps/app.xml", _APP.format(n=n))
        z.writestr("ppt/presentation.xml", _PRES.format(ids=ids))
        z.writestr("ppt/_rels/presentation.xml.rels", _PRES_RELS.format(slides=rels))
        z.writestr("ppt/slideMasters/slideMaster1.xml", _MASTER)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _MASTER_RELS)
        z.writestr("ppt/slideLayouts/slideLayout1.xml", _LAYOUT)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _LAYOUT_RELS)
        z.writestr("ppt/theme/theme1.xml", _THEME)
        for i, s in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml",
                       _slide_xml(i, n, s["title"], s.get("bullets", []),
                                  i == 1, author))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", _SLIDE_RELS)
    return path
