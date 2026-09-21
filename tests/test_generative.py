"""Tests for the local generative layer (talk / images / voice / slides).

Fake HTTP servers stand in for Ollama, Stable Diffusion WebUI and
whisper.cpp so the wire protocols are exercised without any model.
"""

from __future__ import annotations

import base64
import json
import struct
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from realaiagent.config import Config
from realaiagent.generative import Generative, write_pptx
from tests.helpers import make_agent

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg==")


def _wav() -> bytes:
    data = b"\x00\x00" * 100
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


class FakeBackends(BaseHTTPRequestHandler):
    """One server pretending to be Ollama + SD WebUI + whisper + piper."""
    seen = []

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        FakeBackends.seen.append((self.path, self.headers.get("Content-Type", ""), raw))
        if self.path == "/api/chat":
            payload = json.loads(raw)
            last = payload["messages"][-1]["content"]
            if "presentation about" in last:
                content = json.dumps([
                    {"title": "Solar power", "bullets": ["a subtitle"]},
                    {"title": "Why", "bullets": ["sun", "cheap"]},
                    {"title": "Summary", "bullets": ["do it"]}])
            else:
                content = f"echo:{last} (system={payload['messages'][0]['content'][:9]})"
            if payload.get("stream"):
                body = b"".join(
                    json.dumps({"message": {"content": ch}, "done": False}).encode() + b"\n"
                    for ch in ("hel", "lo")) + json.dumps({"done": True}).encode() + b"\n"
                return self._send(200, body, "application/x-ndjson")
            return self._send(200, {"message": {"role": "assistant",
                                                "content": content},
                                    "eval_count": 7})
        if self.path == "/sdapi/v1/txt2img":
            return self._send(200, {"images": [base64.b64encode(PNG).decode()],
                                    "parameters": {}, "info": ""})
        if self.path == "/inference":
            ok = b"multipart/form-data" in self.headers.get("Content-Type", "").encode() \
                and b'name="file"' in raw and b"RIFF" in raw
            return self._send(200, {"text": " hello agent " if ok else ""})
        if self.path == "/tts":
            return self._send(200, _wav(), "audio/wav")
        # ---- hosted provider (OpenAI-compatible text API + image CDN) ----
        if self.path == "/openai":
            payload = json.loads(raw)
            msgs = payload["messages"]
            content = msgs[-1]["content"]
            if payload.get("model") == "openai-audio":
                return self._send(200, {"choices": [{"message": {
                    "content": "hello from hosted whisper"}}]})
            if isinstance(content, str) and "presentation about" in content:
                text = json.dumps([{"title": "Hosted deck", "bullets": ["sub"]},
                                   {"title": "Point", "bullets": ["a", "b"]},
                                   {"title": "Key takeaways", "bullets": ["c"]}])
            else:
                text = f"hosted:{content} sys={msgs[0]['content'][:9]}"
            return self._send(200, {"model": "openai-large", "choices": [
                {"message": {"role": "assistant", "content": text}}]})
        self._send(404, {"error": "no"})

    def do_GET(self):
        FakeBackends.seen.append((self.path, "", b""))
        if self.path.startswith("/prompt/"):
            return self._send(200, PNG, "image/png")
        if "model=openai-audio" in self.path:
            return self._send(200, b"ID3\x04\x00" + b"\x00" * 200, "audio/mpeg")
        self._send(404, {"error": "no"})


class GenerativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackends)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _cfg(self, **kw) -> Config:
        return Config(data_dir=Path(tempfile.mkdtemp(prefix="realai-gen-")),
                      tick_seconds=3600, **kw)

    def _on(self) -> Generative:
        return Generative(self._cfg(llm_url=self.base, image_url=self.base,
                                    stt_url=self.base, tts_url=self.base + "/tts"),
                          agent_name="REAL", owner_name="Boss")

    # ---- off by default ------------------------------------------------
    def test_everything_off_without_config(self):
        g = Generative(self._cfg())
        caps = g.capabilities()
        self.assertFalse(any(caps[k]["enabled"] for k in ("talk", "images",
                                                          "listen", "speak")))
        self.assertTrue(caps["slides"]["enabled"])
        self.assertFalse(g.chat([{"role": "user", "content": "hi"}]).ok)
        self.assertFalse(g.image("cat").ok)
        self.assertFalse(g.speak("hi").ok)
        self.assertFalse(g.transcribe(b"x").ok)
        self.assertIn("REALAI_LLM_URL", g.chat([{"role": "user", "content": "hi"}]).error)

    # ---- talk ----------------------------------------------------------
    def test_talk_via_ollama_protocol(self):
        g = self._on()
        r = g.chat([{"role": "user", "content": "how are you?"}])
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.text.startswith("echo:how are you?"))
        self.assertIn("You are R", r.text)          # persona reached the model
        self.assertEqual(r.meta["eval_count"], 7)

    def test_stream(self):
        g = self._on()
        self.assertEqual("".join(g.stream_chat([{"role": "user", "content": "x"}])),
                         "hello")

    def test_unreachable_backend_is_a_clean_error(self):
        g = Generative(self._cfg(llm_url="http://127.0.0.1:9", llm_timeout=2))
        r = g.chat([{"role": "user", "content": "x"}])
        self.assertFalse(r.ok)
        self.assertIn("not reachable", r.error)

    # ---- images --------------------------------------------------------
    def test_image_saved_and_served_name(self):
        g = self._on()
        r = g.image("a red fox in snow")
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.path.exists())
        self.assertEqual(r.path.read_bytes(), PNG)
        self.assertTrue(r.url.startswith("/media/"))
        self.assertIsNotNone(g.resolve_media(r.path.name))
        self.assertIsNone(g.resolve_media("../agent.db"))
        self.assertIsNone(g.resolve_media("owner_key.txt"))

    # ---- voice ---------------------------------------------------------
    def test_transcribe_multipart(self):
        g = self._on()
        r = g.transcribe(_wav())
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "hello agent")

    def test_speak_http(self):
        g = self._on()
        r = g.speak("**Hello** there")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.mime, "audio/wav")
        self.assertTrue(r.path.read_bytes().startswith(b"RIFF"))

    # ---- slides --------------------------------------------------------
    def test_pptx_is_valid_package(self):
        d = Path(tempfile.mkdtemp())
        p = write_pptx(d / "t.pptx", "T", [
            {"title": "T", "bullets": ["sub"]},
            {"title": "A & B <c>", "bullets": ["x", "y"]}])
        with zipfile.ZipFile(p) as z:
            names = set(z.namelist())
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("ppt/slides/slide2.xml", names)
            self.assertIn("ppt/slides/_rels/slide2.xml.rels", names)
            s2 = z.read("ppt/slides/slide2.xml").decode()
            self.assertIn("A &amp; B &lt;c&gt;", s2)
            self.assertIn("2/2", s2)
            self.assertIsNone(z.testzip())

    def test_presentation_llm_outline(self):
        g = self._on()
        r = g.presentation("solar power")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.meta, {"slides": 3, "outline": "llm", "cover_image": True})
        self.assertIn("Solar power", r.text)

    def test_presentation_template_without_llm(self):
        g = Generative(self._cfg())
        r = g.presentation("cats", count=4)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.meta["outline"], "template")
        self.assertEqual(r.meta["slides"], 4)

    def test_presentation_explicit_slides(self):
        g = Generative(self._cfg())
        r = g.presentation("", slides=[{"title": "Only", "bullets": "one"}])
        self.assertTrue(r.ok)
        self.assertEqual(r.meta["slides"], 1)

    # ---- prune ---------------------------------------------------------
    def test_prune(self):
        import os, time
        g = Generative(self._cfg(media_ttl_hours=1))
        g.media_dir.mkdir(parents=True)
        old = g.media_dir / "1-abcdefabcd.png"
        old.write_bytes(b"x")
        os.utime(old, (time.time() - 7200, time.time() - 7200))
        self.assertEqual(g.prune(), 1)
        self.assertFalse(old.exists())


class HostedProviderTests(unittest.TestCase):
    """REALAI_PROVIDER=hosted: keyless public inference, no local backends."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackends)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _gen(self, **kw) -> Generative:
        cfg = Config(data_dir=Path(tempfile.mkdtemp(prefix="realai-hosted-")),
                     tick_seconds=3600, provider="hosted",
                     hosted_text_url=self.base, hosted_image_url=self.base, **kw)
        return Generative(cfg, agent_name="NOVA", owner_name="Ali")

    def test_capabilities_all_on(self):
        caps = self._gen().capabilities()
        self.assertEqual(caps["provider"], "hosted")
        for k in ("talk", "images", "listen", "speak", "slides"):
            self.assertTrue(caps[k]["enabled"], k)
        self.assertEqual(caps["talk"]["engine"], "hosted")

    def test_talk(self):
        r = self._gen().chat([{"role": "user", "content": "hey"}])
        self.assertTrue(r.ok, r.error)
        self.assertTrue(r.text.startswith("hosted:hey"))
        self.assertIn("You are N", r.text)              # persona sent
        self.assertEqual(r.meta["engine"], "hosted")

    def test_image_speak_listen(self):
        g = self._gen()
        r = g.image("a fox")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.mime, "image/png")
        self.assertIsNotNone(g.fetch_media(r.path.name))
        r = g.speak("hello")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.mime, "audio/mpeg")
        self.assertTrue(r.path.read_bytes().startswith(b"ID3"))
        r = g.transcribe(b"RIFFxxxx", filename="a.webm")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.text, "hello from hosted whisper")

    def test_presentation_with_cover_image(self):
        r = self._gen().presentation("hosted deck")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.meta, {"slides": 3, "outline": "llm", "cover_image": True})
        with zipfile.ZipFile(r.path) as z:
            self.assertIn("ppt/media/cover.png", z.namelist())
            self.assertIn("r:embed", z.read("ppt/slides/slide1.xml").decode())
            self.assertIn("image", z.read("ppt/slides/_rels/slide1.xml.rels").decode())
            self.assertIsNone(z.testzip())

    def test_local_url_beats_hosted(self):
        g = self._gen(llm_url=self.base)       # local Ollama configured too
        self.assertEqual(g.capabilities()["talk"]["engine"], "ollama")
        r = g.chat([{"role": "user", "content": "x"}])
        self.assertTrue(r.text.startswith("echo:"))

    def test_hosted_down_is_clean(self):
        cfg = Config(data_dir=Path(tempfile.mkdtemp()), tick_seconds=3600,
                     provider="hosted", hosted_text_url="http://127.0.0.1:9",
                     hosted_image_url="http://127.0.0.1:9", hosted_timeout=2,
                     image_timeout=2)
        g = Generative(cfg)
        self.assertFalse(g.chat([{"role": "user", "content": "x"}]).ok)
        self.assertFalse(g.image("x").ok)


class ChatIntegrationTests(unittest.TestCase):
    """The website voice routes media requests and falls back to the LLM."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackends)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _agent(self, **kw):
        d = Path(tempfile.mkdtemp(prefix="realai-gen-"))
        agent = make_agent(tmp=d)
        for k, v in kw.items():
            setattr(agent.cfg, k, v)
        return agent

    def test_without_backends_still_honest(self):
        agent = self._agent()
        r = agent.reply("draw me a cat")
        self.assertIn("not connected a local generative model", r["response"])
        self.assertEqual(r["meta"]["intent"], "scope")

    def test_presentation_from_chat_without_llm(self):
        agent = self._agent()
        r = agent.reply("make a presentation about honey bees")
        self.assertEqual(r["meta"]["intent"], "slides")
        self.assertIn("honey bees", r["response"])
        att = r["meta"]["attachments"][0]
        self.assertEqual(att["kind"], "file")
        name = att["url"].split("/")[-1]
        self.assertIsNotNone(agent.generative.resolve_media(name))

    def test_image_from_chat(self):
        agent = self._agent(image_url=self.base)
        r = agent.reply("draw me a picture of a lighthouse at night")
        self.assertEqual(r["meta"]["intent"], "image")
        self.assertIn("![", r["response"])
        self.assertEqual(r["meta"]["attachments"][0]["kind"], "image")

    def test_llm_fallback_for_free_chat(self):
        agent = self._agent(llm_url=self.base)
        r = agent.reply("what do you think about the meaning of life?")
        self.assertEqual(r["meta"]["intent"], "talk")
        self.assertTrue(r["response"].startswith("echo:what do you think"))
        # real commands still go through the engine, not the LLM
        r2 = agent.reply("status")
        self.assertEqual(r2["meta"]["intent"], "status")

    def test_media_route_and_api(self):
        from realaiagent.web import WebApp
        agent = self._agent(image_url=self.base, llm_url=self.base)
        web = WebApp(agent)
        st, ct, body, _ = web.handle("GET", "/v1/generate/capabilities", {}, {})
        self.assertEqual(st, 200)
        caps = json.loads(body)["capabilities"]
        self.assertTrue(caps["images"]["enabled"])
        res = agent.generative.image("x")
        st, ct, body, hdr = web.handle("GET", res.url, {}, {})
        self.assertEqual((st, ct, body), (200, "image/png", PNG))
        st, *_ = web.handle("GET", "/media/../agent.db", {}, {})
        self.assertEqual(st, 404)
        # public transcribe rejects garbage cleanly
        st, _, body, _ = web.handle("POST", "/public/transcribe", {},
                                    {"content-type": "application/json"},
                                    json.dumps({"audio": ""}).encode())
        self.assertEqual(st, 400)
        # keyed talk endpoint
        _, key, _ = agent.keys.ensure_owner_key()
        st, _, body, _ = web.handle(
            "POST", "/v1/generate/talk", {},
            {"authorization": f"Bearer {key}", "content-type": "application/json"},
            json.dumps({"message": "hi"}).encode())
        self.assertEqual(st, 200, body)
        self.assertTrue(json.loads(body)["response"].startswith("echo:hi"))


if __name__ == "__main__":
    unittest.main()


class ArchiveHookTests(unittest.TestCase):
    """Every generated file is handed to the archive callable (Telegram)."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackends)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_image_and_deck_are_archived(self):
        cfg = Config(data_dir=Path(tempfile.mkdtemp(prefix="realai-arch-")),
                     tick_seconds=3600, provider="hosted",
                     hosted_text_url=self.base, hosted_image_url=self.base)
        gen = Generative(cfg, agent_name="NOVA", owner_name="Ali")
        seen = []
        gen.archive = lambda name, data, mime, kind: seen.append(
            (name, len(data) > 0, mime, kind)) or "file-1"
        self.assertTrue(gen.image("a cat").ok)
        self.assertTrue(gen.presentation("tea", with_image=False).ok)
        kinds = [s[3] for s in seen]
        self.assertIn("image", kinds)
        self.assertIn("slides", kinds)
        self.assertTrue(all(s[1] for s in seen))

    def test_archive_failure_never_breaks_generation(self):
        cfg = Config(data_dir=Path(tempfile.mkdtemp(prefix="realai-arch-")),
                     tick_seconds=3600, provider="hosted",
                     hosted_text_url=self.base, hosted_image_url=self.base)
        gen = Generative(cfg, agent_name="NOVA", owner_name="Ali")

        def boom(*a):
            raise RuntimeError("telegram down")
        gen.archive = boom
        self.assertTrue(gen.image("a dog").ok)


class TelegramSendFileTests(unittest.TestCase):
    def test_multipart_upload_returns_file_id(self):
        import json as _json
        from realaiagent import telegram as tg_mod
        from realaiagent.storage import Storage

        class Fake(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                ok = (b'name="document"; filename="x.png"' in body
                      and b"#image x.png" in body
                      and b'name="message_thread_id"\r\n\r\n12' in body
                      and self.path.endswith("/sendDocument"))
                out = _json.dumps({"ok": ok, "result": {
                    "document": {"file_id": "FID" if ok else ""}}}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(out)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        old = tg_mod._API
        tg_mod._API = f"http://127.0.0.1:{httpd.server_address[1]}/bot{{token}}/{{method}}"
        try:
            st = Storage(Path(tempfile.mkdtemp(prefix="realai-tg-")) / "db.sqlite")
            tg = tg_mod.Telegram("t", "1", st)
            fid = tg.send_file("x.png", b"\x89PNG", "image/png",
                               caption="#image x.png", thread_id=12)
            self.assertEqual(fid, "FID")
        finally:
            tg_mod._API = old
            httpd.shutdown()
