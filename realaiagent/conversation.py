"""Conversational voice — multi-turn context, memory recall, honest scope.

The engine (:mod:`realaiagent.engine`) is a *pipeline*: it parses, plans,
acts and reports. That is exactly right for an API, and exactly wrong for a
chat window — a pipeline answers each message as if the conversation had
just begun, in the voice of a status report.

This module is the layer that makes it talk like a person. Three things,
all of them local, none of them an external model:

1. **Multi-turn context.** "turn it off" knows what *it* is. "and now?"
   continues the previous topic instead of restarting. Pronouns, ellipsis
   and bare follow-ups are resolved against the session's own history,
   which lives in the database so it survives serverless freezes.

2. **Memory recall.** When what you say overlaps something it already
   stored, it says so — and quotes it. "I remember you told me…" is only
   ever produced from a real row in ``memories``.

3. **Honest scope.** It has no internet, no external AI, no live data feed
   and no feelings beyond its own local mood model. When asked for
   something it does not have, it says that plainly and offers what it
   *can* do. Pretending is the one thing a local agent must never do,
   because the owner can read every line of its source.

Everything here is deterministic: no randomness, no network, no model
calls. Same input + same history → same reply, which is what makes it
testable.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .nlp import normalize, tokenize

#: Words that carry no topical signal when searching memory.
STOPWORDS = frozenset("""
a about an and are as at be been being but by can could did do does doing
done down for from had has have having he her here hers him his how i if in
into is it its just me my no not of on or our she so some than that the
their them then there these they this those to too was we were what when
where which who why will with you your yours please tell says said say
""".split())

#: Bare follow-ups: no new information, so the previous topic continues.
FOLLOWUPS: Dict[str, Tuple[str, ...]] = {
    "more": ("tell me more", "more", "go on", "continue", "keep going",
             "elaborate", "and then", "what else"),
    "why": ("why", "why is that", "how come", "explain why", "reason"),
    "again": ("again", "repeat", "say that again", "once more"),
    "next": ("and now", "now what", "what now", "what's next",
             "what is next", "anything else", "so"),
}

#: Words that point back at something already mentioned.
ANAPHORA = ("it", "that", "this", "them", "those", "the same", "same one",
            "there", "he", "she")

#: Messages that open with one of these are *instructions*, not references:
#: in "remember that X" the word "that" is a complementiser, not a pronoun,
#: so rewriting the sentence against the last topic would destroy it.
INSTRUCTION_RE = re.compile(
    r"^\s*(remember|teach|learn|train|note|goal|new goal|set goal|approve|"
    r"deny|run|execute|status|counters|help|hi|hello|hey|thanks|thank)\b",
    re.IGNORECASE)

#: An anaphora rewrite is only attempted on a short message: a long one is
#: carrying its own subject and does not need one.
ANAPHORA_MAX_WORDS = 9

#: Requests this agent genuinely cannot fulfil — grouped so the answer can
#: name the missing capability instead of a vague "I can't".
OUT_OF_SCOPE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("internet", ("browse the internet", "search the web", "search online",
                  "google it", "look it up online", "go online",
                  "open a website", "fetch a url", "download from",
                  "check the internet", "web search")),
    ("live data", ("weather", "stock price", "share price", "latest news",
                   "today's news", "todays news", "news right now",
                   "live score", "current price", "exchange rate",
                   "flight status", "traffic right now")),
    ("external AI", ("ask chatgpt", "ask gpt", "use gpt", "call openai",
                     "ask gemini", "ask claude", "ask an llm",
                     "use a language model", "call another ai",
                     "ask another ai", "let chatgpt")),
    ("media generation", ("draw me", "draw a", "generate an image",
                          "generate image", "make a picture",
                          "create an image", "paint a", "make me a logo",
                          "sing a song", "sing me", "make a video",
                          "create a video", "generate audio",
                          "make music")),
    ("arbitrary messaging", ("send an email to", "email my friend",
                             "text my friend", "message my friend",
                             "call my", "whatsapp", "post on twitter",
                             "tweet this")),
    ("money", ("buy it", "purchase", "pay for", "transfer money",
               "send money", "order it online", "checkout", "pay the bill")),
)

#: Honest answers per missing capability. They state the limit, then what
#: the agent *can* do — the useful half of an honest "no".
SCOPE_ANSWERS: Dict[str, str] = {
    "internet": (
        "I have no internet access — that is deliberate. My owner built me "
        "so that nothing outside this machine is ever called: no external "
        "AI, no external API keys, no third party reading our chat. "
        "What I do have is my own world: registered devices I can control "
        "(with permission), my memory of what you have told me, my goal "
        "queue, and a ledger that counts every single thing I do. Ask me "
        "for `status` or `counters` and I will show you the numbers."),
    "live data": (
        "I cannot see live data — no weather, no prices, no news feed. "
        "Nothing reaches me from outside; I only know what happens on this "
        "machine and what you or my owner teach me. If you tell me a fact, "
        "I store it in my own memory and I can recall it later, honestly "
        "labelled as something you said rather than something I looked up."),
    "external AI": (
        "There is no other AI inside me to ask. My language understanding "
        "is a small trainable classifier my owner wrote — every weight of "
        "it lives in my own database, and every example I am taught "
        "reweights it immediately. That means I am narrow but mine: I will "
        "never hand your words to somebody else's model. Teach me "
        "phrasing with `teach: \"<sentence>\" means <intent>` and I get "
        "better at understanding you specifically."),
    "media generation": (
        "I cannot make images, audio or video right now — my owner has not "
        "connected a local generative model. I can once they run Stable "
        "Diffusion (images) or Piper (voice) on this machine and point me "
        "at it — nothing external, still no keys. What I can do today is "
        "build a presentation (.pptx) and act: "
        "switch a registered device on or off, run a command my owner "
        "allows, read or write a file inside the roots I am given, and "
        "learn a multi-step skill from something that worked. Each of "
        "those goes through a permission gate, and each one is counted."),
    "arbitrary messaging": (
        "I do not message people. I have exactly two outbound channels and "
        "both belong to my owner: their own Telegram bot, and their own "
        "SMTP inbox (used so access requests reach them fast). I will never "
        "contact a third party on your behalf."),
    "money": (
        "I never spend money or place orders. The only billing in me is "
        "internal bookkeeping my owner runs: clients start PENDING, the "
        "owner approves them and mints their key, and each request either "
        "deducts from a balance or is free for VIP clients. No card, no "
        "wallet, no checkout — nothing leaves this machine."),
}

#: Questions about my own nature deserve a straight answer, not a dodge.
SELF_QUESTIONS: Tuple[Tuple[str, str], ...] = (
    (("are you alive", "are you conscious", "are you sentient",
      "do you have feelings", "do you feel", "are you real",
      "are you human", "do you dream"),
     "Honestly? No — and I would rather say that than perform. I am a "
     "program with a local mood model: two numbers, valence and arousal, "
     "that move when things succeed or fail, plus five drives my owner can "
     "inspect and adjust. That is a useful control signal for me, not an "
     "inner life. What is real is the counting: every request, thought and "
     "action I take is written to a durable ledger you can audit."),
    (("who made you", "who built you", "who created you", "who owns you",
      "whose are you"),
     "My owner wrote me — every line, in pure Python standard library. No "
     "framework, no pretrained model, no vendor. My name is {agent}, my "
     "owner is {owner}, and my weights, memory and ledger all live in a "
     "SQLite file on their machine. That is the whole point: you can read "
     "my source and know exactly what I am."),
    (("what can you do", "what are you capable of", "help me understand you",
      "what do you actually do"),
     "Concretely: I parse what you say with a trainable local intent model, "
     "plan steps, push them through a permission gate (allow / ask / deny), "
     "act on registered devices, run commands and touch files inside the "
     "roots my owner granted, remember what you tell me, keep a goal queue "
     "that I advance on my own timer, learn skills from what worked, and "
     "count every single event. Say `status` for my state or `counters` "
     "for the ledger."),
)


class Session:
    """One conversation: an id plus its recent turns."""

    def __init__(self, session_id: str, max_turns: int = 12) -> None:
        self.id = session_id
        self.max_turns = max(1, int(max_turns))
        self.turns: List[Dict[str, Any]] = []
        self.created = time.time()
        self.last_seen = self.created

    # ------------------------------------------------------------ history

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last(self) -> Optional[Dict[str, Any]]:
        return self.turns[-1] if self.turns else None

    def add(self, message: str, response: str, intent: str = "",
            slots: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        turn = {"ts": time.time(), "message": message, "response": response,
                "intent": intent, "slots": slots or {}}
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        self.last_seen = turn["ts"]
        return turn

    def topic(self) -> Dict[str, Any]:
        """The subject carried into the next turn (last non-empty slots)."""
        for turn in reversed(self.turns):
            slots = turn.get("slots") or {}
            if any(slots.get(k) for k in ("device", "device_name", "note",
                                          "number", "verb", "path", "skill")):
                return {"intent": turn.get("intent", ""), **slots}
        return {"intent": self.turns[-1].get("intent", "")} if self.turns \
            else {}


class ConversationManager:
    """Sessions + the voice that turns pipeline output into conversation."""

    #: Keep at most this many live sessions in memory (the DB is the truth).
    MAX_SESSIONS = 256

    def __init__(self, agent: Any, max_turns: Optional[int] = None) -> None:
        self.agent = agent
        self.max_turns = int(max_turns or getattr(
            getattr(agent, "cfg", None), "conversation_turns", 12) or 12)
        self._sessions: Dict[str, Session] = {}

    # ----------------------------------------------------------- sessions

    def new_session_id(self, visitor: str = "") -> str:
        seed = f"{visitor}|{time.time()}|{len(self._sessions)}"
        return "s-" + hashlib.sha1(seed.encode()).hexdigest()[:16]

    def session(self, session_id: Optional[str] = None,
                visitor: str = "") -> Session:
        """Fetch (or create) a session, rehydrated from the database."""
        sid = (session_id or "").strip()
        if not sid or len(sid) > 64 or not re.match(r"^[A-Za-z0-9_.-]+$", sid):
            sid = self.new_session_id(visitor)
        cached = self._sessions.get(sid)
        if cached is not None:
            return cached
        sess = Session(sid, self.max_turns)
        try:
            sess.turns = self.agent.storage.session_turns(sid, self.max_turns)
        except Exception:  # noqa: BLE001 - a cold DB must not break chat
            sess.turns = []
        if len(self._sessions) >= self.MAX_SESSIONS:
            oldest = min(self._sessions.values(),
                         key=lambda s: s.last_seen, default=None)
            if oldest is not None:
                self._sessions.pop(oldest.id, None)
        self._sessions[sid] = sess
        return sess

    def forget(self, session_id: str) -> bool:
        self._sessions.pop(session_id, None)
        try:
            return self.agent.storage.session_clear(session_id)
        except Exception:  # noqa: BLE001
            return False

    def sessions(self, limit: int = 50) -> List[str]:
        try:
            return self.agent.storage.session_ids(limit)
        except Exception:  # noqa: BLE001
            return list(self._sessions)

    # --------------------------------------------------------------- reply

    def reply(self, text: str, session_id: Optional[str] = None,
              sender: str = "web", scopes: Optional[List[str]] = None,
              key_id: Optional[str] = None,
              visitor: str = "") -> Dict[str, Any]:
        """One conversational turn.

        Returns the shape the API already returns (``response`` + ``meta``)
        plus ``session_id``, so a caller can swap this in without changing
        its client. ``meta`` gains ``context`` (what history was used) and
        ``recall`` (the memories that were quoted back).
        """
        t0 = time.time()
        raw = (text or "").strip()
        sess = self.session(session_id, visitor=visitor)

        answer, intent, confidence, extra = self._answer(
            raw, sess, sender=sender, scopes=scopes, key_id=key_id)
        # ``extra`` may carry the flag the voice needs to know who it is
        # talking to; it is popped before it can leak into meta.

        slots = extra.pop("_slots", {})
        sess.add(raw, answer, intent, slots)
        self._persist(sess.id, raw, answer, sender, intent, slots)

        meta: Dict[str, Any] = {
            "intent": intent,
            "confidence": confidence,
            "top_k": extra.get("top_k", []),
            "plan": extra.get("plan", {}),
            "actions": extra.get("actions", []),
            "success": extra.get("success", True),
            "request_id": extra.get("request_id"),
            "context": extra.get("context", {"turn": sess.turn_count,
                                             "resolved": False}),
            "recall": extra.get("recall", []),
            "attachments": extra.get("attachments", []),
            "generation": extra.get("generation"),
            "voice": True,
            "duration_ms": round((time.time() - t0) * 1000, 1),
        }
        return {"response": answer, "session_id": sess.id, "meta": meta}

    def _answer(self, raw: str, sess: Session, sender: str,
                scopes: Optional[List[str]],
                key_id: Optional[str]) -> Tuple[str, str, float,
                                                Dict[str, Any]]:
        """Decide what to say, in the order the product promises:

        honesty first (never fake a capability), then conversation
        (follow-ups and pronouns), then the engine itself.
        """
        if not raw:
            return ("I did not get a message there — say something and I "
                    "will answer.", "chat", 0.0, {})

        media = self._media_request(raw, sess, sender)
        if media is not None:
            return media

        hosted = bool(getattr(getattr(self.agent, "generative", None),
                              "hosted", False))
        scope = None if hosted else self._out_of_scope(raw)
        if scope:
            self.agent.storage.count("chat", "honest_scope")
            return scope, "scope", 0.95, {}

        about_self = None if hosted else self._self_question(raw)
        if about_self:
            self.agent.storage.count("chat", "identity_answer")
            return about_self, "identity", 0.9, {}

        follow = self._followup_answer(raw, sess)
        if follow is not None:
            self.agent.storage.count("chat", "context_followup")
            answer, intent = follow
            return answer, intent, 0.85, {"context": {
                "turn": sess.turn_count, "resolved": True, "kind": intent}}

        expanded, context = self._resolve(raw, sess)
        recalled = self._recall(raw)
        result = self.agent.handle_message(
            expanded, sender=sender, scopes=scopes or [], key_id=key_id)
        meta = result.get("meta") or {}
        intent = meta.get("intent", "chat")
        answer = self._voice(result.get("response", ""), intent, sess,
                             recalled, context,
                             is_owner="owner" in (scopes or []))
        llm = self._llm_fallback(raw, sess, meta, result.get("response", ""))
        if llm is not None:
            answer, intent = llm, "talk"
        return answer, intent, meta.get("confidence", 0.0), {
            "top_k": meta.get("top_k", []),
            "plan": meta.get("plan", {}),
            "actions": meta.get("actions", []),
            "success": meta.get("success", True),
            "request_id": meta.get("request_id"),
            "context": context,
            "recall": recalled,
            "_slots": self._topic_slots(meta, expanded),
        }

    def _persist(self, session_id: str, message: str, response: str,
                 sender: str, intent: str,
                 slots: Dict[str, Any]) -> None:
        try:
            self.agent.storage.session_add_turn(
                session_id, message[:2000], response[:4000], sender=sender,
                intent=intent, slots=slots)
        except Exception:  # noqa: BLE001 - chat must survive a locked DB
            self.agent.storage.count("errors", "session_persist")

    # ------------------------------------------------------- 1. honesty

    # ------------------------------------------------- generative abilities

    _IMAGE_RE = re.compile(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
        r"(?:draw|paint|sketch|illustrate|generate|create|make|render|imagine)"
        r"\s+(?:me\s+)?(?:an?\s+|the\s+)?"
        r"(?:image|picture|photo|illustration|drawing|painting|logo|artwork|"
        r"poster|icon|wallpaper)?\s*(?:of|about|showing|with|for|:)?\s*(?P<what>.+)$",
        re.IGNORECASE)
    _DRAW_WORDS = ("draw", "paint", "sketch", "illustrat", "image", "picture",
                   "photo", "logo", "poster", "wallpaper", "icon", "artwork",
                   "render", "imagine")
    _SLIDES_RE = re.compile(
        r"(?:presentation|slides?|slide deck|deck|pptx|powerpoint|keynote)"
        r"\s*(?:on|about|for|of|:)?\s*(?P<what>.+)$", re.IGNORECASE)
    _SPEAK_RE = re.compile(
        r"^(?:please\s+)?(?:say|speak|read (?:this |it )?(?:aloud|out loud)|"
        r"say (?:this |it )?(?:aloud|out loud)|talk)\s*:?\s*(?P<what>.*)$",
        re.IGNORECASE)

    def _media_request(self, raw: str, sess: Session,
                       sender: str) -> Optional[Tuple[str, str, float,
                                                      Dict[str, Any]]]:
        """Image / presentation / voice requests → the generative layer.

        Returns ``None`` when the message is not such a request, or when
        the needed backend is not configured (the honest scope answer
        then handles it, exactly as before).
        """
        gen = getattr(self.agent, "generative", None)
        if gen is None:
            return None
        t = raw.strip()
        low = t.lower()

        m = self._SLIDES_RE.search(t)
        if m and any(w in low for w in ("make", "create", "build", "prepare",
                                        "generate", "write", "presentation",
                                        "deck", "pptx", "powerpoint")):
            topic = m.group("what").strip(" .!?\"'")
            n = re.search(r"(\d{1,2})[- ]slide", low)
            res = gen.presentation(topic, count=int(n.group(1)) if n else 6)
            self.agent.storage.count("chat", "media_slides")
            if not res.ok:
                return (f"I tried to build that deck but could not: {res.error}",
                        "slides", 0.9, {"generation": res.to_dict()})
            how = ("written by my local language model" if res.meta.get("outline")
                   == "llm" else "a structured skeleton you can fill in — "
                   "connect a local language model and I will write the "
                   "content too")
            return (f"Here is your presentation on **{topic}** "
                    f"({res.meta['slides']} slides, {how}):\n\n{res.text}\n\n"
                    f"[Download the .pptx]({res.url})",
                    "slides", 0.9, {"generation": res.to_dict(),
                                    "attachments": [{"kind": "file",
                                                     "url": res.url,
                                                     "mime": res.mime,
                                                     "name": "presentation.pptx"}]})

        m = self._IMAGE_RE.match(t)
        if m and any(w in low for w in self._DRAW_WORDS):
            if not gen.can_draw:
                return None       # fall through to the honest scope answer
            what = m.group("what").strip(" .!?\"'")
            res = gen.image(what)
            self.agent.storage.count("chat", "media_image")
            if not res.ok:
                return (f"I tried to draw that but could not: {res.error}",
                        "image", 0.9, {"generation": res.to_dict()})
            return (f"Here is what I made for *{what}*:\n\n![{what}]({res.url})",
                    "image", 0.9, {"generation": res.to_dict(),
                                   "attachments": [{"kind": "image",
                                                    "url": res.url,
                                                    "mime": res.mime}]})

        m = self._SPEAK_RE.match(t)
        if m and m.group("what").strip() and gen.can_speak:
            what = m.group("what").strip(" \"'")
            res = gen.speak(what)
            self.agent.storage.count("chat", "media_speak")
            if not res.ok:
                return (f"I could not voice that: {res.error}", "speak", 0.9,
                        {"generation": res.to_dict()})
            return (f"🔊 *{what}*", "speak", 0.9,
                    {"generation": res.to_dict(),
                     "attachments": [{"kind": "audio", "url": res.url,
                                      "mime": res.mime}]})
        return None

    def _llm_fallback(self, raw: str, sess: Session, meta: Dict[str, Any],
                      engine_text: str) -> Optional[str]:
        """When the trainable classifier did not understand and a local
        language model is configured, let the model talk — with the recent
        turns as context. The engine still handled every real command."""
        gen = getattr(self.agent, "generative", None)
        if gen is None or not gen.can_talk:
            return None
        intent = meta.get("intent", "chat")
        # Only where the engine had nothing specific to say: the generic
        # "chat" reply or its explicit "I'm not sure" fallback. Commands,
        # status, teaching, goals and permissions stay with the engine.
        unsure = ("I'm not sure what you'd like me to do there" in engine_text
                  or intent in ("chat", "unknown"))
        if not unsure:
            return None
        history: List[Dict[str, str]] = []
        for turn in list(sess.turns)[-6:]:
            history.append({"role": "user", "content": turn.get("message", "")})
            history.append({"role": "assistant", "content": turn.get("response", "")})
        history.append({"role": "user", "content": raw})
        facts = self._recall(raw)
        system = gen.PERSONA.format(name=gen.agent_name, owner=gen.owner_name)
        if facts:
            system += " Things the user told you earlier (your own memory): " \
                      + "; ".join(facts)
        res = gen.chat(history, system=system)
        if not res.ok:
            self.agent.storage.count("chat", "talk_unavailable")
            return None
        self.agent.storage.count("chat", "talk")
        return res.text

    def _out_of_scope(self, text: str) -> Optional[str]:
        t = normalize(text)
        for capability, phrases in OUT_OF_SCOPE:
            for phrase in phrases:
                if phrase in t:
                    return SCOPE_ANSWERS[capability]
        return None

    def _self_question(self, text: str) -> Optional[str]:
        t = normalize(text)
        snap = self.agent.mind.snapshot()
        for triggers, answer in SELF_QUESTIONS:
            if any(trig in t for trig in triggers):
                return answer.format(agent=snap["agent"], owner=snap["owner"])
        return None

    # --------------------------------------------------- 2. follow-ups

    def _followup_answer(self, text: str,
                         sess: Session) -> Optional[Tuple[str, str]]:
        """Answer a message that carries no new subject of its own.

        "again", "why", "tell me more" and "and now?" are the four things a
        real conversation is full of and a stateless pipeline cannot answer
        at all. Each one is built from the session's own history — never
        invented.
        """
        if not sess.turns:
            return None
        kind = self._followup_kind(normalize(text))
        if kind is None or self._has_content(text):
            return None
        last = sess.last or {}
        last_response = str(last.get("response") or "").strip()
        last_intent = str(last.get("intent") or "")

        if kind == "again":
            if not last_response:
                return None
            return (f"Sure — once more.\n\n{last_response}", "repeat")

        if kind == "why":
            reason = self._why(last_intent, sess)
            return reason, "explain"

        if kind == "more":
            return self._more(last_response, last_intent), "elaborate"

        # "next": what is actually on my plate right now
        return self._next(), "next"

    def _why(self, intent: str, sess: Session) -> str:
        """Explain the previous turn from real state, not from a guess."""
        snap = self.agent.mind.snapshot()
        topic = sess.topic()
        parts = ["Here is my honest reasoning."]
        if intent in ("repeat", "elaborate", "next", "explain"):
            parts.append(
                "That last message was a follow-up, so I answered it out of "
                "our own conversation history rather than building a plan — "
                "there was nothing new to parse and nothing to act on.")
        elif intent in ("status", "counters"):
            parts.append(
                "You asked to see inside, so I read my own state straight "
                "out of the database — mood, drives, goals and the ledger. "
                "I never estimate those numbers, I report them.")
        elif intent in ("command", "device.control") or topic.get("device"):
            parts.append(
                f"Your message matched the `{intent}` intent in my local "
                f"model, so I built a plan and pushed every step through my "
                f"permission gate. That gate — allow / ask / deny — is why "
                f"an action can come back as 'awaiting your permission' "
                f"instead of just happening.")
        elif intent == "learn":
            parts.append(
                "You gave me something to remember, so I wrote it to my "
                "memory table. Storing it is the only honest response to a "
                "fact — I do not have a way to verify it against the world.")
        elif intent in ("scope", "identity"):
            parts.append(
                "I answered that one from my own limits rather than from a "
                "parser match: saying what I cannot do is more useful to you "
                "than a confident guess.")
        else:
            parts.append(
                f"My classifier read that as `{intent or 'chat'}` with the "
                f"confidence you saw. When confidence is low I would rather "
                f"ask than act — my owner can teach me the phrasing with "
                f"`teach: \"<sentence>\" means <intent>`.")
        parts.append(
            f"Right now my mood is {snap['mood']['note']} "
            f"(valence {snap['mood']['valence']}) and I have "
            f"{snap['active_goals']} active goal(s), both of which move how "
            f"eagerly I take on new work.")
        return " ".join(parts)

    def _more(self, last_response: str, intent: str) -> str:
        """Elaborate on the previous answer with real extra detail."""
        goals = self.agent.mind.goals(status="active")[:3]
        c = self.agent.storage.get_counters()
        t = c.get("totals", {})
        lines = []
        if last_response:
            lines.append(f"Picking up from there: {last_response}")
        if goals:
            lines.append("On my goal queue right now: " + "; ".join(
                f"“{g['description']}” (priority {g['priority']:.2f})"
                for g in goals) + ".")
        else:
            lines.append("My goal queue is empty — my owner has not given me "
                         "anything to work on between conversations.")
        ledger = (f"By the numbers: {t.get('chat', 0)} chat messages, "
                  f"{t.get('actions', 0)} actions, {t.get('thoughts', 0)} of "
                  f"my own thoughts, {c.get('grand_total', 0)} events counted "
                  f"in total")
        if intent != "counters":
            ledger += " — say `counters` for the full ledger"
        lines.append(ledger + ".")
        return " ".join(lines)

    def _next(self) -> str:
        """'And now?' — answered from the goal queue and recent activity."""
        snap = self.agent.mind.snapshot()
        goals = self.agent.mind.goals(status="active")[:1]
        pending = self.agent.permissions.pending_list()[:3]
        lines = [f"Autonomy is {'on' if snap['autonomy'] else 'OFF'} — "
                 f"I think every {self.agent.cfg.tick_seconds:.0f}s "
                 f"whether or not you message me."]
        if goals:
            g = goals[0]
            done = int(g.get("steps_done") or 0)
            total = len(g.get("steps") or [])
            lines.append(f"Next up is “{g['description']}” "
                         f"({done}/{total} steps done).")
        else:
            lines.append("Nothing is queued, so right now I am just here, "
                         "talking to you.")
        if pending:
            lines.append(f"I also have {len(pending)} action(s) waiting for "
                         f"permission — say `approve` and I may proceed.")
        return " ".join(lines)

    # ------------------------------------------------------- 3. context

    def _resolve(self, text: str,
                 sess: Session) -> Tuple[str, Dict[str, Any]]:
        """Rewrite an elliptical message into a self-contained one.

        The engine is stateless per call, so the context work happens here:
        pronouns and bare follow-ups are expanded using the session's last
        real topic. The rewrite is *conservative* — when in doubt the
        original text is passed through untouched, because a wrong guess is
        worse than no guess.
        """
        t = normalize(text)
        context: Dict[str, Any] = {"turn": sess.turn_count,
                                   "resolved": False, "kind": None}
        if not sess.turns:
            return text, context

        topic = sess.topic()
        if self._refers_back(t) and (topic.get("device_name")
                                     or topic.get("note")):
            expanded = self._rewrite_anaphora(t, topic)
            if expanded:
                context.update(resolved=True, kind="device_anaphora",
                               device=topic.get("device"),
                               device_name=topic.get("device_name"),
                               topic=topic.get("intent", ""))
                return expanded, context

        if self._refers_back(t) and topic.get("intent"):
            context.update(resolved=True, kind="topic_anaphora",
                           topic=topic["intent"])

        return text, context

    @staticmethod
    def _verb_from(text: str, topic: Dict[str, Any]) -> Optional[str]:
        """The verb the *new* message asks for, or the previous one.

        ``extract_verb`` needs an exact phrase ("turn off"), and people write
        "turn *it* off" — the pronoun splits the phrase. So the direction is
        read from the tokens first, and only then from the phrase table.
        """
        from .nlp import extract_verb
        tokens = set(tokenize(text))
        lowered = normalize(text)
        if "off" in tokens or any(w in lowered for w in
                                  ("shut down", "shutdown", "disable",
                                   "deactivate", "stop ", "kill")):
            return "off"
        if "on" in tokens or any(w in lowered for w in
                                 ("enable", "activate", "light up",
                                  "wake up", "power up", "start ")):
            return "on"
        return extract_verb(text) or topic.get("verb")

    @staticmethod
    def _rewrite_anaphora(text: str, topic: Dict[str, Any]) -> Optional[str]:
        """Turn "turn it off" into "turn off the Living Room Lamp".

        The subject always comes from the previous turn; the verb and any
        number come from the new message. If there is nothing to build a
        sentence from, this returns None and the original text is used —
        a wrong guess is worse than no guess.
        """
        name = topic.get("device_name")
        verb = ConversationManager._verb_from(text, topic)
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if name and match and verb in (None, "set"):
            return f"set the {name} to {match.group(0)}"
        if name and verb in ("on", "off"):
            return f"turn {verb} the {name}"
        if name and verb:
            return f"{verb} the {name}"
        if topic.get("note"):
            return str(topic["note"])
        if name:
            return str(name)
        return None

    @staticmethod
    def _followup_kind(text: str) -> Optional[str]:
        t = normalize(text).strip(" .!?…")
        for kind, phrases in FOLLOWUPS.items():
            if t in phrases:
                return kind
        return None

    @staticmethod
    def _refers_back(text: str) -> bool:
        """True only for a short message that points at the previous topic."""
        if INSTRUCTION_RE.match(text):
            return False
        if len(tokenize(text)) > ANAPHORA_MAX_WORDS:
            return False
        return ConversationManager._has_anaphora(text)

    @staticmethod
    def _has_anaphora(text: str) -> bool:
        toks = set(tokenize(text))
        if any(a in toks for a in ("it", "that", "this", "them", "those")):
            return True
        return any(a in text for a in ("the same", "same one"))

    @staticmethod
    def _has_content(text: str) -> bool:
        """True when the message carries its own subject (not just a nudge)."""
        toks = [t for t in tokenize(text) if t not in STOPWORDS]
        return len(toks) >= 2

    def _topic_slots(self, meta: Dict[str, Any],
                     text: str) -> Dict[str, Any]:
        """What the next turn is allowed to refer back to.

        Taken from the plan the engine actually built (never guessed): the
        device it acted on, the verb, the number, and the planner's own
        human phrasing of the step — which is what an anaphora rewrite uses.
        """
        plan = meta.get("plan") or {}
        steps = list(plan.get("steps") or []) + list(meta.get("actions") or [])
        device: Optional[str] = None
        device_name: Optional[str] = None
        verb: Optional[str] = None
        note: Optional[str] = None
        number: Optional[Any] = None
        for step in steps:
            params = step.get("params") or {}
            device = device or step.get("device_id")
            verb = verb or params.get("action") or params.get("verb")
            note = note or step.get("note")
            if number is None:
                number = params.get("level", params.get("value"))
        if device or device_name:
            device_name = self._device_name(device) or device_name
        elif note:
            # a file/command step: remember the phrasing, not a device
            device = None
        if not device_name:
            try:
                candidates = [(d["id"], f"{d['name']} {d.get('kind', '')}")
                              for d in self.agent._devices()]
            except Exception:  # noqa: BLE001 - devices are optional
                candidates = []
            if candidates:
                from .nlp import best_fuzzy_match
                device = best_fuzzy_match(text, candidates)
                device_name = self._device_name(device)
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if number is None and match:
            number = match.group(0)
        return {"device": device, "device_name": device_name, "verb": verb,
                "note": note, "number": number,
                "intent": meta.get("intent", "")}

    def _device_name(self, device_id: Optional[str]) -> Optional[str]:
        if not device_id:
            return None
        try:
            for device in self.agent._devices():
                if device.get("id") == device_id:
                    return device.get("name") or device_id
        except Exception:  # noqa: BLE001
            return None
        return None

    # ------------------------------------------------------- 4. recall

    def _recall(self, text: str) -> List[str]:
        """Real memories that overlap what was just said (may be empty)."""
        terms = [t for t in tokenize(text)
                 if t not in STOPWORDS and len(t) > 2][:8]
        if not terms:
            return []
        out: List[str] = []
        try:
            rows = self.agent.storage.search_memories(terms, limit=3)
        except Exception:  # noqa: BLE001
            return []
        for row in rows:
            if row.get("score", 0) < 0.3:
                continue
            value = row.get("value")
            note = None
            if isinstance(value, dict):
                note = value.get("note") or value.get("message") \
                    or value.get("text")
            elif isinstance(value, str):
                note = value
            if note:
                note = str(note).strip().replace("\n", " ")[:200]
                if note and normalize(note) not in normalize(text) \
                        and note not in out:
                    out.append(note)
        return out

    def recall_answer(self, text: str, limit: int = 8) -> List[Dict[str, Any]]:
        """For "what do you remember about X": the matching rows themselves."""
        terms = [t for t in tokenize(text)
                 if t not in STOPWORDS and len(t) > 2][:8]
        if not terms:
            return []
        try:
            return self.agent.storage.search_memories(terms, limit=limit)
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------- 5. voice

    def _voice(self, raw: str, intent: str, sess: Session,
               recalled: List[str], context: Dict[str, Any],
               is_owner: bool = False) -> str:
        """Turn pipeline output into something a person would say.

        ``is_owner`` matters: the engine writes its greetings for the owner
        ("Hello Boss", "You're welcome, boss"), which is right on Telegram
        and on an owner-scoped key, and plainly wrong when a stranger is
        using the public demo chat. The pipeline is left alone — /v1/chat
        keeps the engine's own words — and only the web voice adapts.
        """
        body = self._tidy(raw)
        lead = self._lead(intent, sess, context)

        if intent == "greet" and not is_owner:
            text = self._visitor_greeting(sess)
        elif intent == "thanks" and not is_owner:
            text = "You're welcome — glad to help." + self._tail(sess, body)
        elif intent == "status":
            text = f"{lead}Here is where I am right now:\n{body}"
        elif intent == "counters":
            text = f"{lead}{body}"
        elif intent in ("greet", "thanks"):
            text = f"{body}{self._tail(sess, body)}"
        elif intent == "learn":
            text = f"{body} That is in my memory now — ask me about it " \
                   f"later and I will recall it."
        elif intent == "goal":
            text = f"{body}"
        elif intent in ("approve", "deny"):
            text = f"{body}"
        elif "I'm not sure what you'd like me to do there" in body:
            # the fallback leads with whatever memory matched, so the generic
            # "I also remember…" tail below would only repeat it
            text = self._honest_fallback(body, sess, recalled)
            recalled = []
        else:
            text = f"{lead}{body}{self._tail(sess, body)}"

        if recalled:
            quoted = recalled[0]
            text += f"\n\nI also remember you told me: “{quoted}” — that is " \
                    f"still in my memory."
        return text.strip()

    def _lead(self, intent: str, sess: Session,
              context: Dict[str, Any]) -> str:
        """A short human opener; varied by mood and turn, never random."""
        if context.get("resolved") and context.get("kind") == "again":
            return "Sure — once more. "
        if context.get("resolved"):
            return "Following on from what we were just doing: "
        snap = self.agent.mind.snapshot()
        mood = snap["mood"]["note"]
        pool = {
            "energized": ("Right — ", "On it. ", ""),
            "content": ("", "Okay. ", ""),
            "alert": ("", "Noted. ", ""),
            "steady": ("", "", "Fine — "),
            "low": ("", "Slowly, but yes. ", ""),
            "troubled": ("", "Carefully, then. ", ""),
        }.get(mood, ("", "", ""))
        opener = pool[sess.turn_count % len(pool)]
        if intent == "counters" and not opener:
            opener = "Here are the numbers. "
        return opener

    def _visitor_greeting(self, sess: Session) -> str:
        """First hello for somebody using the public site (not the owner).

        It says what it is, what it is not, and gives one thing to try — the
        three pieces of information a stranger actually needs.
        """
        snap = self.agent.mind.snapshot()
        name = snap["agent"]
        mood = snap["mood"]["note"]
        gen = getattr(self.agent, "generative", None)
        if gen is not None and gen.can_talk:
            if sess.turn_count == 0:
                extras = ["chat about anything"]
                if gen.can_draw:
                    extras.append("draw images (\"draw me a …\")")
                extras.append("build presentations (\"make a presentation about …\")")
                if gen.can_speak:
                    extras.append("read replies aloud")
                return (f"Hi, I'm {name}. I can {', '.join(extras[:-1])} and "
                        f"{extras[-1]}. I also remember what you tell me "
                        f"(\"remember that …\"). What would you like to do?")
            return f"Hello again — {name} here. What's next?"
        if sess.turn_count == 0:
            return (
                f"Hi — I'm {name}. I'm a cognitive AI my owner wrote from "
                f"scratch in pure Python: I have drives, a mood (currently "
                f"{mood}), goals, memory and my own thinking loop, and "
                f"every single thing I do is counted in a ledger.\n\n"
                f"I run entirely on this machine — no external AI, no "
                f"internet, nothing of yours leaves this page. Try "
                f"`status` to see inside me, `counters` for the ledger, or "
                f"`remember that <anything>` and I will recall it later.")
        return (f"Hello again — {name} here, still local, still counting "
                f"everything. Mood is {mood}. Ask me `status` any time.")

    def _tail(self, sess: Session, body: str = "") -> str:
        """Keep the conversation moving — one honest invitation, not filler."""
        said = body.lower()
        if sess.turn_count <= 1:
            if "status" in said:
                return ""
            return " Ask me for `status` if you want to see my state."
        if sess.turn_count == 2:
            return " You can also teach me: `teach: \"<sentence>\" means " \
                   "<intent>`."
        if sess.turn_count % 4 == 0:
            return " Say `counters` any time — I count everything."
        return ""

    def _honest_fallback(self, body: str, sess: Session,
                         recalled: List[str]) -> str:
        """When the parser has no idea: admit it, then be useful."""
        topic = sess.topic()
        if recalled:
            # an unparsed message that still matches a memory is answerable:
            # say what it remembers instead of only admitting confusion
            lines = [f"I could not map that onto an action, but my memory "
                     f"does have something on it: you told me "
                     f"“{recalled[0]}”."]
        else:
            lines = ["I did not follow that, and I would rather say so than "
                     "guess."]
        subject = topic.get("device_name") or topic.get("device")
        if subject:
            lines.append(f"We were talking about {subject} — if that is still "
                         f"the subject, say \"turn it on\" or \"turn it "
                         f"off\".")
        lines.append(
            "My understanding is a local trainable model, so it only knows "
            "the phrasings it has been taught. These all work today: "
            "`status`, `counters`, `turn on the <device>`, "
            "`run: <command>`, `remember that <fact>`, "
            "`goal: <something to work on>`, "
            "`teach: \"<sentence>\" means <intent>`.")
        return " ".join(lines)

    @staticmethod
    def _tidy(text: str) -> str:
        """Normalise engine output for a chat bubble (keeps the structure)."""
        out = re.sub(r"[ \t]{2,}", " ", (text or "").strip())
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out



