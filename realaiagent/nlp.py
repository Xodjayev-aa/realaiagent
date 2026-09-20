"""Local, trainable natural-language layer.

No external models. A linear one-vs-rest intent classifier (SGD-trained on
text features, persisted in the database) plus deterministic slot
extraction (verbs, values, devices, file paths, numbers). The owner can
keep teaching the agent through :class:`IntentModel.learn` - every example
reweights the model, and the model state is saved/loaded with the agent.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_PATH_RE = re.compile(r"(?:/|~|\.\.?/)[\w./-]+|\b[\w-]+\.(?:txt|md|json|csv|log|py|ya?ml|toml|sh|html?|css|js)\b")

# Verbs the agent can act on, mapped to canonical action verbs.
VERB_MAP: Dict[str, List[str]] = {
    "on": ["turn on", "switch on", "power on", "light up", "wake up", "enable", "activate", "start", "boot up", "switch to on"],
    "off": ["turn off", "switch off", "power off", "shut down", "disable", "deactivate", "stop", "kill", "switch to off"],
    "set": ["set", "set to", "make it", "adjust", "change to", "bring to", "set the"],
    "read": ["open", "read", "show", "check", "look at", "display", "print", "display me"],
    "run": ["run", "execute", "perform", "do"],
    "write": ["write", "save", "create", "add", "store", "write to"],
    "delete": ["delete", "remove", "erase", "clear", "drop"],
}

# Seed corpus: enough for the agent to work on day one. The owner's
# teaching (via /v1/learn or the chat "teach:" syntax) refines it.
SEED_INTENTS: Dict[str, List[str]] = {
    "greet": [
        "hi", "hello", "hey", "good morning", "good evening", "good night",
        "howdy", "hi there", "hello there", "morning",
    ],
    "thanks": [
        "thanks", "thank you", "thanks a lot", "thank you so much",
        "appreciate it", "cheers", "thx",
    ],
    "status": [
        "status", "how are you", "how do you feel", "what can you do",
        "help", "who are you", "what is your mood", "report your state",
        "show me your state", "what are your goals", "list your goals",
    ],
    "counters": [
        "how many requests", "show the counts", "counters", "usage report",
        "statistics", "stats", "what have you counted", "how much have you done",
        "total usage", "give me the numbers",
    ],
    "goal": [
        "your goal is", "set your goal", "make sure you", "remember to",
        "from now on", "your task is", "add a goal", "goal:", "new goal",
    ],
    "learn": [
        "learn this", "train yourself", "teach you", "remember that",
        "the rule is", "note this", "teach:", "learn:",
    ],
    "approve": [
        "approve", "approved", "approve it", "yes do it", "allow it",
        "go ahead", "granted",
    ],
    "deny": [
        "deny", "denied", "deny it", "no do not", "reject", "cancel that",
        "not allowed",
    ],
    "command": [
        "turn on the lamp", "turn off the light", "switch on the heater",
        "set the temperature to 21", "run the backup script",
        "open the notes file", "check the log", "delete the temp file",
        "start the server", "stop the fan", "write the report file",
    ],
    "chat": [
        "what do you think", "tell me something", "how do you feel about",
        "what are you thinking", "talk to me", "what is on your mind",
        "do you have a mind of your own", "are you alive",
    ],
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class IntentModel:
    """SGD linear intent classifier with persistent weights."""

    def __init__(self) -> None:
        self.weights: Dict[str, Dict[str, float]] = {}
        self.priors: Counter = Counter()
        self.lr = 0.45

    # ------------------------------------------------------------ structure

    @staticmethod
    def features(text: str) -> List[str]:
        tokens = tokenize(text)
        feats = [f"w:{t}" for t in tokens]
        feats += [f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return feats

    # ------------------------------------------------------------ learning

    def learn(self, text: str, intent: str, lr: Optional[float] = None) -> None:
        feats = self.features(text)
        if not feats:
            return
        rate = lr if lr is not None else self.lr
        self.priors[intent] += 1
        pos = self.weights.setdefault(intent, {})
        for f in set(feats):
            pos[f] = pos.get(f, 0.0) + rate
        for other, vec in self.weights.items():
            if other == intent:
                continue
            for f in set(feats):
                vec[f] = vec.get(f, 0.0) - rate * 0.25

    def seed(self) -> None:
        for intent, examples in SEED_INTENTS.items():
            for ex in examples:
                self.learn(ex, intent, lr=0.3)

    # ------------------------------------------------------------ predict

    def predict(self, text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return (intent, confidence, top-k (intent, score) list)."""
        if not self.priors:
            return "chat", 0.0, []
        feats = set(self.features(text))
        scores: Dict[str, float] = {}
        for intent in self.priors:
            vec = self.weights.get(intent, {})
            s = 0.35 * math.log1p(self.priors[intent])
            for f in feats:
                s += vec.get(f, 0.0)
            scores[intent] = s
        if not scores:
            return "chat", 0.0, []
        # numerically-stable softmax (scores can be strongly negative)
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        total = sum(exps.values())
        probs = {k: v / total for k, v in exps.items()}
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return ranked[0][0], ranked[0][1], ranked[:3]

    # ------------------------------------------------------------ persist

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "priors": dict(self.priors)})

    @classmethod
    def from_json(cls, raw: str) -> "IntentModel":
        model = cls()
        try:
            data = json.loads(raw)
            model.weights = {k: dict(v) for k, v in data.get("weights", {}).items()}
            model.priors = Counter(data.get("priors", {}))
        except (ValueError, TypeError):
            model.seed()
        return model


# --------------------------------------------------------------------- slots

def extract_verb(text: str) -> Optional[str]:
    """Return the canonical action verb present in the text (or None)."""
    t = normalize(text)
    best: Optional[Tuple[int, str]] = None
    for verb, phrases in VERB_MAP.items():
        for ph in phrases:
            # require phrase boundaries so "on" does not fire in "season"
            if re.search(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", t):
                if best is None or len(ph) > len(best[0]):
                    best = (ph, verb)
    return best[1] if best else None


def extract_number(text: str) -> Optional[float]:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_quoted(text: str) -> Optional[str]:
    m = _QUOTED_RE.search(text)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)


def extract_path(text: str) -> Optional[str]:
    m = _PATH_RE.search(text)
    return m.group(0) if m else None


def best_fuzzy_match(
    text: str,
    candidates: Sequence[Tuple[str, str]],
    min_ratio: float = 0.6,
) -> Optional[str]:
    """Fuzzy-match free text against (id, display) candidates.

    Returns the winning id, or None. Handles "the living room lamp" style
    references by checking token overlap and sequence ratio.
    """
    t = normalize(text)
    if not t:
        return None
    best_id: Optional[str] = None
    best_score = min_ratio
    for cid, display in candidates:
        d = normalize(display)
        if not d:
            continue
        if d in t or t in d:
            score = 0.95
        else:
            score = difflib.SequenceMatcher(None, d, t).ratio()
            dtoks = set(d.split())
            ttoks = set(t.split())
            overlap = len(dtoks & ttoks) / max(1, min(len(dtoks), len(ttoks)))
            score = max(score, overlap * 0.9)
            # a distinctive name token in the sentence ("the LAMP") is a
            # strong signal
            for tok in dtoks:
                if len(tok) > 2 and tok in ttoks:
                    score = max(score, 0.85)
                    break
        if score > best_score:
            best_id, best_score = cid, score
    return best_id


def parse_message(
    model: IntentModel,
    text: str,
    devices: List[Dict[str, Any]],
    skills: List[str],
) -> Dict[str, Any]:
    """Full parse: intent, confidence, and slots.

    Slots: verb, number, quoted, path, device_id, skill, approve_id,
    learn (for "teach: ... means ..." lines), goal (for "goal: ..." lines).
    """
    raw_intent, conf, topk = model.predict(text)
    slots: Dict[str, Any] = {
        "verb": extract_verb(text),
        "number": extract_number(text),
        "quoted": extract_quoted(text),
        "path": extract_path(text),
        "device_id": None,
        "skill": None,
        "approve_id": None,
        "learn": None,
        "goal": None,
    }

    if devices:
        slots["device_id"] = best_fuzzy_match(
            text, [(d["id"], f"{d['name']} {d.get('kind','')}") for d in devices])

    t_norm = normalize(text)
    skill_hits = []
    for s in skills:
        if len(s) <= 2:
            continue
        s_norm = s.replace("-", " ")
        if (s_norm in t_norm or s in t_norm or
                difflib.SequenceMatcher(None, s_norm, t_norm).ratio() > 0.8):
            skill_hits.append(s)
            break
    if skill_hits:
        slots["skill"] = skill_hits[0]

    # "remember that …" is always a memory instruction
    if t_norm.startswith("remember that") and raw_intent != "learn":
        raw_intent, conf = "learn", max(conf, 0.85)

    m = re.search(r"(?<!\d)(\d{1,6})(?!\d)", text)
    if raw_intent in ("approve", "deny") and m:
        slots["approve_id"] = int(m.group(1))
    if raw_intent in ("approve", "deny") and slots["approve_id"] is None:
        slots["approve_id"] = _latest_pending_hint(text)

    teach = _parse_teach(text)
    if teach:
        raw_intent, slots["learn"] = "learn", teach
        conf = max(conf, 0.9)

    goalm = re.match(r"^\s*(?:goal|new goal|set goal)\s*[:\-]\s*(.+)$",
                     normalize(text))
    if goalm:
        raw_intent, slots["goal"] = "goal", goalm.group(1).strip()
        conf = max(conf, 0.9)

    return {
        "text": text,
        "intent": raw_intent,
        "confidence": round(conf, 4),
        "top_k": [[i, round(s, 4)] for i, s in topk],
        "slots": slots,
    }


def _latest_pending_hint(text: str) -> Optional[int]:
    """'approve the last one' / 'approve it' -> -1 meaning 'latest pending'."""
    t = normalize(text)
    if any(w in t for w in ("last", "it", "that", "the request")):
        return -1
    return None


def _parse_teach(text: str) -> Optional[Dict[str, str]]:
    """Parse 'teach: "sentence" means intent' / 'learn: "s" -> intent'."""
    t = normalize(text)
    m = re.match(r"^(?:teach|learn|train)\s*[:\-]\s*(.+)$", t)
    if not m:
        return None
    rest = m.group(1)
    qm = re.match(r'"([^"]+)"\s*(?:means|->|→|is)\s*([a-z_]+)', rest)
    if not qm:
        return None
    return {"text": qm.group(1), "intent": qm.group(2).strip()}
