"""ReAct execution loop (Think ➔ Act ➔ Observe) — pure stdlib, no frameworks.

This module is the hand-written "agent runtime" pattern: an explicit loop
that asks a language model for a THOUGHT + one JSON ACTION, runs that
action through a small tool registry, feeds the OBSERVATION back into the
message memory, and stops when the model calls ``final_answer`` (or the
iteration budget runs out).

Design rules (same as the rest of RealAI):

- **No third-party packages.** HTTP goes through ``urllib``; there is no
  pydantic / httpx / LangChain.
- **No ``eval``.** ``calculate_expression`` walks an ``ast`` tree and
  allows only numeric literals and arithmetic operators.
- **No mock tools.** ``read_local_file`` really reads, but only inside the
  owner-configured ``Config.file_roots`` (same guard as ``files.read``).
- **Backend-agnostic.** The model is any ``Callable[[messages], str]``.
  ``OllamaBackend`` talks to a *local* Ollama server; tests inject a
  scripted callable so nothing here needs a running model.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import Config

Messages = List[Dict[str, str]]
LLMBackend = Callable[[Messages], str]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"

AGENT_SYSTEM_PROMPT = """You are an autonomous senior-level execution agent. You solve problems by executing iterative cycles of thought, choosing an action, and reading the observation.

Your responses MUST follow this exact format and contain nothing else:

THOUGHT: Your current reasoning step.
ACTION:
```json
{"name": "tool_name", "arguments": {"arg1": "value"}}
```

Available tools:
1. "calculate_expression": Evaluates an arithmetic expression (+ - * / ** % // and parentheses).
   - arguments: {"expression": "string"}
2. "read_local_file": Reads a text file from the agent's allowed data directory.
   - arguments: {"file_path": "string"}
3. "final_answer": Stop and return the final answer to the user.
   - arguments: {"output": "string"}

CRITICAL: Take exactly ONE action per response. After the ACTION block, stop and wait for the OBSERVATION.
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POW = 10_000


def safe_calculate(expression: str) -> float:
    """Evaluate arithmetic via the AST — no names, calls, or attributes."""
    expression = expression.strip()
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")

    return walk(tree)


class ToolRegistry:
    """Named tools the loop may call. Every tool returns a plain string."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._tools: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "calculate_expression": self._calculate,
            "read_local_file": self._read_file,
        }

    def register(self, name: str, fn: Callable[[Dict[str, Any]], str]) -> None:
        self._tools[name] = fn

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return (f"Error: tool '{name}' is not registered. "
                    f"Available: {', '.join(self.names)}, final_answer")
        try:
            return fn(arguments or {})
        except Exception as exc:  # noqa: BLE001 - surface to the model
            return f"Execution Error: {exc}"

    def _calculate(self, args: Dict[str, Any]) -> str:
        expr = str(args.get("expression", ""))
        try:
            result = safe_calculate(expr)
        except ZeroDivisionError:
            return "Execution Error: division by zero"
        except (ValueError, SyntaxError) as exc:
            return f"Execution Error: {exc}"
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"Success: Result is {result}"

    def _read_file(self, args: Dict[str, Any]) -> str:
        from .actions.builtins import ScopeError, files_read
        try:
            ok, out, err = files_read(self.cfg, {"path": str(args.get("file_path", ""))})
        except ScopeError as exc:
            return f"File System Error: {exc}"
        return out if ok else f"File System Error: {err}"


# ---------------------------------------------------------------------------
# Model backend (local Ollama over urllib) — optional
# ---------------------------------------------------------------------------

class BackendError(RuntimeError):
    """The model backend could not produce a response."""


class OllamaBackend:
    """Minimal ``/api/chat`` client for a *local* Ollama server."""

    def __init__(self, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def __call__(self, messages: Messages) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            return str(data.get("message", {}).get("content", ""))
        except urllib.error.HTTPError as exc:
            raise BackendError(f"inference node returned status {exc.code}") from exc
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"connection to LLM backend at {self.base_url} "
                               f"failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

_ACTION_FENCED = re.compile(r"ACTION:\s*```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_ACTION_INLINE = re.compile(r"ACTION:\s*(\{.*\})", re.DOTALL | re.IGNORECASE)
_THOUGHT = re.compile(r"THOUGHT:\s*(.*?)(?:\n\s*ACTION:|\Z)", re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON action from a model response; ``None`` if invalid."""
    for pattern in (_ACTION_FENCED, _ACTION_INLINE):
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1)
        # inline fallback may over-capture; try shrinking to balanced braces
        for candidate in _brace_candidates(raw):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                obj.setdefault("arguments", {})
                if not isinstance(obj["arguments"], dict):
                    obj["arguments"] = {}
                return obj
    return None


def _brace_candidates(raw: str):
    yield raw
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield raw[: i + 1]
                return


def parse_thought(text: str) -> str:
    m = _THOUGHT.search(text)
    return m.group(1).strip() if m else ""


@dataclass
class Step:
    iteration: int
    thought: str
    action: Optional[Dict[str, Any]]
    observation: str
    raw: str
    ms: float


@dataclass
class RunResult:
    output: str
    completed: bool
    iterations: int
    steps: List[Step] = field(default_factory=list)
    messages: Messages = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "completed": self.completed,
            "iterations": self.iterations,
            "steps": [
                {"iteration": s.iteration, "thought": s.thought,
                 "action": s.action, "observation": s.observation,
                 "ms": round(s.ms, 2)}
                for s in self.steps
            ],
        }


class ReActAgent:
    """Explicit Think ➔ Act ➔ Observe loop with injectable model backend."""

    def __init__(self, backend: LLMBackend, tools: ToolRegistry,
                 system_prompt: str = AGENT_SYSTEM_PROMPT,
                 max_iterations: int = 5,
                 on_step: Optional[Callable[[Step], None]] = None) -> None:
        self.backend = backend
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max(1, int(max_iterations))
        self.on_step = on_step

    def run(self, task: str) -> RunResult:
        messages: Messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]
        steps: List[Step] = []

        for i in range(1, self.max_iterations + 1):
            t0 = time.perf_counter()
            try:
                raw = self.backend(messages)
            except BackendError as exc:
                return RunResult(f"Agent aborted: {exc}", False, i - 1, steps, messages)
            messages.append({"role": "assistant", "content": raw})

            action = parse_action(raw)
            thought = parse_thought(raw)

            if action is None:
                observation = ("Error: your ACTION section was missing or not "
                               "valid JSON. Reply with THOUGHT and exactly one "
                               "ACTION ```json``` block.")
                step = Step(i, thought, None, observation, raw,
                            (time.perf_counter() - t0) * 1000)
                steps.append(step)
                self._emit(step)
                messages.append({"role": "user", "content": observation})
                continue

            name = action["name"]
            args = action["arguments"]

            if name == "final_answer":
                output = str(args.get("output", "")).strip() or \
                    "Task complete, no specific output captured."
                step = Step(i, thought, action, output, raw,
                            (time.perf_counter() - t0) * 1000)
                steps.append(step)
                self._emit(step)
                return RunResult(output, True, i, steps, messages)

            observation = self.tools.execute(name, args)
            step = Step(i, thought, action, observation, raw,
                        (time.perf_counter() - t0) * 1000)
            steps.append(step)
            self._emit(step)
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        return RunResult(
            "Agent aborted: maximum iterations reached before a final answer.",
            False, self.max_iterations, steps, messages)

    def _emit(self, step: Step) -> None:
        if self.on_step:
            try:
                self.on_step(step)
            except Exception:  # noqa: BLE001 - callbacks never break the loop
                pass


# ---------------------------------------------------------------------------
# Arena.ai-style handler
# ---------------------------------------------------------------------------

AGENT_NAME = "RealAI-ReAct-v1"


def arena_agent_handler(payload: Dict[str, Any],
                        backend: Optional[LLMBackend] = None,
                        cfg: Optional[Config] = None,
                        on_step: Optional[Callable[[Step], None]] = None) -> Dict[str, Any]:
    """Entry point for an evaluation runner.

    ``payload`` keys: ``prompt`` (required), ``max_steps`` (default 5),
    ``model`` (Ollama model name), ``ollama_url``.
    """
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return {"status": "error", "agent_name": AGENT_NAME,
                "output": "", "error": "payload.prompt is required"}
    try:
        max_steps = int(payload.get("max_steps", 5))
    except (TypeError, ValueError):
        max_steps = 5

    cfg = cfg or Config.from_env()
    backend = backend or OllamaBackend(
        model=str(payload.get("model", DEFAULT_MODEL)),
        base_url=str(payload.get("ollama_url", DEFAULT_OLLAMA_URL)))

    agent = ReActAgent(backend, ToolRegistry(cfg), max_iterations=max_steps,
                       on_step=on_step)
    result = agent.run(prompt)
    return {
        "status": "success" if result.completed else "incomplete",
        "agent_name": AGENT_NAME,
        "output": result.output,
        "iterations": result.iterations,
        "trace": result.to_dict()["steps"],
    }
