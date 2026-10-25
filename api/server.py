"""HTTP server: ``POST /generate`` streams tokens back with Server-Sent Events.

    uvicorn api.server:app --host 0.0.0.0 --port 8000

How it fits together:

* One background thread owns the :class:`LLMEngine` and loops ``engine.step()``.
  GPU work is blocking, so it must not run on the asyncio event loop that
  serves HTTP; and keeping every engine call on one thread means the engine
  needs no locks.
* Request handlers never touch the engine. They put a submission into a
  thread-safe inbox and then read tokens from their own ``asyncio.Queue``.
* After every step the engine thread pushes each new token to the right
  queue with ``loop.call_soon_threadsafe`` (the safe way to hand data from a
  plain thread to asyncio).
* If a client disconnects, its handler sends an abort; the engine thread
  cancels the request and its cache slot is freed for someone else.

Settings come from environment variables (``MIE_MODEL``, ``MIE_MAX_NUM_SEQS``,
``MIE_MAX_MODEL_LEN``, ``MIE_DEVICE``, ``MIE_DTYPE``).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(128, ge=1)
    temperature: float = Field(0.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    ignore_eos: bool = False
    chat: bool = Field(True, description="wrap the prompt in the model's chat template")
    stream: bool = True


@dataclass
class _Submission:
    prompt_ids: list[int]
    params: SamplingParams
    events: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    request_id: int | None = None  # filled in by the engine thread


class EngineWorker:
    """Runs the engine on a background thread and routes its tokens to waiting handlers."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()  # submissions and aborts from handlers
        self._active: dict[int, _Submission] = {}  # request id -> who is listening
        self._texts: dict[int, str] = {}  # request id -> text already sent
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._inbox.put(None)  # wake the thread if it is waiting
        self._thread.join(timeout=10)

    def submit(self, sub: _Submission) -> None:
        self._inbox.put(("add", sub))

    def abort(self, sub: _Submission) -> None:
        self._inbox.put(("abort", sub))

    def stats(self) -> dict:
        sched = self.engine.scheduler
        return {"running": len(sched.running), "waiting": len(sched.waiting),
                "free_slots": self.engine.cache.num_free_slots, "steps": len(self.engine.step_stats)}

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        while not self._stop.is_set():
            # Block for new work only when idle; otherwise just drain the inbox.
            self._drain_inbox(block=not self.engine.has_unfinished_requests())
            if self._stop.is_set():
                break
            if self.engine.has_unfinished_requests():
                new_tokens = self.engine.step()
                self._dispatch(new_tokens)

    def _drain_inbox(self, block: bool) -> None:
        try:
            item = self._inbox.get(timeout=0.5) if block else self._inbox.get_nowait()
        except queue.Empty:
            return
        while item is not None:
            kind, sub = item
            if kind == "add":
                try:
                    sub.request_id = self.engine.add_request(sub.prompt_ids, sub.params)
                    self._active[sub.request_id] = sub
                    self._texts[sub.request_id] = ""
                except ValueError as e:  # e.g. prompt longer than max_model_len
                    self._send(sub, {"error": str(e), "finished": True})
            elif kind == "abort" and sub.request_id in self._active:
                self.engine.abort_request(sub.request_id)
                self._close(sub.request_id)
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                item = None

    def _dispatch(self, new_tokens: list[tuple[int, int]]) -> None:
        for request_id, token in new_tokens:
            sub = self._active.get(request_id)
            if sub is None:
                continue
            seq = self.engine.sequences[request_id]
            event = {"token": token, "text": self._text_delta(request_id)}
            if seq.is_finished:
                event.update(finished=True, finish_reason=seq.finish_reason.value,
                             usage={"prompt_tokens": seq.prompt_len, "completion_tokens": len(seq.output_token_ids)},
                             ttft_s=round(seq.first_token_time - seq.arrival_time, 4))
            self._send(sub, event)
            if seq.is_finished:
                self._close(request_id)

    def _text_delta(self, request_id: int) -> str:
        """New text since the last event.

        We decode the whole output each time and send what was added. A token
        can be part of a multi-byte character; until the character is
        complete the decoder shows a replacement mark, so we hold back.
        """
        text = self.engine.tokenizer.decode(self.engine.sequences[request_id].output_token_ids,
                                            skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        delta = text[len(self._texts[request_id]):]
        self._texts[request_id] = text
        return delta

    def _send(self, sub: _Submission, event: dict) -> None:
        sub.loop.call_soon_threadsafe(sub.events.put_nowait, event)

    def _close(self, request_id: int) -> None:
        self._active.pop(request_id, None)
        self._texts.pop(request_id, None)
        self.engine.release(request_id)


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Build the app. ``config`` defaults to one read from ``MIE_*`` environment variables."""
    config = config or EngineConfig(
        model=os.environ.get("MIE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_num_seqs=int(os.environ.get("MIE_MAX_NUM_SEQS", "32")),
        max_model_len=int(os.environ.get("MIE_MAX_MODEL_LEN", "2048")),
        device=os.environ.get("MIE_DEVICE", "cuda"),
        dtype=os.environ.get("MIE_DTYPE", "float16"),
    )
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = EngineWorker(LLMEngine(config))
        worker.start()
        state["worker"] = worker
        yield
        worker.stop()

    app = FastAPI(title="Mini LLM Inference Engine", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", **state["worker"].stats()}

    @app.post("/generate")
    async def generate(req: GenerateRequest, http_request: Request):
        worker: EngineWorker = state["worker"]
        tok = worker.engine.tokenizer
        text = (tok.apply_chat_template([{"role": "user", "content": req.prompt}], tokenize=False,
                                        add_generation_prompt=True) if req.chat else req.prompt)
        try:
            params = SamplingParams(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                    max_tokens=req.max_tokens, ignore_eos=req.ignore_eos)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        sub = _Submission(tok(text).input_ids, params, asyncio.Queue(), asyncio.get_running_loop())
        worker.submit(sub)

        async def events() -> AsyncIterator[dict]:
            try:
                while True:
                    event = await sub.events.get()
                    yield event
                    if event.get("finished"):
                        return
            finally:
                # Runs on normal completion and when the client goes away
                # (the generator is closed); aborting a finished request is a no-op.
                worker.abort(sub)

        if not req.stream:
            parts, last = [], {}
            async for event in events():
                parts.append(event.get("text", ""))
                last = event
            if "error" in last:
                raise HTTPException(status_code=400, detail=last["error"])
            return {"text": "".join(parts), **{k: last[k] for k in ("finish_reason", "usage", "ttft_s") if k in last}}

        async def sse() -> AsyncIterator[str]:
            # Server-Sent Events: each message is "data: <json>" followed by a blank line.
            async for event in events():
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


# Creating the app is cheap: the engine (and model) load when the server starts.
app = create_app()
