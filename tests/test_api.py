"""The streaming HTTP server, end to end on a real (small-config) engine."""

import json
import threading

import pytest
from conftest import DEVICE
from fastapi.testclient import TestClient

from api.server import create_app
from engine.config import EngineConfig


@pytest.fixture(scope="module")
def client():
    cfg = EngineConfig(device=DEVICE, dtype="float32" if DEVICE == "cpu" else "float16",
                       max_num_seqs=2, max_model_len=256, max_num_batched_tokens=512)
    with TestClient(create_app(cfg)) as c:  # the context manager runs startup/shutdown
        yield c


def read_sse(response) -> list[dict]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


def test_stream_returns_one_event_per_token(client):
    with client.stream("POST", "/generate", json={"prompt": "Count to five.", "max_tokens": 6, "ignore_eos": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = read_sse(r)
    assert len(events) == 6
    assert events[-1]["finished"] and events[-1]["finish_reason"] == "length"
    assert events[-1]["usage"]["completion_tokens"] == 6
    assert "".join(e["text"] for e in events).strip()


def test_non_streaming_matches_streaming(client):
    body = {"prompt": "Name a colour.", "max_tokens": 8}
    full = client.post("/generate", json={**body, "stream": False}).json()
    with client.stream("POST", "/generate", json=body) as r:
        streamed = "".join(e["text"] for e in read_sse(r))
    assert full["text"] == streamed  # greedy: same text either way


def test_concurrent_requests_share_the_engine(client):
    results = {}

    def call(i):
        results[i] = client.post("/generate", json={"prompt": f"Say the number {i}.", "max_tokens": 5,
                                                    "ignore_eos": True, "stream": False}).json()

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]  # 4 requests, only 2 slots
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results[i]["usage"]["completion_tokens"] == 5 for i in range(4))
    assert client.get("/health").json()["free_slots"] == 2  # every slot handed back


def test_prompt_too_long_is_an_error(client):
    r = client.post("/generate", json={"prompt": "word " * 400, "chat": False, "stream": False})
    assert r.status_code == 400 and "max_model_len" in r.json()["detail"]
