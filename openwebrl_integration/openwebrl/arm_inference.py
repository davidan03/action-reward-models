"""Frozen action selection used only by the standalone ARM evaluation runner."""
import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace

CANONICAL_FLAGS = {
    "CATTS_VISION_PROMPT_V2": "1", "CATTS_VISION_COLORED": "1",
    "VISION_NO_SOM": "1", "VISION_ABLATE_DOM": "1", "VISION_ABLATE_VOTES": "1",
    "NORMALIZE_COORDS": "1", "CLUSTER_NO_DOM": "1", "VISION_NO_COT": "1",
    "VISION_NO_COT_THINK": "0", "VISION_ABLATE_IMAGE": "0",
    "VISION_ABLATE_CAND_THOUGHTS": "0", "VISION_ABLATE_HISTORY": "0",
    "VISION_ABLATE_TRAJ_THOUGHTS": "0", "VISION_FULL_HISTORY": "0",
}
SCALAR_SYSTEM = (
    "You are an expert web-agent action evaluator. Given the task, the current "
    "page state and screenshot, and ONE proposed next action, judge how well "
    "the action advances the task."
)


def split_response(response):
    """Keep actor-space tool JSON and coordinates unchanged."""
    if "</think>" in response:
        thought, action = response.split("</think>", 1)
        thought = thought.rsplit("<think>", 1)[-1].strip()
    else:
        thought, action = "", response
    return {"thought": thought, "action": action.strip()}


def selection_schema(count):
    """JSON schema used by the released canonical VISION_NO_COT call."""
    return {"type": "object", "properties": {
        "selection": {"type": "integer", "minimum": 1, "maximum": count}},
        "required": ["selection"]}


def parse_selection(text, count):
    """A malformed selector reply is unavailable, never a fabricated winner."""
    import re
    matches = list(re.finditer(r'\{\s*"selection"\s*:\s*(\d+)\s*\}', text))
    if not matches:
        raise ValueError("ARM response has no selection JSON")
    index = int(matches[-1].group(1)) - 1
    if not 0 <= index < count:
        raise ValueError("ARM selection is out of range")
    return index


def scalar_messages(task, url, history, candidate):
    """Match the OpenWebRL scalar training builder, including 800/400 truncation."""
    hist = "\n".join(f"  {i+1}. {h['action']}" for i, h in enumerate(history[-8:])) or "  (none)"
    user = (f"Task: {task}\nCurrent URL: {url}\nRecent actions:\n{hist}\n"
            "Current page screenshot: ")
    return [
        {"role": "system", "content": SCALAR_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": user}, {"type": "image"},
            {"type": "text", "text": "\n\nProposed action:\n(see candidate below)"}]},
        {"role": "assistant", "content":
            f"Action: {candidate['action'].strip()[:800]}\nReasoning: {candidate['thought'].strip()[:400]}"},
    ]


def load_selection_builder(source_root):
    path = Path(source_root) / "inference/selection_prompt.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "043ab7e98127f07ccd76b0eb736837f8a089c1328fa211c4ca4c20d771a7751c":
        raise ValueError("Selection prompt source does not match pinned release")
    spec = importlib.util.spec_from_file_location("arm_reference_selection_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.update(CANONICAL_FLAGS)
    return module.build_catts_vision_prompt_v2


def selection_messages(builder, task, url, history, candidates, screenshot):
    # Keep JSON point_2d actions untouched. The reference builder rescales textual
    # click(x,y) syntax as raw 1280x720 pixels; our actor emits normalized JSON.
    clusters = [{"rep": SimpleNamespace(molmo_action=c["action"], thought=c["thought"]),
                 "vote_count": 1, "cluster_key": str(i)} for i, c in enumerate(candidates)]
    return builder(task, history, url, clusters, screenshot)


def candidate_seed(seed, task_id, turn, candidate):
    key = f"{seed}:{task_id}:{turn}:{candidate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31 - 1)


async def request_selection_result(endpoint, payload, timeout, connect_timeout=None):
    """Retry only connections that failed before submitting a selector request."""
    import httpx
    async def request():
        limits = timeout if connect_timeout is None else httpx.Timeout(timeout, connect=min(timeout, connect_timeout))
        async with httpx.AsyncClient(timeout=limits, trust_env=False) as client:
            response = await client.post(endpoint + '/select', json=payload)
            response.raise_for_status()
            return response.json()
    if connect_timeout is None:
        return await request()
    async with asyncio.timeout(timeout):
        for attempt in range(3):
            try:
                return await request()
            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                if attempt == 2: raise
                logging.getLogger(__name__).warning(
                    '[selector transport] %s connecting to %s; retry %d/2',
                    type(exc).__name__, endpoint, attempt + 1)
                await asyncio.sleep(.2)


class ActionSelector:
    """Sample five iid proposals at one live state, score, execute one unchanged."""
    def __init__(self, mode, endpoint, output, seed=42, candidates=5, timeout=180, exporter=None, connect_timeout=None):
        if mode not in ("baseline", "selection", "scalar"):
            raise ValueError(mode)
        self.mode, self.endpoint = mode, endpoint.rstrip("/")
        self.output, self.seed = Path(output), seed
        self.candidates = 1 if mode == "baseline" else candidates
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.exporter = exporter
        self.output.mkdir(parents=True, exist_ok=True)

    async def __call__(self, *, infer, url, input_text, sampling_params, images,
                       observation, history, task, task_id, turn, timeout):
        import httpx
        calls = []
        for index in range(self.candidates):
            params = dict(sampling_params, sampling_seed=candidate_seed(self.seed, task_id, turn, index))
            calls.append(infer(url, input_text, params, images, timeout_secs=timeout))
        outputs = await asyncio.gather(*calls)
        candidates = [split_response(o[0]) for o in outputs]
        index, scores, raw, fallback = 0, None, None, None
        if self.mode != "baseline":
            payload = {
                "mode": self.mode, "task": task,
                "url": observation.get("active_tab_url", ""),
                "history": [split_response(r) for r in history],
                "candidates": candidates,
                "screenshot": base64.b64encode(observation["screenshot"]).decode(),
            }
            result = await request_selection_result(self.endpoint, payload, self.timeout, self.connect_timeout)
            scores, raw = result.get("scores"), result.get("raw")
            if self.mode == "scalar":
                if len(scores or []) != len(candidates) or not all(math.isfinite(s) for s in scores):
                    raise ValueError("Invalid scalar score vector")
                index = max(range(len(scores)), key=lambda j: scores[j])
            else:
                # Match the released inference demo's first-candidate fallback,
                # but retain its frequency separately in evaluation traces.
                try:
                    index = parse_selection(raw or "", len(candidates))
                except ValueError as exc:
                    fallback = str(exc)
                    index = 0
        record = {"mode": self.mode, "task_id": task_id, "turn": turn,
                  "selected_index": index, "scores": scores, "selector_raw": raw,
                  "fallback": fallback, "candidates": candidates,
                  "candidate_seeds": [candidate_seed(self.seed, task_id, turn, j)
                                      for j in range(self.candidates)],
                  "finish_types": [o[3] for o in outputs],
                  "url": observation.get("active_tab_url", ""),
                  "prompt_sha256": hashlib.sha256(input_text.encode()).hexdigest(),
                  "screenshot_sha256": hashlib.sha256(observation["screenshot"]).hexdigest()}
        path = self.output / (hashlib.sha256(str(task_id).encode()).hexdigest()[:20] + ".jsonl")
        if self.exporter is not None:
            from openwebrl.artifact_io import run_artifact_io
            await run_artifact_io(self._persist, path, record, input_text, images, outputs)
        else:
            self._persist(path, record, input_text, images, outputs)
        return outputs[index], {"selected_index": index, "mode": self.mode,
                                "fallback": fallback, "trace_path": str(path)}

    def _persist(self, path, record, input_text, images, outputs):
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.exporter is not None:
            self.exporter(record=record, prompt=input_text, images=images, outputs=outputs)
