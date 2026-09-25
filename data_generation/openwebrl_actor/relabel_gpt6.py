#!/usr/bin/env python3
"""Relabel the published selection-ARM examples with GPT-6: same screens, same candidates, same prompt text.

Input: a ShareGPT file from build_selection_sft.py as published in PTeterwak/action-reward-models-data
(openwebrl_actor/selection_sft/owrl_selection_{train,val}.json + images/). Each example is sent to GPT-6 UNCHANGED —
its system prompt and its user text with the screenshot in place of '<image>' — exactly what the GPT-5.5 teacher saw
(explain in 2-4 sentences, then {"selection": N}). The reply's last {"selection": N} is the new label; the original
target is kept as the GPT-5.5 label for comparison.

Spend: each call reserves an upper bound in --ledger before it is sent and settles to the billed usage afterwards; a
call that would push the ledger total past --cap-usd is refused and the run stops. Resumable: indices already in --out
are skipped. --dry-run builds every request and makes no call.

Usage:
  python relabel_gpt6.py --examples .../owrl_selection_val.json --images-root .../selection_sft \
      --out labels_val.jsonl --ledger ledger.jsonl --cap-usd 15 --limit 100 [--dry-run]
"""
import argparse
import asyncio
import base64
import fcntl
import json
import os
import re
import time
import uuid
from pathlib import Path

MODEL = os.environ.get("GPT6_MODEL", "gpt-6-astra")
EFFORT = os.environ.get("GPT6_EFFORT", "medium")
MAX_OUTPUT = 2048
P_IN, P_CACHED, P_WRITE, P_OUT = 10.0, 1.0, 12.5, 50.0   # USD per 1M tokens (same table as OpenWebRL's gpt6_step_selector)
IMG_TOKENS_UPPER = 2500
SEL_RE = re.compile(r'\{\s*"selection"\s*:\s*(\d+)\s*\}')
CAND_RE = re.compile(r"^  (\d+)\. ", re.M)


def n_candidates(user_text):
    block = user_text.split("#### Candidate Actions ####", 1)[-1].split("#### Instructions ####", 1)[0]
    return len(CAND_RE.findall(block))


def build_request(example, images_root):
    system, user, target = (m["content"] for m in example["messages"])
    before, after = user.split("<image>") if user.count("<image>") == 1 else (None, None)
    if before is None:
        raise ValueError("expected exactly one <image> in the user text")
    image = Path(images_root) / example["images"][0]
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    content = [{"type": "input_text", "text": before},
               {"type": "input_image", "image_url": url, "detail": "auto"},
               {"type": "input_text", "text": after}]
    est = (len(system) + len(user)) // 3 + IMG_TOKENS_UPPER
    m = SEL_RE.findall(target)
    return ([{"role": "system", "content": system}, {"role": "user", "content": content}], est,
            int(m[-1]) if m else None, n_candidates(user))


class Ledger:
    def __init__(self, path, cap):
        self.path, self.cap, self.lock = Path(path), cap, asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.total = sum(json.loads(l)["usd_delta"] for l in open(self.path)) if self.path.exists() else 0.0

    def _append(self, entry):
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.total += entry["usd_delta"]

    async def reserve(self, est):
        upper = (est * P_WRITE + MAX_OUTPUT * P_OUT) / 1e6
        async with self.lock:
            if self.total + upper > self.cap:
                raise RuntimeError(f"cap reached: ledger {self.total:.2f} + upper {upper:.3f} > cap {self.cap:.2f}")
            token = uuid.uuid4().hex
            self._append({"event": "reserve", "id": token, "time": time.time(), "usd_delta": upper, "est_input_tokens": est})
        return token, upper

    def settle(self, reservation, usage, response_id):
        token, upper = reservation
        d = usage.get("input_tokens_details") or {}
        cached, writes = d.get("cached_tokens") or 0, d.get("cache_write_tokens") or 0
        actual = ((usage.get("input_tokens", 0) - cached - writes) * P_IN + cached * P_CACHED + writes * P_WRITE
                  + usage.get("output_tokens", 0) * P_OUT) / 1e6
        self._append({"event": "settle", "id": token, "time": time.time(), "usd_delta": actual - upper, "usage": usage,
                      "response_id": response_id})
        return actual


async def label_one(client, ledger, idx, example, args, out_lock, out_f):
    msgs, est, gpt55, ncand = build_request(example, args.images_root)
    rec = {"idx": idx, "source": args.examples, "image": example["images"][0], "n_candidates": ncand,
           "gpt55_selection": gpt55, "model": MODEL, "effort": EFFORT}
    if args.dry_run:
        rec.update(status="dry_run", est_input_tokens=est)
    else:
        res = await ledger.reserve(est)
        try:
            reply = await client.responses.create(model=MODEL, input=msgs, reasoning={"effort": EFFORT},
                                                  max_output_tokens=MAX_OUTPUT, store=False)
        except Exception as e:  # reservation stays counted as spent (conservative)
            rec.update(status="api_error", error=repr(e)[:300])
        else:
            raw = reply.model_dump(mode="json")
            usd = ledger.settle(res, raw.get("usage") or {}, raw.get("id"))
            text = reply.output_text or ""
            m = SEL_RE.findall(text)
            sel = int(m[-1]) if m else None
            ok = raw.get("status") == "completed" and sel is not None and 1 <= sel <= ncand
            rec.update(status="labelled" if ok else "parse_fail", gpt6_selection=sel if ok else None, text=text,
                       api_status=raw.get("status"), usage=raw.get("usage"), usd=round(usd, 6))
    async with out_lock:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
    return rec


async def main_async(args):
    examples = json.load(open(args.examples))
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["idx"] for l in open(args.out) if json.loads(l).get("status") in ("labelled", "dry_run")}
    todo = [i for i in range(len(examples)) if i not in done][: args.limit or None]
    client = None
    if not args.dry_run:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=240, max_retries=2)
    ledger = Ledger(args.ledger, args.cap_usd)
    print(f"{len(examples)} examples, {len(done)} already labelled, {len(todo)} to do | model={MODEL} effort={EFFORT} "
          f"cap=${args.cap_usd} ledger_total=${ledger.total:.2f} dry_run={args.dry_run}", flush=True)
    sem, out_lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    with open(args.out, "a") as out_f:
        async def run(i):
            async with sem:
                return await label_one(client, ledger, i, examples[i], args, out_lock, out_f)
        results = await asyncio.gather(*(run(i) for i in todo), return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    for e in errors[:3]:
        print("ERROR:", repr(e)[:300])
    print(f"finished: {len(results) - len(errors)} records, {len(errors)} errors | ledger_total=${ledger.total:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--images-root", required=True, help="directory the examples' image paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
