#!/usr/bin/env python3
"""OpenWebRL's WebGym training pool (parquet) -> the task JSONL the OpenWebRL eval harness reads.

Collection runs the policy on TRAINING tasks only, never on the Online-Mind2Web test set. Rows are written in the
Online-Mind2Web layout (metadata.intent / start_url / task_id). WebGym ids look like 'webvoyager/213974'; the '/'
would become a directory in the harness's per-task result filenames, so ids are rewritten to 'webgym_213974' and the
original is kept as metadata.webgym_task_id. Tasks whose start host is in the navigation-failure blacklist are
dropped, and a task whose intent exactly matches an eval task (--exclude-intents) is refused.

Usage:
  python make_webgym_tasks.py --out tasks/webgym_pilot50.jsonl --n 50 --seed 0 \
      --exclude-intents <OpenWebRL>/openwebrl/data/online-mind2web.jsonl <OpenWebRL>/openwebrl/data/eval/webvoyager_fara.jsonl
"""
import argparse
import ast
import json
import random
from pathlib import Path
from urllib.parse import urlparse

OWRL_DATA = Path("/mmfs1/gscratch/krishna/dan29/OpenWebRL/openwebrl/data")


def host(url):
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def eval_intents(paths):
    out = set()
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            m = d.get("metadata") or {}
            out.add((m.get("intent") or d.get("ques") or "").strip().lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(OWRL_DATA / "webgym_filtered_popular_2102_cleaned.parquet"))
    ap.add_argument("--blacklist", default=str(OWRL_DATA / "webgym_filtered_popular_blacklist_hosts.txt"))
    ap.add_argument("--exclude-intents", nargs="*", default=[], help="eval task JSONLs whose intents must not appear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="sample this many tasks (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet)
    bad = {l.strip().lower() for l in open(args.blacklist) if l.strip() and not l.startswith("#")}
    banned = eval_intents(args.exclude_intents)
    rows, dropped = [], 0
    for raw in df["metadata"]:
        m = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
        if host(m["start_url"]) in bad:
            dropped += 1
            continue
        if m["task"].strip().lower() in banned:
            raise ValueError(f"WebGym task duplicates an eval task: {m['task']!r}")
        tid = "webgym_" + m["task_id"].split("/")[-1]
        rows.append({"benchmark_name": "webgym", "task_name": m["task"], "website": m["start_url"], "task_id": tid,
                     "metadata": {"intent": m["task"], "start_url": m["start_url"], "sites": "[]", "task_id": tid,
                                  "webgym_task_id": m["task_id"], "require_login": "False", "storage_state": "None",
                                  "require_reset": "False", "intent_template_id": "0", "benchmark_name": "webgym"}})
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("task_id collision after sanitising")
    if args.n:
        rows = random.Random(args.seed).sample(rows, args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} tasks -> {args.out} (dropped {dropped} blacklisted-host tasks; "
          f"{len(banned)} eval intents checked)")


if __name__ == "__main__":
    main()
