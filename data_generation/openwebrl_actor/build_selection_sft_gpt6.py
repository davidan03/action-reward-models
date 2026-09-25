#!/usr/bin/env python3
"""GPT-6-labelled step records -> LLaMA-Factory SFT dataset for the selection ARM.

Input: one or more record directories written live by OpenWebRL's gpt6_step_selector
(SLIME_BROWSER_SEL_RECORD_DIR=<dir>, SLIME_BROWSER_SEL_PROMPT=canonical): <dir>/records.jsonl plus
<dir>/images/<sha256>.png. Each record is one policy step: task, current URL, the split history of the
executed responses, all K candidate responses in draw order, the judge's pick (index into that order) and
the content-addressed screenshot.

Output (same layout as build_selection_sft.py): <name>_train.json / <name>_val.json in ShareGPT format,
images/, dataset_info.json registering <name> and <name>_val, stats.json.

Each example is the canonical OpenWebRL selection prompt built by
openwebrl_integration/openwebrl/arm_inference.selection_messages over the pinned builder — the function the
annotator and serve_arm.py use, so train and serve inputs are byte-identical. Candidates are shown in a
fresh seeded order per example (so the teacher's own position bias is not taught), and the target is
'{"selection": N}' for the judge's pick in that order. Held-out examples are chosen by TASK, so no task
appears in both splits.

Usage:
  python build_selection_sft_gpt6.py --records /path/rec_dir [...] --out-dir /path/selection_sft
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import random
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# the generator appends an end-of-turn token to every response; same rule as gpt6_step_selector._clean (idempotent)
EOT_RE = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+\s*$")


def clean(response):
    return EOT_RE.sub("", response or "")


def load_arm_inference():
    path = REPO / "openwebrl_integration/openwebrl/arm_inference.py"
    spec = importlib.util.spec_from_file_location("arm_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def held_out(task_id, seed, frac):
    h = int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16)
    return h / 2 ** 256 < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help="record directories (records.jsonl + images/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="owrl_gpt6_selection")
    ap.add_argument("--holdout", type=float, default=0.02, help="fraction of TASKS held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-dry-run", action="store_true", help="keep records labelled by the dry-run judge (tests only)")
    args = ap.parse_args()

    ai = load_arm_inference()
    builder = ai.load_selection_builder(str(REPO))  # refuses unless the builder matches the pinned sha256
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    status, skipped, positions = collections.Counter(), collections.Counter(), collections.Counter()
    train, val, tasks = [], [], {"train": set(), "val": set()}
    n_cands, picked_first = [], 0
    for rec_dir in map(Path, args.records):
        for line in open(rec_dir / "records.jsonl"):
            r = json.loads(line)
            status[r["status"]] += 1
            if r["status"] != "selected" or r.get("pick") is None:
                skipped["no_label:" + r["status"]] += 1
                continue
            if r.get("prompt_mode") != "canonical":
                skipped["not_canonical_prompt"] += 1  # labelled on a prompt the ARM will never see
                continue
            if r.get("dry_run") and not args.allow_dry_run:
                skipped["dry_run"] += 1
                continue
            keys = r["keys"]
            if keys[r["pick"]] is None:
                skipped["pick_unparseable"] += 1
                continue
            if len({json.dumps(k, sort_keys=True) for k in keys if k is not None}) < 2:
                skipped["single_distinct_action"] += 1
                continue
            image = rec_dir / r["screenshot"]
            if sha256_file(image) != r["screenshot_sha256"]:
                raise ValueError(f"screenshot hash mismatch: {image}")

            order = list(range(len(r["candidates"])))
            random.Random(f"{args.seed}:{r['task_id']}:{r['step']}").shuffle(order)
            cands = [ai.split_response(clean(r["candidates"][i])) for i in order]
            history = [{"thought": h["thought"], "action": clean(h["action"])} for h in r["history"]]
            msgs = ai.selection_messages(builder, r["task"], r["url"], history, cands, b"")
            user = msgs[1]["content"]
            if len(user) != 3 or user[1].get("type") != "image_url":
                raise ValueError("canonical prompt no longer has text/image/text user blocks")
            target = json.dumps({"selection": order.index(r["pick"]) + 1})

            dst = out / "images" / f"{r['screenshot_sha256']}.png"
            if not dst.exists():
                shutil.copyfile(image, dst)
            example = {
                "messages": [
                    {"role": "system", "content": msgs[0]["content"]},
                    {"role": "user", "content": user[0]["text"] + "<image>" + user[2]["text"]},
                    {"role": "assistant", "content": target},
                ],
                "images": [f"images/{dst.name}"],
            }
            split = "val" if held_out(r["task_id"], args.seed, args.holdout) else "train"
            (val if split == "val" else train).append(example)
            tasks[split].add(r["task_id"])
            positions[order.index(r["pick"]) + 1] += 1
            n_cands.append(len(order))
            picked_first += r["pick"] == 0

    for split, rows in (("train", train), ("val", val)):
        (out / f"{args.name}_{split}.json").write_text(json.dumps(rows))
    info = {}
    for name, split in ((args.name, "train"), (args.name + "_val", "val")):
        info[name] = {
            "file_name": f"{args.name}_{split}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                     "assistant_tag": "assistant", "system_tag": "system"},
        }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    kept = len(train) + len(val)
    stats = {
        "records_by_status": dict(status), "skipped": dict(skipped),
        "train": len(train), "val": len(val),
        "train_tasks": len(tasks["train"]), "val_tasks": len(tasks["val"]),
        "overlapping_tasks": len(tasks["train"] & tasks["val"]),
        "target_position_counts": dict(sorted(positions.items())),
        "mean_candidates": round(sum(n_cands) / kept, 3) if kept else None,
        "judge_picked_policy_sample_rate": round(picked_first / kept, 4) if kept else None,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
