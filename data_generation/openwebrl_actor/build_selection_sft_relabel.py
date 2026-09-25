#!/usr/bin/env python3
"""Published selection examples + GPT-6 relabels -> two LLaMA-Factory selection sets over the same examples.

Input: owrl_selection_{train,val}.json as published in PTeterwak/action-reward-models-data (written by
build_selection_sft.py) and the matching relabel_gpt6.py output (idx = index into that file).

Output in --out-dir: <name>_gpt6_<split>.json (target = GPT-6's pick) and <name>_gpt55_<split>.json (target = the
published GPT-5.5 pick, the control) with identical prompts; images/ (symlink to the published images);
dataset_info.json registering <name>_gpt6[_val] and <name>_gpt55[_val]; permutations.jsonl; stats.json.

Only examples GPT-6 labelled go into either set. Each example's candidate blocks are shown in a seeded per-example
order (the same order in both sets) and renumbered 1..K; each target follows its candidate, so neither teacher's
position bias is taught. Nothing else in the prompt changes: before anything is written, every example in each input
file must re-render byte-identically in its original order.

--derange draws orders in which every candidate moves. Used for the order-consistency check: that build's
<name>_gpt55_val.json goes back through relabel_gpt6.py.

Usage:
  python build_selection_sft_relabel.py --val owrl_selection_val.json labels_val.jsonl \
      [--train owrl_selection_train.json labels_train.jsonl] --images-root .../selection_sft --out-dir DIR
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

HEAD, TAIL = "#### Candidate Actions ####\n", "\n\n#### Instructions ####"
NUM_RE = re.compile(r"^  (\d+)\. ", re.M)  # candidate number at a line start, as build_messages writes it
TARGET_RE = re.compile(r'\{"selection": (\d+)\}')


def split_candidates(user):
    """user text -> (before, [candidate body without its '  N. ' prefix], after)."""
    if user.count(HEAD) != 1 or user.count(TAIL) != 1:
        raise ValueError("expected one candidate header and one instructions header")
    before, rest = user.split(HEAD)
    block, after = rest.split(TAIL)
    nums = list(NUM_RE.finditer(block))
    if not nums or nums[0].start() != 0 or [int(m.group(1)) for m in nums] != list(range(1, len(nums) + 1)):
        raise ValueError("candidate numbers are not 1..K in order")
    ends = [m.start() - 1 for m in nums[1:]] + [len(block)]  # -1 drops the newline before the next number
    return before, [block[m.end():end] for m, end in zip(nums, ends)], after


def render(before, bodies, after, order):
    return before + HEAD + "\n".join(f"  {j + 1}. {bodies[o]}" for j, o in enumerate(order)) + TAIL + after


def draw_order(seed, split, idx, k, derange):
    rng, order = random.Random(f"{seed}:{split}:{idx}"), list(range(k))
    while True:
        rng.shuffle(order)
        if not derange or all(o != j for j, o in enumerate(order)):
            return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs=2, metavar=("EXAMPLES", "LABELS"))
    ap.add_argument("--val", nargs=2, metavar=("EXAMPLES", "LABELS"))
    ap.add_argument("--images-root", required=True, help="directory the examples' image paths are relative to")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="owrl_selection")
    ap.add_argument("--k", type=int, default=5, help="candidates per example (sample_candidates.py --n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--derange", action="store_true", help="every candidate moves (order-consistency check)")
    args = ap.parse_args()
    splits = [(split, paths) for split, paths in (("train", args.train), ("val", args.val)) if paths]
    if not splits:
        ap.error("give --train and/or --val")

    out, root = Path(args.out_dir), Path(args.images_root)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        (out / "images").symlink_to((root / "images").resolve())
    info, stats, perms = {}, {}, []
    for split, (examples_path, labels_path) in splits:
        examples = json.load(open(examples_path))
        parsed = []
        for i, e in enumerate(examples):
            system, user, target = (m["content"] for m in e["messages"])
            before, bodies, after = split_candidates(user)
            if len(bodies) != args.k:
                raise ValueError(f"{split} {i}: {len(bodies)} candidates, expected {args.k}")
            if render(before, bodies, after, range(len(bodies))) != user:
                raise AssertionError(f"{split} {i}: original order does not re-render byte-identically")
            parsed.append((system, before, bodies, after, int(TARGET_RE.fullmatch(target).group(1))))

        labels = {}
        for line in open(labels_path):
            r = json.loads(line)
            if r.get("status") == "labelled":
                if r["idx"] in labels:
                    raise ValueError(f"{split}: two labels for idx {r['idx']}")
                labels[r["idx"]] = r

        rows = {"gpt6": [], "gpt55": []}
        positions = {key: collections.Counter() for key in ("gpt6_published", "gpt6", "gpt55_published", "gpt55")}
        agree = unchanged = 0
        for idx in sorted(labels):
            r, e = labels[idx], examples[idx]
            system, before, bodies, after, gpt55 = parsed[idx]
            if (r["image"], r["n_candidates"], r["gpt55_selection"]) != (e["images"][0], len(bodies), gpt55):
                raise ValueError(f"{split} {idx}: label record does not match the example")
            if not (root / e["images"][0]).is_file():
                raise FileNotFoundError(root / e["images"][0])
            order = draw_order(args.seed, split, idx, len(bodies), args.derange)
            user = render(before, bodies, after, order)
            if split_candidates(user)[1] != [bodies[o] for o in order]:
                raise AssertionError(f"{split} {idx}: shuffled prompt does not parse back to the shuffled candidates")
            unchanged += user == render(before, bodies, after, range(len(bodies)))
            agree += r["gpt6_selection"] == gpt55
            for teacher, pick in (("gpt6", r["gpt6_selection"]), ("gpt55", gpt55)):
                new = order.index(pick - 1) + 1
                positions[teacher + "_published"][pick] += 1
                positions[teacher][new] += 1
                rows[teacher].append({
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": json.dumps({"selection": new})},
                    ],
                    "images": e["images"],
                })
            perms.append({"split": split, "idx": idx, "row": len(rows["gpt6"]) - 1, "order": order,
                          "image": e["images"][0]})

        for teacher, teacher_rows in rows.items():
            file_name = f"{args.name}_{teacher}_{split}.json"
            (out / file_name).write_text(json.dumps(teacher_rows))
            info[f"{args.name}_{teacher}" + ("_val" if split == "val" else "")] = {
                "file_name": file_name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                         "assistant_tag": "assistant", "system_tag": "system"},
            }
        stats[split] = {
            "examples": examples_path, "labels": labels_path,
            "examples_in_file_round_trip_ok": len(examples), "built": len(rows["gpt6"]),
            "gpt6_equals_gpt55": agree, "prompt_unchanged_by_shuffle": unchanged,
            "target_positions": {key: dict(sorted(c.items())) for key, c in positions.items()},
        }

    stats["args"] = vars(args)
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    (out / "permutations.jsonl").write_text("".join(json.dumps(p) + "\n" for p in perms))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
