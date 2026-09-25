#!/usr/bin/env python3
"""Pick a nested per-state subset of the published selection examples for relabelling.

The published training file holds ~13 candidate sets (examples) per state. This keeps up to --per-state of them for
every state, chosen by a seeded shuffle within the state, and writes them rank-major: every state's first pick, then
every state's second, ... So any prefix of the output (and so any relabel run stopped part-way) is a complete
"k per state" subset, and k = 1, 2, ... are nested for a learning curve. A state is the example's screenshot plus its
prompt text outside the candidate block. Examples whose candidate block build_selection_sft_relabel.py cannot split
unambiguously (a malformed candidate containing its own '  N. ' lines: 2 of 39,155 in the published train file) are
left out, since their candidates could not be shuffled.

Output: --out (same format as the input; feed it to relabel_gpt6.py and build_selection_sft_relabel.py) and --index
(one line per output example: idx, source_idx, state, rank, state_size).

Usage:
  python select_relabel_subset.py --examples owrl_selection_train.json --per-state 3 \
      --out owrl_selection_train_3ps.json --index owrl_selection_train_3ps_index.jsonl
"""
import argparse
import collections
import hashlib
import json
import random

from build_selection_sft_relabel import split_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--per-state", type=int, required=True)
    ap.add_argument("--k", type=int, default=5, help="candidates per example (sample_candidates.py --n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", required=True)
    args = ap.parse_args()

    examples = json.load(open(args.examples))
    states = collections.defaultdict(list)  # insertion order = first appearance in the file
    excluded = []  # candidate blocks the builder cannot split unambiguously; they could not be shuffled
    for i, e in enumerate(examples):
        try:
            before, bodies, after = split_candidates(e["messages"][1]["content"])
        except ValueError:
            excluded.append(i)
            continue
        if len(bodies) != args.k:
            excluded.append(i)
            continue
        key = json.dumps([e["messages"][0]["content"], e["images"][0], before, after])
        states[hashlib.sha256(key.encode()).hexdigest()[:16]].append(i)
    for state, idxs in states.items():
        random.Random(f"{args.seed}:{state}").shuffle(idxs)

    picked = [(rank, state, idxs[rank], len(idxs)) for rank in range(args.per_state)
              for state, idxs in states.items() if rank < len(idxs)]
    with open(args.out, "w") as f:
        json.dump([examples[i] for _, _, i, _ in picked], f)
    with open(args.index, "w") as f:
        for j, (rank, state, i, size) in enumerate(picked):
            f.write(json.dumps({"idx": j, "source_idx": i, "state": state, "rank": rank, "state_size": size}) + "\n")
    sizes = collections.Counter(len(v) for v in states.values())
    print(json.dumps({"examples": len(examples), "excluded_unsplittable": excluded, "states": len(states),
                      "state_sizes": dict(sorted(sizes.items())), "picked": len(picked),
                      "picked_per_rank": dict(collections.Counter(r for r, *_ in picked))}))


if __name__ == "__main__":
    main()
