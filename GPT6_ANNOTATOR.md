# GPT-6 as the ARM annotator — plan and status

Goal: train a selection ARM with **GPT-6** as the teacher instead of GPT-5.5, for use with OpenWebRL-style
Qwen3-VL-4B actors (inference-time selection, SFT-data generation, RL turn bonuses).

## Step 1 (current): relabel the published examples

We start by relabelling the examples this repo already published, changing **only the teacher**:

| | kept from the published data | changed |
|---|---|---|
| states ("screens") | 3,085 per-turn states of the 412 released OpenWebRL SFT trajectories | — |
| candidate sets | the published sets: 5 actions per state from the released OpenWebRL-4B-SFT (~75 % sampled at T 0.7, ~25 % at T 1.0, top-p 0.9) | — |
| prompt | the exact published prompt text (explain in 2–4 sentences, then `{"selection": N}`) | — |
| teacher | — | GPT-5.5 → GPT-6 (`gpt-6-astra`, reasoning effort medium) |

Why this first:
- **Cheapest.** No candidate sampling, no browsers, no GPU jobs — only GPT-6 calls.
- **Cleanest comparison.** Same states, candidates and prompts, so an ARM trained on GPT-6 labels can be compared
  directly with one trained on the GPT-5.5 labels.
- **Transfer is plausible.** The published ARM, trained only on the released SFT's candidates, already improved a
  reproduction of that SFT (+12.7 pp task success on 100 WebGym training tasks, GPT-4.1 judge, single run).

Source data: `PTeterwak/action-reward-models-data`, `openwebrl_actor/selection_sft/` —
`owrl_selection_train.json` (~40k examples), `owrl_selection_val.json` (782 examples, the published 2 % split),
`images/` (2,558 screenshots).

## How

`data_generation/openwebrl_actor/relabel_gpt6.py` sends each example to GPT-6 **unchanged** (system prompt, user
text, screenshot in place of `<image>`) and records the reply's last `{"selection": N}` next to the original GPT-5.5
label. Each call reserves an upper-bound cost in a ledger before it is sent and settles to the billed usage; the run
stops at `--cap-usd`. Resumable; `--dry-run` builds every request without calling the API.

```bash
python data_generation/openwebrl_actor/relabel_gpt6.py \
    --examples <selection_sft>/owrl_selection_val.json --images-root <selection_sft> \
    --out labels_val.jsonl --ledger ledger.jsonl --cap-usd 50
```

## Pilot result (100 validation examples)

- 100/100 labelled, 0 parse failures; every reply is a short explanation (median 54 words) then the JSON line.
- Cost **$0.0566 per label** (~4,090 input + ~111 output tokens; every input token was billed as a cache write, with
  no cache hits).
- GPT-6 vs GPT-5.5: same candidate 58/100, same action text 63/100. GPT-6 calls the candidates equivalent (usually
  the same click at slightly different coordinates) in 55/100 explanations, so much of the disagreement is between
  near-identical options.
- **Position bias in both teachers.** GPT-5.5 picked slot 1 or 2 in 57/100 (40 expected); GPT-6 picked slot 1 in
  41/100 (20 expected), mostly when breaking ties between equivalent candidates.

## Order-consistency check (same 100 examples, every candidate moved)

The same 100 examples with each candidate set reordered so that no candidate kept its slot
(`build_selection_sft_relabel.py --derange`), relabelled by GPT-6 with otherwise identical prompts ($5.68, 100/100
labelled). One relabel per order, so order effects and GPT-6's own sampling randomness are not separated.

- **GPT-6's judgement is stable.** It picked the same candidate in 69/100 (20 by chance) and a near-identical action
  (same tool calls, every click within 1 % of the screen) in 93/100 (47 by chance). What order mostly changes is which
  of several near-identical clicks it takes.
- **Those tie-breaks favour the first slot:** 12 of the 31 changed picks went to the new slot 1 (about 8 if spread
  evenly). Shuffling the candidate order in the training sets (step 2) keeps this habit out of the ARM.
- **The teachers really differ on about one example in five.** GPT-6 and GPT-5.5 agree on a near-identical action in
  78/100 (published order) and 80/100 (reordered), against 93/100 for GPT-6 against itself. Those ~20 examples are
  what a GPT-6 relabel changes.
- 14/100 examples offer no real choice (all five candidates near-identical); their label is only a tie-break.
- For scoring, exact-candidate agreement tops out near 70 % even for GPT-6 against itself; compare ARMs on
  near-identical-action agreement instead.

## Next steps

1. Relabel all 782 validation examples and ~3 candidate sets per state from the training file (~9k labels,
   ~$550 at the pilot rate). A learning curve on nested subsets decides whether more labels are worth buying.
2. Build training sets with the **candidate order shuffled per example** (the label follows the chosen candidate), so
   neither teacher's tie-breaking habit becomes a slot preference in the ARM:
   `data_generation/openwebrl_actor/build_selection_sft_relabel.py` writes a GPT-6-label set and a GPT-5.5-label
   control over the same examples with identical prompts. It refuses to write anything unless every example in the
   input file re-renders byte-identically in its original order (checked on all 782 validation examples).
3. Train two ARMs on the same subset with `training/llamafactory/arm_lora.yaml` on `OpenWebRL/OpenWebRL-4B-SFT`:
   one on GPT-6 labels, one on GPT-5.5 labels (control).
4. Evaluate: agreement with GPT-6 on the relabelled validation set (also scoring the published ARM), then best-of-5
   task success on Online-Mind2Web with an OpenWebRL SFT actor.

## Known caveats

- **Teacher-path states only.** All states lie on the teacher's demonstrations of 412 tasks the actor was trained
  on; the ARM never sees the actor's own mistakes. The published ARM is weakest exactly there.
- **Prompt formats differ between training and serving.** The published training prompts ask the model to explain
  first; they came from an unpublished module (`openwebrl.frontier_arbiter`). This repo's
  `inference/selection_prompt.py` only builds the no-explanation variant, which is what the reproduced runs served
  the ARM with. An ARM trained on relabelled published prompts inherits that same train/serve difference when it is
  served on new states.
- **Validation overlaps training by state.** The published split is 2 % of labels, not of tasks, so the same state
  can appear in both; validation agreement overstates generalisation. Task success is the decisive test.
- **End-of-turn token when serving on OpenWebRL outputs.** OpenWebRL responses end in `<|im_end|>`; strip it from
  candidates and history before building an ARM prompt (see `openwebrl_integration/README.md`).

## Later: the actor's own states

If the relabelled ARM falls short on the actor's own states, the same annotator runs **live**: the OpenWebRL actor
browses training tasks, samples 5 candidates per step, and GPT-6 labels them with the canonical prompt. The pieces in
this fork:

| file | purpose |
|---|---|
| `data_generation/openwebrl_actor/make_webgym_tasks.py` | WebGym training pool → task file for live collection (eval intents refused, blacklisted hosts dropped) |
| `data_generation/openwebrl_actor/build_selection_sft_gpt6.py` | live step records → LLaMA-Factory selection-SFT set |
| `training/llamafactory/arm_lora_gpt6.yaml` | the `arm_lora.yaml` recipe with an OpenWebRL SFT reproduction as the base |
| `openwebrl_integration/` | canonical prompt helper and ARM server copied verbatim from zixianma/OpenWebRL@0998eaab |

The live recorder itself lives in the OpenWebRL harness (the GPT-6 step selector's record mode); it has passed a
dry-run smoke test on 3 WebGym tasks but has not yet made paid calls.
