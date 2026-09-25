# OpenWebRL integration files (copied, unmodified)

Copied verbatim from Zixian Ma's OpenWebRL fork, branch `arm`, commit
`0998eaabda26c7a77e0e1dba4687d2932bc382d2`
(https://github.com/zixianma/OpenWebRL/tree/arm):

| file here | source path | used for |
|---|---|---|
| `openwebrl/arm_inference.py` | `openwebrl/arm_inference.py` | the canonical selection prompt for OpenWebRL actors (`split_response`, `selection_messages`, `load_selection_builder`) — shared by the GPT-6 annotator, the dataset builder and serving, so all three see byte-identical ARM inputs |
| `scripts/serve_arm.py` | `scripts/serve_arm.py` | serving a trained selection ARM (or scalar LoRA) behind her `/select` HTTP contract, e.g. for her RL turn-bonus code |

`load_selection_builder(source_root)` loads `<source_root>/inference/selection_prompt.py`
and refuses to run unless its sha256 is `043ab7e9…751c`. This repository's copy matches,
so pass the repository root as `source_root`.

`serve_arm.py` imports `openwebrl.arm_inference`, so to serve an ARM, copy
`openwebrl/arm_inference.py` into an OpenWebRL checkout's `openwebrl/` package and run
`serve_arm.py` from there.

**End-of-turn token.** OpenWebRL's generator appends `<|im_end|>` to every response. Our annotator records and
`build_selection_sft_gpt6.py` strip it before `split_response` (otherwise the tokenizer turns it into a real turn boundary
inside the ARM's user message). `ActionSelector` passes raw outputs, so when serving an ARM trained on this data, strip the
same token from each candidate (and history response) before `split_response` to keep train and serve inputs identical.
