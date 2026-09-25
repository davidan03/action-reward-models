#!/usr/bin/env python3
"""Serve a frozen OpenWebRL selection ARM or scalar LoRA/head on an assigned GPU."""
import argparse
import base64
import io
import json
import threading
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from openwebrl.arm_inference import load_selection_builder, scalar_messages, selection_messages, selection_schema


class SelectionRequest(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str


def make_app(args):
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from safetensors.torch import load_file

    base_config = AutoConfig.from_pretrained(args.base, local_files_only=True)
    # Newer merged configs moved mrope settings to rope_parameters. Use the
    # original actor's configuration with 4.57 only after checking architecture.
    if args.mode == "selection":
        released = json.loads((Path(args.model) / "config.json").read_text())
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "vocab_size"):
            if released["text_config"][key] != getattr(base_config.text_config, key):
                raise ValueError(f"Selection architecture differs from base: {key}")
    processor = AutoProcessor.from_pretrained(
        args.base, local_files_only=True, min_pixels=65536, max_pixels=args.max_pixels)
    if args.mode == "selection":
        from tokenizers import Tokenizer
        released_tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
        if released_tokenizer.get_vocab() != processor.tokenizer.get_vocab():
            raise ValueError("Selection vocabulary differs from actor processor")
        if (Path(args.model) / "chat_template.jinja").read_text() != (Path(args.base) / "chat_template.jinja").read_text():
            raise ValueError("Selection chat template differs from actor processor")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model if args.mode == "selection" else args.base,
        config=base_config, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa", local_files_only=True)
    value_head = None
    if args.mode == "scalar":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model, local_files_only=True)
        value_head = {k: v.to(dtype=torch.bfloat16) for k, v in
                      load_file(str(Path(args.model) / "value_head.safetensors"), device=args.device).items()}
    model.eval()
    model.requires_grad_(False)
    builder = load_selection_builder(args.source_root)
    grammar_compiler = None
    if args.mode == "selection":
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            processor.tokenizer, vocab_size=base_config.text_config.vocab_size)
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    app, lock = FastAPI(), threading.Lock()

    @app.get("/health")
    def health():
        return {"mode": args.mode, "model": args.model, "base": args.base,
                "max_pixels": args.max_pixels, "config_source": args.base,
                "source_root": args.source_root,
                **({"selection_decoding": "canonical_no_cot_json_schema/xgrammar"}
                   if args.mode == "selection" else {})}

    @app.post("/select")
    def select(request: SelectionRequest):
        if request.mode != args.mode or not 1 <= len(request.candidates) <= 5:
            from fastapi import HTTPException
            raise HTTPException(400, "Wrong mode or unsupported candidate count")
        image_bytes = base64.b64decode(request.screenshot, validate=True)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with lock, torch.inference_mode():
            if args.mode == "selection":
                messages = selection_messages(builder, request.task, request.url,
                                              request.history, request.candidates, image_bytes)
                # Processor chat templates use type=image. Preserve the exact text
                # blocks and screenshot position produced by the canonical builder.
                messages[1]["content"][1] = {"type": "image"}
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], return_tensors="pt").to(args.device)
                if inputs.input_ids.shape[-1] + args.max_new_tokens > args.context_length:
                    raise ValueError("Selection prompt exceeds configured context length")
                # The canonical released no-CoT inference function supplies a
                # strict JSON schema. Its minimal demo omits that constraint.
                # Recreate the canonical constraint instead of accepting bare
                # numbers or silently executing candidate 1 after a parse error.
                grammar = grammar_compiler.compile_json_schema(selection_schema(len(request.candidates)))
                output = model.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                    logits_processor=[LogitsProcessor(grammar)])
                raw = processor.tokenizer.decode(output[0, inputs.input_ids.shape[-1]:],
                                                  skip_special_tokens=True)
                return {"raw": raw}
            scores = []
            weight = value_head["v_head.summary.weight"]
            bias = value_head.get("v_head.summary.bias")
            for candidate in request.candidates:
                messages = scalar_messages(request.task, request.url, request.history, candidate)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = processor(text=[text], images=[img], return_tensors="pt").to(args.device)
                if inputs.input_ids.shape[-1] > args.context_length:
                    raise ValueError("Scalar prompt exceeds configured context length")
                result = model(**inputs, output_hidden_states=True, use_cache=False)
                h = result.hidden_states[-1][0, -1].to(weight.dtype)
                score = (h @ weight.T).squeeze()
                if bias is not None:
                    score = score + bias.squeeze()
                scores.append(float(score))
            return {"scores": scores}

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["selection", "scalar"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
    ap.add_argument("--source-root", default="/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=19101)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    uvicorn.run(make_app(args), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
