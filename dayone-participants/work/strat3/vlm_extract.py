"""Strategy 3: local VLM (Qwen2.5-VL-3B-Instruct, Apache-2.0) as a second reading engine.

Variant tested: the "hybrid" of strategy 3 §5.3 — the VLM reads the *registered zone crops* produced by
strategy 2 (a 3B model is much more reliable on a single cell than on a dense 280-field page), with the
field label and allowed vocabulary in the prompt. Runs fully offline (edge box / server), no third party.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_DIR = Path.home() / "dayone_local" / "hf" / "Qwen2.5-VL-3B-Instruct"
PROMPT = ("This is one cell of a handwritten maternal health registry form (French, sometimes Arabic or English). "
          "Field: {label}. {vocab}Transcribe exactly the handwritten or typed value written in the cell. "
          "Ignore printed form labels and table lines. If the cell is empty, answer EMPTY. "
          "Answer with the value only.")


class VLMReader:
    def __init__(self, four_bit=True):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        kw = dict(torch_dtype=torch.float16, device_map="cuda")
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_quant_type="nf4")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(MODEL_DIR), **kw).eval()
        self.proc = AutoProcessor.from_pretrained(str(MODEL_DIR), min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 256)
        self.proc.tokenizer.padding_side = "left"

    @torch.no_grad()
    def read(self, crops: list[np.ndarray], labels: list[str], vocabs: list[list[str]] | None = None, bs=8):
        out = []
        for s in range(0, len(crops), bs):
            msgs, ims = [], []
            for i in range(s, min(s + bs, len(crops))):
                c = crops[i]
                # upscale small cells so the vision tower sees enough pixels
                im = Image.fromarray(c)
                if im.height < 56:
                    f = 56 / im.height
                    im = im.resize((int(im.width * f), 56))
                voc = ""
                if vocabs and vocabs[i]:
                    voc = "Typical values: " + ", ".join(vocabs[i][:12]) + ". "
                text = PROMPT.format(label=labels[i], vocab=voc)
                m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
                msgs.append(self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
                ims.append(im)
            inp = self.proc(text=msgs, images=ims, return_tensors="pt", padding=True).to("cuda")
            gen = self.model.generate(**inp, max_new_tokens=24, do_sample=False)
            dec = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            for d in dec:
                d = d.strip().strip('"').strip()
                out.append(None if d.upper() in ("EMPTY", "") else d)
        return out
