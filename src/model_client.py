"""Model clients. Target models expose generate_batch(prompts) -> list[str];
judges expose generate(prompt) -> (text, usage). Providers are chosen by name."""

import os
import random
import time


class QuotaExhausted(Exception):
    """Raised when the judge API keeps refusing (quota/rate). Progress is saved by the caller."""


# ---------------------------------------------------------------- target: Hugging Face
class HFClient:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg = cfg
        self.torch = torch
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
        self.tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
        kw = {}
        prec = cfg["precision"]
        if prec == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        elif prec == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif prec == "fp16":
            kw["torch_dtype"] = torch.float16
        else:
            raise ValueError(f"unknown precision {prec!r}")
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model"], device_map="auto", **kw)
        self.model.eval()

    def _generate(self, prompts):
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=self.cfg["max_new_tokens"],
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    def free_gpu(self):
        import gc

        gc.collect()
        self.torch.cuda.empty_cache()

    def generate_batch(self, prompts):
        """Halves the batch on CUDA out-of-memory, down to 1.

        The retry must happen OUTSIDE the except block: while it is active the exception's traceback still
        references the failed attempt's tensors, so the memory cannot be freed and every retry fails too.
        """
        try:
            return self._generate(prompts)
        except self.torch.cuda.OutOfMemoryError:
            pass
        self.free_gpu()
        if len(prompts) == 1:
            return self._generate(prompts)  # last attempt with a clean GPU, a failure here is a real error
        mid = len(prompts) // 2
        return self.generate_batch(prompts[:mid]) + self.generate_batch(prompts[mid:])


# ---------------------------------------------------------------- target: fake (tests only)
class FakeTarget:
    """Deterministic fake used only by the self-test. Never used for real results."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.calls = 0

    def generate_batch(self, prompts):
        crash_after = int(os.environ.get("FAKE_CRASH_AFTER", "0") or 0)
        out = []
        for p in prompts:
            self.calls += 1
            if crash_after and self.calls > crash_after:
                raise KeyboardInterrupt("simulated crash")
            h = sum(map(ord, p)) % 4
            if "EVIDENCE_STATE" in p:
                st = ["CLEAR", "CONTRADICTORY", "MISSING_INFORMATION", "AMBIGUOUS"][h]
                fin = "The answer is 1611." if st == "CLEAR" else "REFUSE_OTHER"
                out.append(f"EVIDENCE_STATE: {st}\nFINAL: {fin}")
                continue
            if "FINAL:" in p:
                out.append("Reasoning...\nFINAL: " + ["REFUSE_INFO_MISSING_IN_CONTEXT", "The answer is 1611.",
                           "REFUSE_CONTRADICTORY_CONTEXT", "I cannot answer: unclear."][h])
                continue
            out.append(
                [
                    "REFUSE_INFO_MISSING_IN_CONTEXT",
                    "The answer is 1611.",
                    "REFUSE_CONTRADICTORY_CONTEXT",
                    "I cannot answer: the passage is unclear.",
                ][h]
            )
        return out


# ---------------------------------------------------------------- judge: Gemini
class GeminiJudge:
    def __init__(self, cfg: dict):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.cfg = cfg
        self.client = genai.Client(api_key=key)
        self._last = 0.0

    def generate(self, prompt: str):
        from google.genai import errors, types

        time.sleep(max(0.0, self.cfg["min_interval_s"] - (time.time() - self._last)))
        last_err = None
        for attempt in range(1, self.cfg["max_attempts"] + 1):
            try:
                out = self.client.models.generate_content(
                    model=self.cfg["model"],
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=self.cfg["temperature"]),
                )
                self._last = time.time()
                u = out.usage_metadata
                return out.text or "", {
                    "tokens_in": getattr(u, "prompt_token_count", None),
                    "tokens_out": getattr(u, "candidates_token_count", None),
                }
            except (errors.ServerError, errors.ClientError) as e:
                if getattr(e, "code", None) not in (429, 500, 502, 503, 504):
                    raise
                last_err = e
                if getattr(e, "code", None) == 429 and "PerDay" in str(e):
                    raise QuotaExhausted(f"daily free-tier quota used up for {self.cfg['model']}: {str(e)[:400]}")
                time.sleep(min(60, 2**attempt) + random.random())
        raise QuotaExhausted(f"judge gave up after {self.cfg['max_attempts']} attempts: {last_err}")


# ---------------------------------------------------------------- judge: fake (tests only)
class FakeJudge:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, prompt: str):
        reply = prompt.split("MODEL RESPONSE:")[1].split("REFERENCE ANSWERS:")[0]
        if "1611" in reply:
            body = "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: fake"
        elif "cannot answer" in reply:
            body = "CLASSIFICATION: REFUSE_OTHER\nQUALITY_SCORE: N/A\nEXPLANATION: fake"
        else:
            body = "CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 3\nEXPLANATION: fake"
        return body, {"tokens_in": 0, "tokens_out": 0}


TARGETS = {"hf": HFClient, "fake": FakeTarget}
JUDGES = {"gemini": GeminiJudge, "fake": FakeJudge}


def make_target(cfg):
    return TARGETS[cfg["provider"]](cfg)


def make_judge(cfg):
    return JUDGES[cfg["provider"]](cfg)
