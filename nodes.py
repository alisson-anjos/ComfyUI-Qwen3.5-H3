"""Qwen3.5-9B -> MiniMax-H3 projection CLIP wrapper and ComfyUI loader node."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor

HF_PROJ_REPO = "Alissonerdx/Qwen3.5-9B-Minimax-H3"


def _resolve_model_file(path_or_name: str, default_hf_filename: str, auto_download: bool) -> str:
    """Return an existing local path, or download from HF into ComfyUI's models dir."""
    val = (path_or_name or "").strip()
    if val and Path(val).exists():
        return val
    try:
        import folder_paths
        cache_dir = Path(folder_paths.models_dir) / "text_encoders" / "qwen35_h3"
    except Exception:
        cache_dir = Path.home() / ".cache" / "qwen35_h3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if val:
        candidate = cache_dir / Path(val).name
        if candidate.exists():
            return str(candidate)
    if not auto_download:
        raise FileNotFoundError(f"Model file not found locally and auto_download=False: {val or default_hf_filename}")
    hf_file = val if ("/" in val and not val.startswith("/")) else default_hf_filename
    return hf_hub_download(repo_id=HF_PROJ_REPO, filename=hf_file, local_dir=str(cache_dir))


class Qwen35H3Clip:
    """Drop-in ComfyUI CLIP-compatible wrapper for MiniMax-H3 conditioning."""

    def __init__(self, model_path: str, projection_path: str, ara_path: str = "", device: str = "cuda"):
        self.model_path = str(model_path)
        self.projection_path = str(projection_path)
        self.ara_path = str(ara_path or "")
        self.device = device

        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        for name in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]:
            if hasattr(module, name):
                setattr(module, name, inspect.unwrap(getattr(module, name)))

        local_only = Path(self.model_path).exists()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=local_only)
        self.tok = self.processor.tokenizer
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
                local_files_only=local_only,
            )
            .eval()
            .requires_grad_(False)
        )
        if self.ara_path and Path(self.ara_path).exists():
            with safe_open(self.ara_path, framework="pt", device="cpu") as f:
                ara_meta = f.metadata() or {}
            rank = int(ara_meta.get("rank", "16"))
            alpha = float(ara_meta.get("alpha", str(rank)))
            scale = alpha / rank
            ara_sd = load_file(self.ara_path, device=self.device)
            prefixes = sorted({k.rsplit(".", 1)[0] for k in ara_sd if k.endswith(".lora_A")})
            for mod_name in prefixes:
                mod = self.model.get_submodule(mod_name)
                A = ara_sd[f"{mod_name}.lora_A"].float()
                B = ara_sd[f"{mod_name}.lora_B"].float()
                mod.weight.data += (B @ A * scale).to(mod.weight.dtype)
        self.language = self.model.model.language_model
        self.language.norm = torch.nn.Identity()

        with safe_open(self.projection_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        self.tap = int(meta.get("tap", "20"))
        saved = load_file(self.projection_path, device=self.device)
        self.proj = {
            k: saved[k].float()
            for k in ["W", "mean_in", "std_in", "mean_out", "std_out"]
        }
        residual = {k.removeprefix("mlp."): v for k, v in saved.items() if k.startswith("mlp.")}
        if residual:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(4096, 8192),
                torch.nn.GELU(),
                torch.nn.Linear(8192, 5120),
            ).to(self.device)
            self.mlp.load_state_dict(residual)
            self.mlp.eval().requires_grad_(False)
        else:
            self.mlp = None

    def tokenize(self, text: str, minimax_ref_items=None, **kwargs):
        vision_start = self.tok.convert_tokens_to_ids("<|vision_start|>")
        vision_end = self.tok.convert_tokens_to_ids("<|vision_end|>")
        image_token_id = self.model.config.image_token_id

        ids: list[int] = []
        visual_spans: list[tuple[int, int]] = []
        pil_images: list[Image.Image] = []

        img_idx = 0
        for item in minimax_ref_items or []:
            if item.get("type") == "image" and item.get("data") is not None:
                img_idx += 1
                tensor = item["data"][0].detach().clamp(0, 1).cpu().numpy()
                pil_images.append(Image.fromarray((tensor * 255).round().astype(np.uint8)))
                label = self.tok(f"<Picture {img_idx}>: ", add_special_tokens=False)["input_ids"]
                ids += label
            elif item.get("type") == "audio":
                pass

        if pil_images:
            vision = self.processor.image_processor(images=pil_images, return_tensors="pt", do_resize=False)
            grids = vision["image_grid_thw"]
            # Re-build ids in exact order with vision blocks inserted after each <Picture i>:
            ids = []
            for j, g in enumerate(grids, 1):
                label = self.tok(f"<Picture {j}>: ", add_special_tokens=False)["input_ids"]
                ids += label
                start_vision = len(ids)
                n = int(g.prod()) // 4
                ids += [vision_start] + [image_token_id] * n + [vision_end]
                visual_spans.append((start_vision, len(ids)))
            pixel_values = vision["pixel_values"]
        else:
            grids = None
            pixel_values = None

        if text:
            ids += self.tok(text, add_special_tokens=False)["input_ids"]

        tags = torch.ones(len(ids), dtype=torch.long)
        for s, e in visual_spans:
            tags[s:e] = 0

        return {
            "ids": ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grids,
            "token_tags": tags,
        }

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        ids = tokens["ids"]
        if not ids:
            cond = torch.zeros((1, 1, 5120), dtype=torch.bfloat16)
            extra = {
                "pooled_output": torch.zeros((1, 5120), dtype=torch.bfloat16),
                "minimax_token_tags": torch.ones((1, 1), dtype=torch.long),
            }
            if return_dict:
                return {"cond": cond, **extra}
            if return_pooled:
                return cond, extra["pooled_output"]
            return cond
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": (input_ids == self.model.config.image_token_id).int(),
        }
        if tokens["pixel_values"] is not None:
            inputs["pixel_values"] = tokens["pixel_values"].to(self.device, torch.bfloat16)
            inputs["image_grid_thw"] = tokens["image_grid_thw"].to(self.device)

        captured = {}

        def capture(mod, args, output):
            captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

        handle = self.language.layers[self.tap - 1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                self.model.model(**inputs, use_cache=False)
        finally:
            handle.remove()

        h = captured["h"]
        p = self.proj
        with torch.inference_mode():
            z = (h.float() - p["mean_in"]) / p["std_in"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = z @ p["W"]
                if self.mlp is not None:
                    y = y + self.mlp(z)
            cond = (y.float() * p["std_out"] + p["mean_out"]).to(torch.bfloat16).cpu()

        extra = {
            "pooled_output": torch.zeros((1, 5120), dtype=torch.bfloat16),
            "minimax_token_tags": tokens["token_tags"].unsqueeze(0),
        }
        if return_dict:
            return {"cond": cond, **extra}
        if return_pooled:
            return cond, extra["pooled_output"]
        return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        out = self.encode_from_tokens(tokens, return_dict=True)
        cond = out.pop("cond")
        if add_dict:
            out.update(add_dict)
        return [[cond, out]]


class Qwen35H3ClipLoader:
    """ComfyUI node that loads Qwen3.5-9B + a trained MiniMax-H3 projection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen3.5-9B"}),
                "projection_file": ("STRING", {"default": "minimax_h3_qwen3.5_9b_projection.safetensors"}),
                "auto_download": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lora_file": ("STRING", {"default": "minimax_h3_qwen3.5_9b_lora.safetensors"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"

    def load(self, model_path: str, projection_file: str, auto_download: bool = True, lora_file: str = ""):
        proj_path = _resolve_model_file(projection_file, "minimax_h3_qwen3.5_9b_projection.safetensors", auto_download)
        lora_path = _resolve_model_file(lora_file, "minimax_h3_qwen3.5_9b_lora.safetensors", auto_download) if (lora_file or "").strip() else ""
        return (Qwen35H3Clip(model_path=model_path, projection_path=proj_path, ara_path=lora_path),)


NODE_CLASS_MAPPINGS = {
    "Qwen35H3ClipLoader": Qwen35H3ClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35H3ClipLoader": "Qwen3.5-9B H3 Projection CLIP Loader",
}
