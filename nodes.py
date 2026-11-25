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


class Qwen35H3Apply:
    """Apply MiniMax-H3 projection (and optional LoRA) to a native ComfyUI Qwen3.5-9B CLIP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Native Qwen3.5-9B CLIP loaded via ComfyUI's Load CLIP node."}),
                "use_lora": ("BOOLEAN", {"default": True, "tooltip": "Enable the trained rank-16 LoRA (visual.merger + layers 12-19) to boost <Picture 1..N> reference identity fidelity and prompt alignment."}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Automatically download missing projection/LoRA weights from Alissonerdx/Qwen3.5-9B-Minimax-H3 if not already cached locally."}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"

    def apply(self, clip, use_lora: bool = True, auto_download: bool = True):
        from types import MethodType
        from comfy.weight_adapter import LoRAAdapter

        proj_name = "minimax_h3_qwen3.5_9b_projection.safetensors"
        lora_name = "minimax_h3_qwen3.5_9b_lora.safetensors"
        proj_path = _resolve_model_file(proj_name, proj_name, auto_download)
        print(f"[ComfyUI-Qwen3.5-H3] Projection ready: {proj_path}", flush=True)
        with safe_open(proj_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        tap = int(meta.get("tap", "20"))
        saved = load_file(proj_path, device="cpu")
        proj = {k: saved[k].float() for k in ["W", "mean_in", "std_in", "mean_out", "std_out"]}
        residual = {k.removeprefix("mlp."): v.float() for k, v in saved.items() if k.startswith("mlp.")}
        mlp = None
        if residual:
            mlp = torch.nn.Sequential(
                torch.nn.Linear(4096, 8192),
                torch.nn.GELU(),
                torch.nn.Linear(8192, 5120),
            )
            mlp.load_state_dict(residual)
            mlp.eval().requires_grad_(False)

        out_clip = clip.clone()
        # Configure native Qwen3.5 TE to output unnormalized hidden state at layer tap - 1 (20th layer = idx 19).
        te_name = getattr(out_clip.cond_stage_model, "clip_name", None) or "qwen35_9b"
        te_mod = getattr(out_clip.cond_stage_model, te_name, None)
        if te_mod is None:
            for attr in dir(out_clip.cond_stage_model):
                obj = getattr(out_clip.cond_stage_model, attr, None)
                if hasattr(obj, "layer_idx") and hasattr(obj, "transformer"):
                    te_mod = obj
                    te_name = attr
                    break
        if te_mod is None:
            raise ValueError("Expected a Qwen3.5 CLIP model from Load CLIP.")
        te_mod.layer = "hidden"
        te_mod.layer_idx = tap - 1
        te_mod.layer_norm_hidden_state = False

        if use_lora:
            lora_path = _resolve_model_file(lora_name, lora_name, auto_download)
            print(f"[ComfyUI-Qwen3.5-H3] LoRA ready: {lora_path}", flush=True)
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                ara_meta = f.metadata() or {}
            rank = int(ara_meta.get("rank", "16"))
            alpha = float(ara_meta.get("alpha", str(rank)))
            lora_sd = load_file(lora_path, device="cpu")
            sd_keys = set(out_clip.patcher.model.state_dict().keys())
            patches = {}
            for prefix in {k.rsplit(".", 1)[0] for k in lora_sd if k.endswith(".lora_A")}:
                mapped = prefix.replace("model.language_model.", "model.").replace("model.visual.", "visual.")
                target_key = f"{te_name}.transformer.{mapped}.weight"
                if target_key not in sd_keys:
                    matches = [k for k in sd_keys if k.endswith(f".{mapped}.weight")]
                    if not matches:
                        continue
                    target_key = matches[0]
                A = lora_sd[f"{prefix}.lora_A"]
                B = lora_sd[f"{prefix}.lora_B"]
                patches[target_key] = LoRAAdapter(target_key, (B, A, alpha, None, None, None))
            if patches:
                out_clip.add_patches(patches, strength_patch=1.0)

        orig_process_tokens = te_mod.process_tokens
        last_embeds_info = [[]]

        def wrapped_process_tokens(self_te, tokens_in, device, **kwargs):
            embeds, attention_mask, num_tokens, embeds_info = orig_process_tokens(tokens_in, device, **kwargs)
            last_embeds_info[0] = embeds_info
            return embeds, attention_mask, num_tokens, embeds_info

        te_mod.process_tokens = MethodType(wrapped_process_tokens, te_mod)

        orig_tokenize = out_clip.tokenize
        orig_encode = out_clip.encode_from_tokens_scheduled

        def h3_tokenize(self, text, return_word_ids=False, minimax_ref_items=None, images=None, picture_label="<Picture %d>: ", **kwargs):
            ref_images = []
            prefix_parts = []
            if minimax_ref_items:
                img_idx = 0
                for item in minimax_ref_items:
                    if item.get("type") == "image" and item.get("data") is not None:
                        img_idx += 1
                        ref_images.append(item["data"])
                        prefix_parts.append(f"<Picture {img_idx}>: <|vision_start|><|image_pad|><|vision_end|>")
            elif images:
                for i, img in enumerate(images, 1):
                    ref_images.append(img)
                    lbl = picture_label % i if "%d" in picture_label else picture_label
                    prefix_parts.append(f"{lbl}<|vision_start|><|image_pad|><|vision_end|>")
            h3_text = "".join(prefix_parts) + (text or "")
            tokens = orig_tokenize(
                h3_text,
                return_word_ids=return_word_ids,
                llama_template="{}",
                images=ref_images,
                prevent_empty_text=True,
                **kwargs,
            )
            return tokens

        def h3_encode_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
            last_embeds_info[0] = []
            cond_list = orig_encode(tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar)
            out_list = []
            for h, extra in cond_list:
                tags = torch.ones(h.shape[1], dtype=torch.long)
                for e in last_embeds_info[0]:
                    if e.get("type") == "image":
                        tags[max(0, e["index"] - 1):min(h.shape[1], e["index"] + e["size"] + 1)] = 0
                dev = h.device
                z = (h.float() - proj["mean_in"].to(dev)) / proj["std_in"].to(dev)
                with torch.no_grad():
                    y = z @ proj["W"].to(dev)
                    if mlp is not None:
                        y = y + mlp.to(dev)(z)
                proj_h = (y * proj["std_out"].to(dev) + proj["mean_out"].to(dev)).to(h.dtype)
                new_extra = dict(extra)
                new_extra["pooled_output"] = torch.zeros((proj_h.shape[0], 5120), dtype=proj_h.dtype, device=proj_h.device)
                new_extra["minimax_token_tags"] = tags.unsqueeze(0)
                out_list.append([proj_h, new_extra])
            return out_list

        out_clip.tokenize = MethodType(h3_tokenize, out_clip)
        out_clip.encode_from_tokens_scheduled = MethodType(h3_encode_scheduled, out_clip)
        return (out_clip,)


class Qwen35H3ClipLoader:
    """ComfyUI node that loads Qwen3.5-9B + a trained MiniMax-H3 projection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen3.5-9B", "tooltip": "Hugging Face repo ID or local path to Qwen/Qwen3.5-9B."}),
                "projection_file": ("STRING", {"default": "minimax_h3_qwen3.5_9b_projection.safetensors", "tooltip": "4096->5120 projection weights mapping Qwen3.5-9B layer 20 to MiniMax-H3."}),
                "use_lora": ("BOOLEAN", {"default": True, "tooltip": "Enable the trained rank-16 LoRA (visual.merger + layers 12-19) to boost <Picture 1..N> reference identity fidelity and prompt alignment."}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Automatically download missing projection/LoRA weights from Alissonerdx/Qwen3.5-9B-Minimax-H3."}),
            },
            "optional": {
                "lora_file": ("STRING", {"default": "minimax_h3_qwen3.5_9b_lora.safetensors", "tooltip": "LoRA weights file (only used when use_lora is enabled)."}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"

    def load(self, model_path: str, projection_file: str, use_lora: bool = True, auto_download: bool = True, lora_file: str = ""):
        proj_path = _resolve_model_file(projection_file, "minimax_h3_qwen3.5_9b_projection.safetensors", auto_download)
        lora_path = _resolve_model_file(lora_file, "minimax_h3_qwen3.5_9b_lora.safetensors", auto_download) if (use_lora and (lora_file or "").strip()) else ""
        return (Qwen35H3Clip(model_path=model_path, projection_path=proj_path, ara_path=lora_path),)


NODE_CLASS_MAPPINGS = {
    "Qwen35H3Apply": Qwen35H3Apply,
    "Qwen35H3ClipLoader": Qwen35H3ClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35H3Apply": "Apply Qwen3.5-H3 Projection & LoRA",
    "Qwen35H3ClipLoader": "Qwen3.5-9B H3 Projection CLIP Loader (Standalone)",
}
