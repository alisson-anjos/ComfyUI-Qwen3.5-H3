# ComfyUI-Qwen3.5-H3

Use **Qwen/Qwen3.5-9B** as a multimodal (`<Picture 1..N>` + text) conditioning encoder for **MiniMax-H3 Ref2VA** in ComfyUI.

This node extracts intermediate layer representations (tap 20, unnormalized) from `Qwen/Qwen3.5-9B`, formats prompts and static image references using MiniMax-H3's exact layout (`<Picture i>: <vision block> ... <prompt>`, no chat template), applies a trained `4096 -> 5120` projection (and optional LoRA), and emits standard ComfyUI `CONDITIONING` with `minimax_token_tags`.

All downstream MiniMax-H3 nodes (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3SigmaShift`, `KSampler`, `VAEDecode`) work out of the box.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alisson-anjos/ComfyUI-Qwen3.5-H3.git
cd ComfyUI-Qwen3.5-H3
pip install -r requirements.txt
```

Restart ComfyUI.

## Node Usage

1. Load your Qwen3.5-9B text encoder (`qwen3.5_9b_bf16.safetensors`, `qwen3.5_9b_int8_convrot.safetensors`, or `qwen3.5_9b_w4a8_gptq.safetensors` from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders)) using ComfyUI's native **`Load CLIP`** (`CLIPLoader`).
2. Connect its `CLIP` output to **`Apply Qwen3.5-H3 Projection & LoRA`** (`Qwen35H3Apply`, under `conditioning/minimax`):
   - **`use_lora`**: `True` *(enables the trained rank-16 LoRA on `visual.merger` + layers 12–19)*
   - **`auto_download`**: `True` *(automatically downloads `minimax_h3_qwen3.5_9b_projection.safetensors` and `minimax_h3_qwen3.5_9b_lora.safetensors` from [Alissonerdx/Qwen3.5-9B-Minimax-H3](https://huggingface.co/Alissonerdx/Qwen3.5-9B-Minimax-H3))*
3. Connect the `CLIP` output directly to ComfyUI's official **`MiniMax H3 Reference to Video`** (`MiniMaxH3ReferenceToVideo`) node.

## What is `minimax_h3_qwen3.5_9b_lora.safetensors`?

It is a **rank-16 LoRA (34 linear modules)** trained **jointly** with `minimax_h3_qwen3.5_9b_projection.safetensors` against MiniMax-H3's `Qwen3-VL-32B` teacher:
- **`model.visual.merger` (`linear_fc1`, `linear_fc2`)**: adapts how `<Picture 1..N>` reference image patches are projected into the language stream, significantly improving reference/identity fidelity (`vision` cosine `0.7873 -> 0.8867`).
- **Language layers `12–19` (`o_proj`, `gate_proj`, `up_proj`, `down_proj`, `out_proj`)**: aligns the hidden states right before the layer-20 tap (`text` cosine `0.8919 -> 0.9492`).

At load time, `Qwen35H3ClipLoader` merges the LoRA directly into the linear weights (`W += (alpha / r) * B @ A`), adding **zero runtime overhead**.

## Results: Projection Only vs Joint LoRA + Projection (Epoch 3)

| Variant | Text Cosine | Vision (`refs`) Cosine | Overall Score | Video Mean PSNR | Video Frame Corr |
|---|---|---|---|---|---|
| Projection Only (No LoRA) | `0.8919` | `0.7873` | `0.8396` | `12.877 dB` | `0.4410` |
| Joint LoRA + Projection — Epoch 1 | `0.9417` | `0.8801` | `0.9109` | — | — |
| Joint LoRA + Projection — Epoch 2 | `0.9492` | `0.8867` | `0.9180` | — | — |
| **Joint LoRA + Projection — Epoch 3** | **`0.9507`** | **`0.8879`** | **`0.9193`** | **`14.764 dB`** | **`0.6310`** |

### Visual Examples (With Epoch 3 LoRA + Projection, Clean H3 Prompts)

Each comparison video shows the `<Picture 1..N>` references at the top, **Native H3 (32B INT8)** on the left, **Qwen3.5-9B + Projection + LoRA** on the right, and the clean H3 prompt at the bottom:

#### Case 01 (1 reference — PSNR `18.353 dB`, Frame Corr `0.8359`)
- [Side-by-side comparison video (`case_01_proper_comparison.mp4`)](showcase/videos/case_01_proper_comparison.mp4)

#### Case 02 (2 references — PSNR `14.858 dB`, Frame Corr `0.8223`)
- [Side-by-side comparison video (`case_02_proper_comparison.mp4`)](showcase/videos/case_02_proper_comparison.mp4)

#### Case 03 (3 references — PSNR `14.743 dB`, Frame Corr `0.3852`)
- [Side-by-side comparison video (`case_03_proper_comparison.mp4`)](showcase/videos/case_03_proper_comparison.mp4)

#### Case 04 (4 references — PSNR `11.104 dB`, Frame Corr `0.4808`)
- [Side-by-side comparison video (`case_04_proper_comparison.mp4`)](showcase/videos/case_04_proper_comparison.mp4)

### Baseline Examples (Projection Only, No LoRA)

#### Case 01 (1 reference — PSNR `14.165 dB`, Frame Corr `0.4378`)
- [Side-by-side comparison video (`case_01_comparison.mp4`)](showcase/videos/case_01_comparison.mp4)

| Native H3 (32B INT8) | Qwen3.5-9B + Projection (No LoRA) |
|---|---|
| ![Case 01 Native](showcase/images/case_01_native_h3_5x4.png) | ![Case 01 Qwen3.5](showcase/images/case_01_qwen35_projected_5x4.png) |

#### Case 02 (2 references — PSNR `12.538 dB`, Frame Corr `0.6657`)
- [Side-by-side comparison video (`case_02_comparison.mp4`)](showcase/videos/case_02_comparison.mp4)

| Native H3 (32B INT8) | Qwen3.5-9B + Projection (No LoRA) |
|---|---|
| ![Case 02 Native](showcase/images/case_02_native_h3_5x4.png) | ![Case 02 Qwen3.5](showcase/images/case_02_qwen35_projected_5x4.png) |

#### Case 03 (3 references — PSNR `15.498 dB`, Frame Corr `0.5220`)
- [Side-by-side comparison video (`case_03_comparison.mp4`)](showcase/videos/case_03_comparison.mp4)

| Native H3 (32B INT8) | Qwen3.5-9B + Projection (No LoRA) |
|---|---|
| ![Case 03 Native](showcase/images/case_03_native_h3_5x4.png) | ![Case 03 Qwen3.5](showcase/images/case_03_qwen35_projected_5x4.png) |

## Evidence Files

- [`evidence/validation_metrics.json`](evidence/validation_metrics.json)
- [`evidence/video_comparison.json`](evidence/video_comparison.json)
- [`evidence/lora_video_comparison.json`](evidence/lora_video_comparison.json)
