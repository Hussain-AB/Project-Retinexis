"""
Project Retinexis — Gradio Web Application
===========================================

Retinal fundus image analysis for cardiovascular disease (CVD) risk
and hypertension detection using deep learning models.

Team Assignments:
    Person 1 (Core UI):     Sections marked [CORE]
    Person 2 (Models):      Sections marked [MODELS]
    Person 3 (Extras):      Sections marked [EXTRAS]


"""

import gradio as gr
import numpy as np
import json
import time
from pathlib import Path
from PIL import Image
from datetime import datetime
import torch
import torch.nn as nn
import cv2

# [MODELS] Model registry & inference
# ──────────────────────────────────────────────
# # TODO: Replace with actual imports once model pipeline is integrated
# from src.models import create_model, MODEL_REGISTRY
# from src.data.transforms import build_transforms

# Available models per task. Update these as models finish training.
# Format: { "Display Name": {"architecture": str, "variant": str, "description": str} }
TASK_MODELS = {
    "CVD Risk": {
        "RETFound-DINOv2": {
            "architecture": "retfound_dinov2",
            "variant": "default",
            "description": "Retinal specialist — top performer on CVD risk",
        },
        "DINOv2-Large": {
            "architecture": "dinov2",
            "variant": "large",
            "description": "General-purpose ViT — capacity-matched generalist",
        },
        "EfficientNet-B2": {
            "architecture": "efficientnet",
            "variant": "b2",
            "description": "CNN baseline — lightweight and fast",
        },
    },
    "Hypertension": {
        "RETFound-DINOv2": {
            "architecture": "retfound_dinov2",
            "variant": "default",
            "description": "Retinal specialist — top performer on hypertension",
        },
        "DINOv2-Large": {
            "architecture": "dinov2",
            "variant": "large",
            "description": "General-purpose ViT — capacity-matched generalist",
        },
        "ConvNeXt V2-Base": {
            "architecture": "convnextv2",
            "variant": "base",
            "description": "Modern CNN — BRSET literature baseline",
        },
    },
}

# TODO: Update with actual metadata fields per dataset.
#   - CVD Risk (CIMT dataset): likely age, gender, thickness
#   - Hypertension (BRSET dataset): TBD — check with BRSET team
TASK_METADATA_FIELDS = {
    "CVD Risk": {},
    "Hypertension": {},
}


def get_model_choices(task: str) -> list[str]:
    """Return model display names for the selected task."""
    return list(TASK_MODELS.get(task, {}).keys())


def get_model_description(task: str, model_name: str) -> str:
    """Return the one-line description for a model."""
    models = TASK_MODELS.get(task, {})
    if model_name in models:
        return models[model_name]["description"]
    return ""



# [MODELS] Inference pipeline


def load_model(task: str, model_name: str):
    """
    Load a trained model checkpoint.

    TODO [Person 2]: Replace stub with actual model loading:
        1. Look up architecture + variant from TASK_MODELS
        2. Call create_model(architecture, variant=variant, pretrained=True)
        3. Load the trained checkpoint weights
        4. Set model to eval mode
        5. Return (model, preprocessing_config)

    Checkpoints should live in: outputs/{task}/{model_name}/fold_0/checkpoints/best.pt
    (or whichever fold / ensemble strategy we decide on for deployment)
    """
    # STUB — remove when implementing
    return None, None


def preprocess_image(image: np.ndarray, preprocessing_config) -> np.ndarray:
    """
    Preprocess uploaded image according to model's preprocessing_config.

    TODO [Person 2]: Replace stub with actual preprocessing:
        1. Read input_size, mean, std from preprocessing_config
        2. Resize image to model's native resolution
        3. Apply normalization
        4. Convert to tensor
        5. Return preprocessed tensor

    The model's preprocessing_config (from model.preprocessing_config) defines:
        - input_size: e.g. (260, 260) for EfficientNet, (224, 224) for ViTs
        - mean / std: channel-wise normalization stats
        - crop_strategy: center crop vs resize
    """
    # STUB — remove when implementing
    return image


def run_inference(model, image_tensor, metadata: dict | None = None) -> dict:
    """
    Run model inference on a preprocessed image.

    TODO [Person 2]: Replace stub with actual inference:
        1. Move tensor to device (cuda if available)
        2. Run model.forward() or model(image_tensor)
        3. Apply sigmoid to get probability
        4. Return {"probability": float, "prediction": int, "confidence": float}

    Args:
        model: Loaded model in eval mode
        image_tensor: Preprocessed image tensor
        metadata: Optional patient metadata dict (age, sex, etc.)

    Returns:
        dict with keys: probability, prediction, confidence
    """
    # STUB — returns dummy prediction. Remove when implementing.
    dummy_prob = float(np.random.uniform(0.2, 0.8))
    return {
        "probability": dummy_prob,
        "prediction": int(dummy_prob > 0.5),
        "confidence": abs(dummy_prob - 0.5) * 2,
    }



# [CORE] Main prediction handler


def predict(image, task, model_name, age, sex):
    """
    Main prediction function wired to the Gradio interface.

    [Person 1] owns the orchestration logic.
    [Person 2] owns the model/inference functions called within.
    """
    # --- Validation ---
    if image is None:
        raise gr.Error("Please upload a retinal fundus image.")
    if not model_name:
        raise gr.Error("Please select a model.")

    # --- Collect metadata ---
    metadata = {}
    if age is not None and age != "":
        metadata["age"] = float(age)
    if sex is not None and sex != "":
        metadata["sex"] = int(sex)

    # --- Load model ---
    model, preprocessing_config = load_model(task, model_name)

    # --- Preprocess ---
    image_tensor = preprocess_image(np.array(image), preprocessing_config)

    # --- Inference ---
    result = run_inference(model, image_tensor, metadata)

    # --- Format output ---
    prob = result["probability"]
    pred = result["prediction"]
    confidence = result["confidence"]

    task_label = "CVD Risk" if task == "CVD Risk" else "Hypertension"
    risk_level = "Positive" if pred == 1 else "Negative"

    summary = (
        f"## {task_label} Prediction\n\n"
        f"**Risk classification:** {risk_level}\n\n"
        f"**Probability:** {prob:.1%}\n\n"
        f"**Confidence:** {confidence:.1%}\n\n"
        f"**Model:** {model_name}\n\n"
        f"---\n"
        f"*This is a research tool — not for clinical diagnosis.*"
    )

    # --- FHIR output ---
    fhir_observation = generate_fhir_observation(
        task=task,
        model_name=model_name,
        probability=prob,
        prediction=pred,
        metadata=metadata,
    )

    return summary, json.dumps(fhir_observation, indent=2)



# [MODELS] FHIR output generation


def generate_fhir_observation(
    task: str,
    model_name: str,
    probability: float,
    prediction: int,
    metadata: dict | None = None,
) -> dict:
    """
    Generate a FHIR R4 Observation resource from the prediction.

    TODO [Person 2]: 
    The current structure follows FHIR R4 Observation but uses placeholder codes.
    """
    now = datetime.utcnow().isoformat() + "Z"

    task_codes = {
        "CVD Risk": {
            "system": "http://loinc.org",
            "code": "PLACEHOLDER-CVD",
            "display": "Cardiovascular disease risk assessment from retinal imaging",
        },
        "Hypertension": {
            "system": "http://loinc.org",
            "code": "PLACEHOLDER-HTN",
            "display": "Hypertension risk assessment from retinal imaging",
        },
    }

    code_info = task_codes.get(task, task_codes["CVD Risk"])

    observation = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "imaging",
                        "display": "Imaging",
                    }
                ]
            }
        ],
        "code": {"coding": [code_info]},
        "effectiveDateTime": now,
        "valueQuantity": {
            "value": round(probability, 4),
            "unit": "probability",
            "system": "http://unitsofmeasure.org",
            "code": "1",
        },
        "interpretation": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": "POS" if prediction == 1 else "NEG",
                        "display": "Positive" if prediction == 1 else "Negative",
                    }
                ]
            }
        ],
        "method": {
            "text": f"Deep learning model: {model_name} (Project Retinexis)"
        },
        "note": [
            {
                "text": "AI-generated prediction from retinal fundus image analysis. Not for clinical use without physician review."
            }
        ],
    }

    # Attach patient metadata if provided
    if metadata:
        extensions = []
        if "age" in metadata:
            extensions.append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/patient-age",
                    "valueInteger": int(metadata["age"]),
                }
            )
        if "sex" in metadata:
            extensions.append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/patient-sex",
                    "valueCode": "male" if metadata["sex"] == 1 else "female",
                }
            )
        if extensions:
            observation["extension"] = extensions

    return observation



# [EXTRAS] Attention / saliency maps

def _architecture_type(task: str, model_name: str) -> str:
    """Return 'vit' for Vision Transformers or 'cnn' for convolutional models."""
    arch = TASK_MODELS.get(task, {}).get(model_name, {}).get("architecture", "").lower()
    if any(x in arch for x in ("vit", "dino", "retfound", "transformer")):
        return "vit"
    if arch:
        return "cnn"
    raise gr.Error(f"Unknown architecture for model '{model_name}'. Check TASK_MODELS.")


_GRADCAM_TARGET_LAYER_MAP = {
    "EfficientNet-B2": lambda m: [m.features[-1]],
    "ConvNeXt V2-Base": lambda m: [m.stages[-1]],
}


def _get_target_layers(model: nn.Module, model_name: str = "") -> list:
    """
    Return GradCAM target layer(s) for a CNN model.
    Uses an explicit per-model mapping first, then falls back to scanning
    all modules for the last Conv2d in the network.
    """
    if model_name in _GRADCAM_TARGET_LAYER_MAP:
        return _GRADCAM_TARGET_LAYER_MAP[model_name](model)
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise gr.Error("Could not find a convolutional layer for GradCAM.")
    return [last_conv]


def _to_batched_tensor(image_tensor) -> torch.Tensor:
    """
    Convert the output of preprocess_image() to a (1, C, H, W) float tensor.
    Does NOT re-normalise — assumes preprocess_image already handled that.
    """
    if isinstance(image_tensor, np.ndarray):
        tensor = torch.from_numpy(image_tensor.copy()).float()
    elif isinstance(image_tensor, torch.Tensor):
        tensor = image_tensor.float()
    else:
        raise gr.Error(f"Unsupported preprocessed image type: {type(image_tensor)}")

    if tensor.dim() == 3:
        # (H, W, C) → (C, H, W); already (C, H, W) stays as-is
        if tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise gr.Error(f"Unexpected tensor shape from preprocess_image: {tuple(tensor.shape)}")

    return tensor


def _apply_gradcam(
    model: nn.Module,
    image_tensor,
    original_image: Image.Image,
    model_name: str = "",
    opacity: float = 0.5,
) -> tuple[Image.Image, str]:
    """
    Apply GradCAM to a CNN model (e.g. EfficientNet-B2, ConvNeXt V2-Base).
    Requires pytorch-grad-cam (pip install grad-cam).
    Returns (overlay PIL Image, method name string).
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    device = next(model.parameters()).device
    target_layers = _get_target_layers(model, model_name)
    tensor = _to_batched_tensor(image_tensor).to(device)

    model.eval()
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor)[0]  # (H, W) in [0, 1]

    opacity = float(np.clip(opacity, 0.0, 1.0))
    orig_rgb = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
    orig_resized = cv2.resize(orig_rgb, (grayscale_cam.shape[1], grayscale_cam.shape[0]))
    overlay_arr = show_cam_on_image(orig_resized, grayscale_cam, use_rgb=True, image_weight=1 - opacity)

    return Image.fromarray(overlay_arr), "GradCAM"


def _apply_attention_rollout(
    model: nn.Module,
    image_tensor,
    original_image: Image.Image,
    opacity: float = 0.5,
    discard_ratio: float = 0.1,  
) -> tuple[Image.Image, str]:
    """
    Attention rollout for ViT models (e.g. RETFound-DINOv2, DINOv2-Large).

    Registers forward hooks on nn.MultiheadAttention layers to collect attention
    weights, folds them across all layers using the rollout algorithm, and uses
    the CLS token's attention to each patch as the saliency map.

    Note: works when the model uses nn.MultiheadAttention and returns weights in
    output[1]. Models with fused or custom attention layers may need
    architecture-specific rollout support and will raise a clear error.

    Returns (overlay PIL Image, method name string).
    """
    attention_weights = []
    hooks = []

    def _hook(module, input, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_hook(_hook))

    device = next(model.parameters()).device
    tensor = _to_batched_tensor(image_tensor).to(device)

    model.eval()
    try:
        with torch.no_grad():
            model(tensor)
    finally:
        for hook in hooks:
            hook.remove()

    if not attention_weights:
        raise gr.Error(
            "Could not extract attention weights from this model. "
            "The architecture may need custom attention rollout support."
        )

    # Rollout: fold all attention layers together
    seq_len = attention_weights[0].shape[-1]
    result = torch.eye(seq_len, dtype=attention_weights[0].dtype, device=attention_weights[0].device)
    for attn in attention_weights:
        attn_mat = attn[0].clone()

        # Handle (heads, seq, seq) or (seq, seq) — average over heads if needed
        if attn_mat.dim() == 3:
            attn_mat = attn_mat.mean(dim=0)   # (seq, seq)
        elif attn_mat.dim() != 2:
            raise gr.Error(f"Unexpected attention weight shape: {tuple(attn.shape)}")

        if discard_ratio > 0:
            k = max(1, int(discard_ratio * attn_mat.numel()))
            threshold = attn_mat.flatten().kthvalue(k).values
            attn_mat[attn_mat < threshold] = 0
        attn_mat = attn_mat + torch.eye(attn_mat.shape[0], dtype=attn_mat.dtype, device=attn_mat.device)
        attn_mat = attn_mat / attn_mat.sum(dim=-1, keepdim=True)
        result = torch.mm(attn_mat, result)

    # CLS token (row 0) attention to patch tokens (cols 1:)
    mask = result[0, 1:].cpu().numpy()

    grid = int(len(mask) ** 0.5)
    if grid * grid != len(mask):
        raise gr.Error(
            f"Cannot reshape {len(mask)} patch tokens into a square grid. "
            "The model may use a non-square patch layout."
        )
    mask = mask.reshape(grid, grid)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    orig_rgb = np.array(original_image.convert("RGB"))
    h, w = orig_rgb.shape[:2]
    mask_resized = cv2.resize(mask, (w, h))

    opacity = float(np.clip(opacity, 0.0, 1.0))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_arr = cv2.addWeighted(orig_rgb, 1 - opacity, heatmap_rgb, opacity, 0)

    return Image.fromarray(overlay_arr), "Attention Rollout"


def generate_saliency_and_store(image, task, model_name):
    """
    Wrapper around generate_saliency_map that also returns the original image
    so the UI can toggle between overlay and original.

    Returns:
        tuple: (overlay, info, overlay_for_state, original_image, showing_overlay=True, toggle_label)
    """
    overlay, info = generate_saliency_map(image, task, model_name)
    return overlay, info, overlay, image, True, "Show Original"


def toggle_saliency_view(showing_overlay, overlay_img, original_img):
    """Switch the displayed image between overlay and original."""
    if showing_overlay:
        return original_img, False, "Show Overlay"
    else:
        return overlay_img, True, "Show Original"


def generate_saliency_map(image, task, model_name):
    """
    Generate a heatmap overlay showing which image regions influenced the model's prediction.

    Routes to GradCAM for CNN architectures (EfficientNet-B2, ConvNeXt V2-Base)
    and attention rollout for ViT architectures (RETFound-DINOv2, DINOv2-Large).
    Overlays the heatmap on the original image at 50% opacity.

    Returns:
        tuple: (overlay_image PIL, info markdown string)
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    model, preprocessing_config = load_model(task, model_name)

    # Graceful fallback until Person 2 integrates model loading
    if model is None:
        arch_type = _architecture_type(task, model_name)
        method = "Attention Rollout" if arch_type == "vit" else "GradCAM"
        info = (
            f"**Model not yet loaded** — saliency maps will be available once "
            f"the model pipeline is integrated.\n\n"
            f"Method that will be used: **{method}** ({model_name})"
        )
        return image, info

    arch_type = _architecture_type(task, model_name)
    image_tensor = preprocess_image(np.array(image), preprocessing_config)

    if arch_type == "vit":
        overlay, method = _apply_attention_rollout(model, image_tensor, image)
    else:
        overlay, method = _apply_gradcam(model, image_tensor, image, model_name=model_name)

    info = (
        f"**Method:** {method}  \n"
        f"**Model:** {model_name}  \n\n"
        f"*Highlighted regions indicate areas most influential for the prediction.*"
    )
    return overlay, info


# [EXTRAS] Model comparison


def compare_models(image, task, age, sex):
    """
    Run all available models for a task on the same image and compare results.

    Loads each model, preprocesses the image with its config, runs inference,
    and returns a Markdown table with columns:
        Model | Type | Prediction | Probability | Confidence

    Returns:
        str: Markdown-formatted comparison table
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    metadata = {}
    try:
        if age is not None and age != "":
            metadata["age"] = float(age)
        if sex is not None and sex != "":
            metadata["sex"] = int(sex)
    except ValueError as e:
        raise gr.Error(f"Invalid metadata input: {e}")

    model_names = get_model_choices(task)
    if not model_names:
        raise gr.Error("No models available for the selected task.")

    img_array = np.array(image)
    rows = []  # (name, arch, pred, prob, conf) for successful runs
    error_rows = []  # (name, error_message) for failed runs

    for name in model_names:
        try:
            model, preprocessing_config = load_model(task, name)

            if model is None:
                error_rows.append((name, "Model not yet loaded"))
                continue

            image_tensor = preprocess_image(img_array, preprocessing_config)
            result = run_inference(model, image_tensor, metadata)

            prob = result["probability"]
            pred = "Positive" if result["prediction"] == 1 else "Negative"
            conf = result["confidence"]
            arch = _architecture_type(task, name).upper()
            rows.append((name, arch, pred, prob, conf))

        except gr.Error:
            raise  # let Gradio errors propagate normally
        except Exception as e:
            error_rows.append((name, str(e)))

    table_lines = [
        f"### {task} — Model Comparison\n",
        "| Model | Type | Prediction | Probability | Confidence |",
        "|-------|------|-----------|-------------|------------|",
    ]

    if rows:
        best = max(rows, key=lambda r: r[4])
        for name, arch, pred, prob, conf in rows:
            marker = " [best]" if name == best[0] else ""
            table_lines.append(f"| {name}{marker} | {arch} | {pred} | {prob:.1%} | {conf:.1%} |")

    for name, error_msg in error_rows:
        table_lines.append(f"| {name} | — | — | — | {error_msg} |")

    if rows:
        table_lines.append(f"\nHighest confidence: **{best[0]}** ({best[4]:.1%})")

    if not rows and error_rows:
        table_lines.append("\n*No models produced a valid result.*")
    elif error_rows:
        table_lines.append(
            f"\n*{len(error_rows)} model(s) could not run — see rows above.*"
        )

    return "\n".join(table_lines)


# [EXTRAS] Dual eye input


def predict_dual_eye(image_left, image_right, task, model_name, age, sex):
    """
    Run prediction on both eyes and return results for side-by-side display.

    Reflects the clinical workflow where clinicians capture fundus images of
    both eyes during routine examinations. Inference runs independently on
    each eye using the same model and patient metadata.

    Returns:
        tuple: (left_summary, right_summary, combined_summary)
    """
    if image_left is None and image_right is None:
        raise gr.Error("Please upload at least one retinal fundus image.")

    def _run(label, img):
        if img is None:
            return f"*No image uploaded for {label}.*"
        summary, _ = predict(img, task, model_name, age, sex)
        return f"### {label}\n{summary}"

    left_summary = _run("Left Eye", image_left)
    right_summary = _run("Right Eye", image_right)
    combined_summary = f"{left_summary}\n\n---\n\n{right_summary}"
    return left_summary, right_summary, combined_summary



# [EXTRAS] Session history helpers


def clear_history() -> tuple:
    """Reset session history to empty state."""
    return [], [], "*No predictions yet in this session.*"


def _history_gallery_items(history: list) -> list:
    """Return (image, caption) pairs for gr.Gallery."""
    return [(entry["image"], entry["caption"]) for entry in history]


def _history_markdown(history: list) -> str:
    """Render history entries as a markdown list (most recent first)."""
    if not history:
        return "*No predictions yet in this session.*"
    lines = ["## Session History\n"]
    for entry in reversed(history):
        lines.append(f"**{entry['caption']}**\n\n{entry['summary']}\n\n---\n")
    return "\n".join(lines)


def predict_and_log(image, task, model_name, age, sex, history):
    """Wrap predict() and append the result to session history."""
    history = history or []
    summary, fhir = predict(image, task, model_name, age, sex)
    caption = f"{task} · {model_name} · {datetime.now().strftime('%H:%M:%S')}"
    updated = history + [{"image": image, "caption": caption, "summary": summary}]
    return summary, fhir, updated, _history_gallery_items(updated), _history_markdown(updated)


def dual_predict_and_log(image_left, image_right, task, model_name, age, sex, history):
    """Wrap predict_dual_eye() and append each eye as a separate history entry."""
    history = history or []
    left_summary, right_summary, combined_summary = predict_dual_eye(
        image_left, image_right, task, model_name, age, sex
    )
    updated = list(history)
    ts = datetime.now().strftime("%H:%M:%S")
    if image_left is not None:
        updated.append({
            "image": image_left,
            "caption": f"{task} · {model_name} · Left Eye · {ts}",
            "summary": left_summary,
        })
    if image_right is not None:
        updated.append({
            "image": image_right,
            "caption": f"{task} · {model_name} · Right Eye · {ts}",
            "summary": right_summary,
        })
    return (
        left_summary, right_summary, combined_summary,
        updated, _history_gallery_items(updated), _history_markdown(updated),
    )


# [CORE] Dynamic UI updates


def on_task_change(task):
    """Update model dropdown choices when task changes."""
    choices = get_model_choices(task)
    default = choices[0] if choices else None
    description = get_model_description(task, default) if default else ""
    return gr.update(choices=choices, value=default), description


def on_model_change(task, model_name):
    """Update model description when model selection changes."""
    return get_model_description(task, model_name)



# [CORE] Build Gradio interface


CUSTOM_CSS = """
.main-title {
    text-align: center;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.disclaimer {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #92400e;
    margin-top: 1rem;
}
"""

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Project Retinexis",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    ) as app:

        # ── Header ──
        gr.Markdown(
            "<h1 class='main-title'>Project Retinexis</h1>"
            "<p class='subtitle'>Deep Learning for Non-Invasive Systemic Disease Detection from Retinal Images</p>"
        )

        # Session history state (per-user, resets on page refresh)
        session_history = gr.State([])

        with gr.Tabs() as tabs:

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 1: Single Prediction [CORE]
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Prediction", id="prediction"):
                with gr.Row():

                    # -- Left column: inputs --
                    with gr.Column(scale=1):
                        task_dropdown = gr.Dropdown(
                            choices=["CVD Risk", "Hypertension"],
                            value="CVD Risk",
                            label="Task",
                            info="Select the prediction task",
                        )
                        model_dropdown = gr.Dropdown(
                            choices=get_model_choices("CVD Risk"),
                            value=get_model_choices("CVD Risk")[0],
                            label="Model",
                            info="Select which trained model to use",
                        )
                        model_description = gr.Textbox(
                            value=get_model_description("CVD Risk", get_model_choices("CVD Risk")[0]),
                            label="Model Info",
                            interactive=False,
                            lines=1,
                        )

                        image_input = gr.Image(
                            label="Upload Retinal Fundus Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=300,
                        )

                        # Patient metadata
                        with gr.Accordion("Patient Metadata (optional)", open=False):
                            age_input = gr.Number(
                                label="Patient Age",
                                precision=0,
                                minimum=0,
                                maximum=120,
                            )
                            sex_input = gr.Dropdown(
                                choices=[("Female", 0), ("Male", 1)],
                                label="Sex",
                                type="index",
                            )

                        predict_btn = gr.Button(
                            "Run Prediction",
                            variant="primary",
                            size="lg",
                        )

                    # -- Right column: outputs --
                    with gr.Column(scale=1):
                        result_output = gr.Markdown(
                            value="*Upload an image and click 'Run Prediction' to see results.*",
                            label="Prediction Result",
                        )
                        with gr.Accordion("FHIR Observation (JSON)", open=False):
                            fhir_output = gr.Code(
                                language="json",
                                label="FHIR R4 Observation",
                                lines=20,
                            )
                            fhir_download_btn = gr.Button(
                                "Download FHIR JSON",
                                size="sm",
                            )
                            fhir_file = gr.File(
                                label="Download",
                                visible=False,
                            )

                        gr.Markdown(
                            "<div class='disclaimer'>"
                            "This tool is for research purposes only. Predictions should not be used "
                            "for clinical diagnosis without review by a qualified physician."
                            "</div>"
                        )

                # ── Wire up events ──

                task_dropdown.change(
                    fn=on_task_change,
                    inputs=[task_dropdown],
                    outputs=[model_dropdown, model_description],
                )
                model_dropdown.change(
                    fn=on_model_change,
                    inputs=[task_dropdown, model_dropdown],
                    outputs=[model_description],
                )
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 2: Model Comparison [EXTRAS]
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Compare Models", id="compare"):
                gr.Markdown(
                    "### Model Comparison\n"
                    "Run all available models on the same image to compare predictions."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_task = gr.Dropdown(
                            choices=["CVD Risk", "Hypertension"],
                            value="CVD Risk",
                            label="Task",
                        )
                        compare_image = gr.Image(
                            label="Upload Retinal Fundus Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=300,
                        )
                        with gr.Accordion("Patient Metadata (optional)", open=False):
                            compare_age = gr.Number(label="Patient Age", precision=0)
                            compare_sex = gr.Dropdown(
                                choices=[("Female", 0), ("Male", 1)],
                                label="Sex",
                                type="index",
                            )
                        compare_btn = gr.Button(
                            "Compare All Models",
                            variant="primary",
                        )

                    with gr.Column(scale=1):
                        compare_output = gr.Markdown(
                            value="*Upload an image and click 'Compare All Models' to see results.*"
                        )

                compare_btn.click(
                    fn=compare_models,
                    inputs=[compare_image, compare_task, compare_age, compare_sex],
                    outputs=[compare_output],
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 3: Dual Eye Input [EXTRAS]
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Dual Eye", id="dual"):
                gr.Markdown(
                    "### Dual Eye Analysis\n"
                    "Upload left and right eye images for side-by-side prediction."
                )
                with gr.Row():
                    with gr.Column():
                        dual_task = gr.Dropdown(
                            choices=["CVD Risk", "Hypertension"],
                            value="CVD Risk",
                            label="Task",
                        )
                        dual_model = gr.Dropdown(
                            choices=get_model_choices("CVD Risk"),
                            value=get_model_choices("CVD Risk")[0],
                            label="Model",
                        )
                        with gr.Row():
                            left_eye_input = gr.Image(
                                label="Left Eye",
                                type="pil",
                                sources=["upload", "clipboard"],
                                height=250,
                            )
                            right_eye_input = gr.Image(
                                label="Right Eye",
                                type="pil",
                                sources=["upload", "clipboard"],
                                height=250,
                            )
                        with gr.Accordion("Patient Metadata (optional)", open=False):
                            dual_age = gr.Number(label="Patient Age", precision=0)
                            dual_sex = gr.Dropdown(
                                choices=[("Female", 0), ("Male", 1)],
                                label="Sex",
                                type="index",
                            )
                        dual_btn = gr.Button(
                            "Analyse Both Eyes",
                            variant="primary",
                        )

                with gr.Row():
                    dual_output_left = gr.Markdown(
                        value="*Left eye result will appear here.*",
                        label="Left Eye Result",
                    )
                    dual_output_right = gr.Markdown(
                        value="*Right eye result will appear here.*",
                        label="Right Eye Result",
                    )
                with gr.Accordion("Combined Summary", open=False):
                    dual_output_combined = gr.Markdown(
                        value="*Combined summary will appear here.*",
                    )

                # Update model dropdown when task changes
                dual_task.change(
                    fn=lambda t: gr.update(choices=get_model_choices(t), value=get_model_choices(t)[0]),
                    inputs=[dual_task],
                    outputs=[dual_model],
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 4: Saliency / Attention [EXTRAS]
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("Saliency Map", id="saliency"):
                gr.Markdown(
                    "### Attention & Saliency Visualisation\n"
                    "See which regions of the retinal image the model focuses on."
                )
                with gr.Row():
                    with gr.Column():
                        sal_task = gr.Dropdown(
                            choices=["CVD Risk", "Hypertension"],
                            value="CVD Risk",
                            label="Task",
                        )
                        sal_model = gr.Dropdown(
                            choices=get_model_choices("CVD Risk"),
                            value=get_model_choices("CVD Risk")[0],
                            label="Model",
                        )
                        sal_image = gr.Image(
                            label="Upload Retinal Fundus Image",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=300,
                        )
                        sal_btn = gr.Button(
                            "Generate Saliency Map",
                            variant="primary",
                        )

                    with gr.Column():
                        sal_display = gr.Image(
                            label="Saliency Overlay",
                            height=300,
                        )
                        sal_toggle_btn = gr.Button(
                            "Show Original",
                            size="sm",
                            visible=False,
                        )
                        sal_info = gr.Markdown()

                # States for toggle
                sal_overlay_state = gr.State(None)
                sal_original_state = gr.State(None)
                sal_showing_overlay = gr.State(True)

                # Update model dropdown when task changes
                sal_task.change(
                    fn=lambda t: gr.update(choices=get_model_choices(t), value=get_model_choices(t)[0]),
                    inputs=[sal_task],
                    outputs=[sal_model],
                )

                sal_btn.click(
                    fn=generate_saliency_and_store,
                    inputs=[sal_image, sal_task, sal_model],
                    outputs=[sal_display, sal_info, sal_overlay_state, sal_original_state, sal_showing_overlay, sal_toggle_btn],
                )

                sal_toggle_btn.click(
                    fn=toggle_saliency_view,
                    inputs=[sal_showing_overlay, sal_overlay_state, sal_original_state],
                    outputs=[sal_display, sal_showing_overlay, sal_toggle_btn],
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TAB 5: Session History [EXTRAS]
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with gr.Tab("History", id="history"):
                gr.Markdown(
                    "### Session History\n"
                    "All predictions made during this session. Resets on page refresh."
                )
                history_gallery = gr.Gallery(
                    label="Uploaded Images",
                    columns=4,
                    height=200,
                    object_fit="cover",
                )
                history_md = gr.Markdown(
                    value="*No predictions yet in this session.*"
                )
                clear_history_btn = gr.Button("Clear History", variant="stop", size="sm")

                clear_history_btn.click(
                    fn=clear_history,
                    inputs=[],
                    outputs=[session_history, history_gallery, history_md],
                )

        # ── Cross-tab event wiring (requires all components to be defined first) ──

        predict_btn.click(
            fn=predict_and_log,
            inputs=[image_input, task_dropdown, model_dropdown, age_input, sex_input, session_history],
            outputs=[result_output, fhir_output, session_history, history_gallery, history_md],
        )

        dual_btn.click(
            fn=dual_predict_and_log,
            inputs=[left_eye_input, right_eye_input, dual_task, dual_model, dual_age, dual_sex, session_history],
            outputs=[dual_output_left, dual_output_right, dual_output_combined, session_history, history_gallery, history_md],
        )

    return app



# Entry point


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True to generate a public link
    )
