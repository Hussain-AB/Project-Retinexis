"""
Project Retinexis — Gradio Web Application
===========================================
Retinal fundus image analysis for cardiovascular disease (CVD) risk
and hypertension detection using deep learning models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import numpy as np
import cv2
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime
from huggingface_hub import hf_hub_download
from src.models import create_model
import base64


_HERE = os.path.dirname(os.path.abspath(__file__))
gr.set_static_paths(paths=[os.path.join(_HERE, "images")])




# ── Model cache ──────────────────────────────────────────────────────────────

_MODEL_CACHE = {}


# ── Model & checkpoint configuration ────────────────────────────────────────

TASK_MODELS = {
    "CVD Risk": {
        "RETFound-MAE": {
            "architecture": "retfound_mae",
            "variant": "default",
            "description": "Retinal specialist — top performer on CVD risk",
        },
        "ConvNeXt V2-Base": {
            "architecture": "convnextv2",
            "variant": "convnextv2_base",
            "description": "Modern CNN — second best on CVD risk",
        },
        "EfficientNet-B2": {
            "architecture": "efficientnet",
            "variant": "b2",
            "description": "CNN baseline — lightweight and fast",
        },
    },
    "Hypertension": {
        "DINOv3-Large": {
            "architecture": "dinov3",
            "variant": "large",
            "description": "Latest generalist ViT — top performer on hypertension",
        },
        "RETFound-MAE": {
            "architecture": "retfound_mae",
            "variant": "default",
            "description": "Retinal specialist — strong on hypertension",
        },
        "DINOv2-Large": {
            "architecture": "dinov2",
            "variant": "large",
            "description": "General-purpose ViT — capacity-matched generalist",
        },
    },
}

HF_CHECKPOINTS = {
    "CVD Risk": {
        "RETFound-MAE": "cimt/retfound_mae/best.pt",
        "ConvNeXt V2-Base": "cimt/convnextv2/best.pt",
        "EfficientNet-B2": "cimt/efficientnet/best.pt",
    },
    "Hypertension": {
        "DINOv3-Large": "hypertension_combined/dinov3/best.pt",
        "RETFound-MAE": "hypertension_combined/retfound_mae/best.pt",
        "DINOv2-Large": "hypertension_combined/dinov2/best.pt",
    },
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


# ── Inference pipeline ──────────────────────────────────────────────────────

def load_model(task: str, model_name: str):
    """Load a trained model checkpoint from Hugging Face, with caching."""
    cache_key = f"{task}:{model_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    hf_path = HF_CHECKPOINTS.get(task, {}).get(model_name)
    if hf_path is None:
        return None, None

    model_info = TASK_MODELS.get(task, {}).get(model_name, {})
    architecture = model_info.get("architecture")
    variant = model_info.get("variant")

    if variant and variant != "default":
        model = create_model(architecture, variant=variant, pretrained=False)
    else:
        model = create_model(architecture, pretrained=False)

    checkpoint_path = hf_hub_download(
        repo_id="Faisasa12/Project-Retinexis",
        filename=hf_path,
        token=os.environ.get("HF_TOKEN"),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Remove keys with shape mismatches (e.g. DINOv3 mask_token)
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key in model_state and state_dict[key].shape != model_state[key].shape:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, model.preprocessing_config)
    return model, model.preprocessing_config


# Task-specific preprocessing helpers 

def _remove_top_left_overlay_cimt(img, width=90, height=50, white_threshold=200):
    """CIMT: Remove text overlay from top-left corner (matches training)."""
    img_copy = img.copy()
    top_left = img_copy[:height, :width]
    if len(top_left.shape) == 3:
        gray = cv2.cvtColor(top_left, cv2.COLOR_RGB2GRAY)
    else:
        gray = top_left
    if np.sum(gray > white_threshold) > 0:
        img_copy[:height, :width] = np.array([6, 6, 6], dtype=img_copy.dtype)
    return img_copy


def _clahe_cimt(img):
    """CIMT: CLAHE on green channel only (matches training)."""
    r, g, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g)
    return cv2.merge((r, g_enhanced, b))







def preprocess_image(image: np.ndarray, preprocessing_config, task=None) -> torch.Tensor:
    """Preprocess uploaded image according to model's preprocessing_config."""
    if preprocessing_config is None:
        return image

    # Apply task-specific preprocessing (matches training pipeline)
    if task == "CVD Risk":
        image = _remove_top_left_overlay_cimt(image)
        image = _clahe_cimt(image)

    input_size = preprocessing_config.input_size
    mean = preprocessing_config.mean
    std = preprocessing_config.std

    image = cv2.resize(image, (input_size, input_size))
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def run_inference(model, image_tensor, metadata: dict | None = None) -> dict:
    """Run model inference on a preprocessed image."""
    if model is None:
        dummy_prob = float(np.random.uniform(0.2, 0.8))
        return {
            "probability": dummy_prob,
            "prediction": int(dummy_prob > 0.5),
            "confidence": abs(dummy_prob - 0.5) * 2,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    prob = torch.sigmoid(logits).item()
    return {
        "probability": prob,
        "prediction": int(prob > 0.5),
        "confidence": abs(prob - 0.5) * 2,
    }


# Main prediction handler
def predict(image, task, model_name, age, sex):
    """Main prediction function wired to the Gradio interface."""
    if image is None:
        raise gr.Error("Please upload a retinal fundus image.")
    if not model_name:
        raise gr.Error("Please select a model.")

    metadata = {}
    if age is not None and age != "":
        metadata["age"] = float(age)
    if sex is not None and sex != "":
        metadata["sex"] = int(sex)

    model, preprocessing_config = load_model(task, model_name)
    image_tensor = preprocess_image(np.array(image), preprocessing_config, task=task)
    result = run_inference(model, image_tensor, metadata)

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

    fhir_observation = generate_fhir_observation(
        task=task, model_name=model_name,
        probability=prob, prediction=pred, metadata=metadata,
    )
    return summary, json.dumps(fhir_observation, indent=2)


# FHIR output generation
def generate_fhir_observation(task, model_name, probability, prediction, metadata=None):
    """Generate a FHIR R4 Observation resource from the prediction."""
    now = datetime.utcnow().isoformat() + "Z"
    task_codes = {
        "CVD Risk": {"system": "http://loinc.org", "code": "PLACEHOLDER-CVD", "display": "Cardiovascular disease risk assessment from retinal imaging"},
        "Hypertension": {"system": "http://loinc.org", "code": "PLACEHOLDER-HTN", "display": "Hypertension risk assessment from retinal imaging"},
    }
    code_info = task_codes.get(task, task_codes["CVD Risk"])

    observation = {
        "resourceType": "Observation", "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "imaging", "display": "Imaging"}]}],
        "code": {"coding": [code_info]},
        "effectiveDateTime": now,
        "valueQuantity": {"value": round(probability, 4), "unit": "probability", "system": "http://unitsofmeasure.org", "code": "1"},
        "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "POS" if prediction == 1 else "NEG", "display": "Positive" if prediction == 1 else "Negative"}]}],
        "method": {"text": f"Deep learning model: {model_name} (Project Retinexis)"},
        "note": [{"text": "AI-generated prediction from retinal fundus image analysis. Not for clinical use without physician review."}],
    }

    if metadata:
        extensions = []
        if "age" in metadata:
            extensions.append({"url": "http://example.org/fhir/StructureDefinition/patient-age", "valueInteger": int(metadata["age"])})
        if "sex" in metadata:
            extensions.append({"url": "http://example.org/fhir/StructureDefinition/patient-sex", "valueCode": "male" if metadata["sex"] == 1 else "female"})
        if extensions:
            observation["extension"] = extensions
    return observation


#  Saliency maps 

def _architecture_type(task, model_name):
    arch = TASK_MODELS.get(task, {}).get(model_name, {}).get("architecture", "").lower()
    if any(x in arch for x in ("vit", "dino", "retfound", "transformer")):
        return "vit"
    if arch:
        return "cnn"
    raise gr.Error(f"Unknown architecture for model '{model_name}'.")


_GRADCAM_TARGET_LAYER_MAP = {
    "EfficientNet-B2": lambda m: [m._backbone.blocks[-1]],
    "ConvNeXt V2-Base": lambda m: [m._backbone.stages[-1]],
}


def _get_target_layers(model, model_name=""):
    if model_name in _GRADCAM_TARGET_LAYER_MAP:
        return _GRADCAM_TARGET_LAYER_MAP[model_name](model)
    backbone = getattr(model, "_backbone", model)
    last_conv = None
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise gr.Error("Could not find a convolutional layer for GradCAM.")
    return [last_conv]


def _to_batched_tensor(image_tensor):
    if isinstance(image_tensor, np.ndarray):
        tensor = torch.from_numpy(image_tensor.copy()).float()
    elif isinstance(image_tensor, torch.Tensor):
        tensor = image_tensor.float()
    else:
        raise gr.Error(f"Unsupported preprocessed image type: {type(image_tensor)}")
    if tensor.dim() == 3:
        if tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise gr.Error(f"Unexpected tensor shape: {tuple(tensor.shape)}")
    return tensor


def _apply_gradcam(model, image_tensor, original_image, model_name="", opacity=0.5):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    device = next(model.parameters()).device
    target_layers = _get_target_layers(model, model_name)
    tensor = _to_batched_tensor(image_tensor).to(device)
    model.eval()
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor)[0]

    orig_rgb = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
    h, w = orig_rgb.shape[:2]
    grayscale_resized = cv2.resize(grayscale_cam, (w, h))
    overlay_arr = show_cam_on_image(orig_rgb, grayscale_resized, use_rgb=True, image_weight=1 - opacity)
    return Image.fromarray(overlay_arr), "GradCAM"


def _apply_vit_gradcam(model, image_tensor, original_image, architecture="", opacity=0.5):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    backbone = getattr(model, "_backbone", model)
    device = next(model.parameters()).device
    tensor = _to_batched_tensor(image_tensor).to(device)

    if architecture in ("retfound_mae", "dinov2"):
        target_layers = [backbone.blocks[-1].norm1]
    elif architecture == "dinov3":
        target_layers = [backbone.encoder.layer[-1].norm1]
    else:
        if hasattr(backbone, "blocks"):
            target_layers = [backbone.blocks[-1]]
        elif hasattr(backbone, "norm"):
            target_layers = [backbone.norm]
        else:
            raise gr.Error(f"No ViT GradCAM target layer found for: {architecture}")

    def reshape_transform(tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        n = tensor.size(1)
        if int((n - 1) ** 0.5) ** 2 == (n - 1):
            tensor = tensor[:, 1:]
        h = w = int(tensor.size(1) ** 0.5)
        return tensor.reshape(tensor.size(0), h, w, tensor.size(-1)).permute(0, 3, 1, 2)

    model.eval()
    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]

    orig_rgb = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
    h, w = orig_rgb.shape[:2]
    grayscale_resized = cv2.resize(grayscale_cam, (w, h))
    overlay_arr = show_cam_on_image(orig_rgb, grayscale_resized, use_rgb=True, image_weight=1 - opacity)
    return Image.fromarray(overlay_arr), "GradCAM"


def generate_saliency_and_store(image, task, model_name):
    overlay, info = generate_saliency_map(image, task, model_name)
    return overlay, info, overlay, image, True, "Show Original"


def toggle_saliency_view(showing_overlay, overlay_img, original_img):
    if showing_overlay:
        return original_img, False, "Show Overlay"
    return overlay_img, True, "Show Original"


def generate_saliency_map(image, task, model_name):
    if image is None:
        raise gr.Error("Please upload an image first.")

    model, preprocessing_config = load_model(task, model_name)
    if model is None:
        return image, "**Model not yet loaded.**"

    arch_type = _architecture_type(task, model_name)
    architecture = TASK_MODELS.get(task, {}).get(model_name, {}).get("architecture", "")
    image_tensor = preprocess_image(np.array(image), preprocessing_config, task=task)

    if arch_type == "vit":
        overlay, method = _apply_vit_gradcam(model, image_tensor, image, architecture=architecture)
    else:
        overlay, method = _apply_gradcam(model, image_tensor, image, model_name=model_name)

    info = f"**Method:** {method}  \n**Model:** {model_name}  \n\n*Highlighted regions indicate areas most influential for the prediction.*"
    return overlay, info


#  Model comparison 

def compare_models(image, task, age, sex):
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
    rows, error_rows = [], []

    for name in model_names:
        try:
            model, preprocessing_config = load_model(task, name)
            if model is None:
                error_rows.append((name, "Model not yet loaded"))
                continue
            image_tensor = preprocess_image(img_array, preprocessing_config, task=task)
            result = run_inference(model, image_tensor, metadata)
            rows.append((name, _architecture_type(task, name).upper(), "Positive" if result["prediction"] == 1 else "Negative", result["probability"], result["confidence"]))
        except gr.Error:
            raise
        except Exception as e:
            error_rows.append((name, str(e)))

    table_lines = [f"### {task} — Model Comparison\n", "| Model | Type | Prediction | Probability | Confidence |", "|-------|------|-----------|-------------|------------|"]
    if rows:
        best = max(rows, key=lambda r: r[4])
        for name, arch, pred, prob, conf in rows:
            marker = " [best]" if name == best[0] else ""
            table_lines.append(f"| {name}{marker} | {arch} | {pred} | {prob:.1%} | {conf:.1%} |")
    for name, msg in error_rows:
        table_lines.append(f"| {name} | — | — | — | {msg} |")
    if rows:
        table_lines.append(f"\nHighest confidence: **{best[0]}** ({best[4]:.1%})")
    if not rows and error_rows:
        table_lines.append("\n*No models produced a valid result.*")
    elif error_rows:
        table_lines.append(f"\n*{len(error_rows)} model(s) could not run.*")
    return "\n".join(table_lines)


# Dual eye input 

def predict_dual_eye(image_left, image_right, task, model_name, age, sex):
    if image_left is None and image_right is None:
        raise gr.Error("Please upload at least one retinal fundus image.")

    def _run(label, img):
        if img is None:
            return f"*No image uploaded for {label}.*"
        summary, _ = predict(img, task, model_name, age, sex)
        return f"### {label}\n{summary}"

    left_summary = _run("Left Eye", image_left)
    right_summary = _run("Right Eye", image_right)
    return left_summary, right_summary, f"{left_summary}\n\n---\n\n{right_summary}"


#  Session history 
def clear_history():
    return [], [], "*No predictions yet in this session.*"

def _history_gallery_items(history):
    return [(e["image"], e["caption"]) for e in history]

def _history_markdown(history):
    if not history:
        return "*No predictions yet in this session.*"
    lines = ["## Session History\n"]
    for e in reversed(history):
        lines.append(f"**{e['caption']}**\n\n{e['summary']}\n\n---\n")
    return "\n".join(lines)

def predict_and_log(image, task, model_name, age, sex, history):
    history = history or []
    summary, fhir = predict(image, task, model_name, age, sex)
    caption = f"{task} · {model_name} · {datetime.now().strftime('%H:%M:%S')}"
    updated = history + [{"image": image, "caption": caption, "summary": summary}]
    return summary, fhir, updated, _history_gallery_items(updated), _history_markdown(updated)

def dual_predict_and_log(image_left, image_right, task, model_name, age, sex, history):
    history = history or []
    left_summary, right_summary, combined = predict_dual_eye(image_left, image_right, task, model_name, age, sex)
    updated = list(history)
    ts = datetime.now().strftime("%H:%M:%S")
    if image_left is not None:
        updated.append({"image": image_left, "caption": f"{task} · {model_name} · Left Eye · {ts}", "summary": left_summary})
    if image_right is not None:
        updated.append({"image": image_right, "caption": f"{task} · {model_name} · Right Eye · {ts}", "summary": right_summary})
    return left_summary, right_summary, combined, updated, _history_gallery_items(updated), _history_markdown(updated)


#  HTML wrappers
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&display=swap');
.gradio-container { border-top: 8px solid #185FA5 !important; }
.retinexis-eyebrow { text-align: center; font-size: 1rem; letter-spacing: 0.12em; text-transform: uppercase; color: #0F6E56; font-weight: 500; margin-bottom: 4px; }
.retinexis-title { text-align: center; font-family: 'DM Serif Display', Georgia, serif !important; font-size: 2rem !important; font-weight: 400 !important; line-height: 1.1; margin-bottom: 4px; }
.retinexis-subtitle { text-align: center; color: #6b7280; font-size: 0.85rem; margin-bottom: 1.5rem; }
.retinexis-header-wrap { position: relative; text-align: center; padding: 2rem 0 0.5rem; overflow: visible; }
.retinexis-header-wrap::before { content: ''; position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 400px; height: 400px; border-radius: 50%; border: 48px solid rgba(24, 95, 165, 0.05); pointer-events: none; }
.retinexis-header-wrap::after { content: ''; position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 240px; height: 240px; border-radius: 50%; border: 30px solid rgba(29, 158, 117, 0.06); pointer-events: none; }
#disease-tabs, #disease-tabs > div, #disease-tabs > div > .tab-container[role="tablist"] { overflow: visible !important; }
#disease-tabs > div > .tab-container[role="tablist"] { display: grid !important; grid-template-columns: 1fr 1fr auto !important; gap: 10px !important; border-bottom: none !important; margin-bottom: 3rem !important; padding: 4px !important; }
#disease-tabs > div > .tab-container[role="tablist"] button { border: 1.5px solid #d1d5db !important; border-radius: 12px !important; padding: 0 18px !important; height: 56px !important; min-height: 56px !important; max-height: unset !important; background: white !important; font-size: 0.9rem !important; font-weight: 500 !important; text-align: left !important; color: #374151 !important; margin: 0 !important; line-height: 1.3 !important; transition: border-color 0.2s, background 0.2s !important; }
#disease-tabs > div > .tab-container[role="tablist"] button:hover { border-color: #1D9E75 !important; }
#disease-tabs > div > .tab-container[role="tablist"] button[data-tab-id="cvd"].selected { border-color: #185FA5 !important; background: #E6F1FB !important; color: #0C447C !important; }
#disease-tabs > div > .tab-container[role="tablist"] button[data-tab-id="htn"].selected { border-color: #0F6E56 !important; background: #E1F5EE !important; color: #085041 !important; }
#disease-tabs > div > .tab-container[role="tablist"] button[data-tab-id="history"] { border: 1px solid #e5e7eb !important; background: transparent !important; color: #6b7280 !important; font-size: 0.82rem !important; padding: 8px 14px !important; min-height: unset !important; align-self: center !important; }
#disease-tabs > div > .tab-container[role="tablist"] button[data-tab-id="history"].selected { border-color: #d1d5db !important; background: #f9fafb !important; color: #374151 !important; }
#disease-tabs > div > .tab-container[role="tablist"]::after { display: none !important; content: none !important; height: 0 !important; }
#disease-tabs > div > .tab-container[role="tablist"] button::after, #disease-tabs > div > .tab-container[role="tablist"] button::before { display: none !important; content: none !important; border: none !important; background: none !important; height: 0 !important; }
#disease-tabs .sub-tab-nav > div > .tab-container[role="tablist"] { display: flex !important; grid-template-columns: unset !important; gap: 0 !important; border-bottom: 1px solid #e5e7eb !important; margin-bottom: 1rem !important; padding: 0 !important; }
#disease-tabs .sub-tab-nav > div > .tab-container[role="tablist"] button { border: none !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; background: transparent !important; font-size: 0.82rem !important; font-weight: 400 !important; padding: 6px 14px !important; color: #6b7280 !important; margin: 0 !important; width: auto !important; }
#disease-tabs .sub-tab-nav > div > .tab-container[role="tablist"] button.selected { color: #185FA5 !important; border-bottom: 2px solid #185FA5 !important; font-weight: 500 !important; background: transparent !important; }
.btn-cvd > button, .btn-htn > button { position: sticky !important; bottom: 1rem !important; z-index: 10 !important; box-shadow: 0 2px 12px rgba(0,0,0,0.12) !important; }
.result-card { background: var(--background-fill-secondary); border: 0.5px solid var(--border-color-primary); border-left: 3px solid #185FA5; border-radius: 12px; padding: 1rem 1.1rem; }
.result-card.htn { border-left-color: #0F6E56; }
.result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.result-title { font-size: 0.875rem; font-weight: 500; }
.result-badge { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 0.72rem; font-weight: 500; }
.badge-positive { background: #FCEBEB; color: #791F1F; }
.badge-negative { background: #EAF3DE; color: #27500A; }
.bar-section { margin-bottom: 10px; }
.bar-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #6b7280; margin-bottom: 4px; }
.bar-labels strong { color: var(--body-text-color); }
.bar-track { height: 6px; background: var(--background-fill-tertiary, #e5e7eb); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; width: 0; animation: barFillIn 0.7s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
@keyframes barFillIn { to { width: var(--bar-width); } }
.bar-fill-cvd { background: #185FA5; animation-delay: 0.05s; }
.bar-fill-htn { background: #0F6E56; animation-delay: 0.05s; }
.bar-fill-conf { animation-delay: 0.25s; }
.stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
.stat-box { background: var(--background-fill-primary); border: 0.5px solid var(--border-color-primary); border-radius: 8px; padding: 10px 12px; }
.stat-val { font-size: 1.2rem; font-weight: 500; }
.stat-lbl { font-size: 0.7rem; color: #9ca3af; margin-top: 1px; }
.model-pill { display: inline-flex; align-items: center; gap: 5px; background: var(--background-fill-tertiary, #f3f4f6); border-radius: 99px; padding: 3px 10px; font-size: 0.7rem; color: #6b7280; margin-bottom: 10px; }
.pill-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.pill-dot-cvd { background: #185FA5; }
.pill-dot-htn { background: #0F6E56; }
.disclaimer { background: #FAEEDA; border: 0.5px solid #FAC775; border-radius: 8px; padding: 8px 12px; font-size: 0.72rem; color: #633806; line-height: 1.5; margin-top: 4px; }
.btn-cvd > button { background: #185FA5 !important; color: #fff !important; border: none !important; }
.btn-htn > button { background: #0F6E56 !important; color: #fff !important; border: none !important; }
.btn-cvd > button:hover { background: #144f8a !important; }
.btn-htn > button:hover { background: #0a4f3b !important; }
"""


def build_result_card(task, model_name, prob, pred, confidence):
    disease_key = "cvd" if task == "CVD Risk" else "htn"
    task_label = "Cardiovascular risk" if task == "CVD Risk" else "Hypertension"
    risk_label = "Positive" if pred == 1 else "Negative"
    badge_class = "badge-positive" if pred == 1 else "badge-negative"
    bar_class = f"bar-fill-{disease_key}"
    dot_class = f"pill-dot-{disease_key}"
    prob_pct = f"{prob * 100:.1f}%"
    conf_pct = f"{confidence * 100:.1f}%"
    return f"""
<div class="result-card{' htn' if disease_key == 'htn' else ''}">
  <div class="result-header"><span class="result-title">{task_label} prediction</span><span class="result-badge {badge_class}">{risk_label}</span></div>
  <div class="bar-section"><div class="bar-labels"><span>Probability</span><strong>{prob_pct}</strong></div><div class="bar-track"><div class="bar-fill {bar_class}" style="--bar-width:{prob_pct}"></div></div></div>
  <div class="bar-section"><div class="bar-labels"><span>Confidence</span><strong>{conf_pct}</strong></div><div class="bar-track"><div class="bar-fill {bar_class} bar-fill-conf" style="--bar-width:{conf_pct}"></div></div></div>
  <div class="stats-row"><div class="stat-box"><div class="stat-val">{prob_pct}</div><div class="stat-lbl">Probability</div></div><div class="stat-box"><div class="stat-val">{conf_pct}</div><div class="stat-lbl">Confidence</div></div></div>
  <div class="model-pill"><span class="pill-dot {dot_class}"></span>{model_name} &nbsp;·&nbsp; {task_label}</div>
  <div class="disclaimer">Research tool only — not for clinical diagnosis. Physician review required before any clinical decision.</div>
</div>"""


EMPTY_RESULT = '<div style="color:#9ca3af;font-size:0.82rem;text-align:center;padding:2.5rem 1rem;line-height:1.7">Upload a retinal fundus image<br>and click <em>Run prediction</em> to see results.</div>'


def predict_and_log_html(image, task, model_name, age, sex, history):
    summary, fhir, updated, gallery, history_md = predict_and_log(image, task, model_name, age, sex, history)
    try:
        prob = float(re.search(r"Probability:\*\* ([\d.]+)%", summary).group(1)) / 100
        confidence = float(re.search(r"Confidence:\*\* ([\d.]+)%", summary).group(1)) / 100
        pred = 1 if "Positive" in summary else 0
        card = build_result_card(task, model_name, prob, pred, confidence)
    except Exception:
        card = f"<div style='padding:1rem'>{summary}</div>"
    return card, fhir, updated, gallery, history_md


def _summary_to_card(summary, task, model_name):
    if summary.startswith("*No image"):
        return f"<div style='color:#9ca3af;font-size:0.82rem;padding:1rem'>{summary}</div>"
    try:
        prob = float(re.search(r"Probability:\*\* ([\d.]+)%", summary).group(1)) / 100
        confidence = float(re.search(r"Confidence:\*\* ([\d.]+)%", summary).group(1)) / 100
        pred = 1 if "Positive" in summary else 0
        return build_result_card(task, model_name, prob, pred, confidence)
    except Exception:
        return f"<div style='padding:1rem'>{summary}</div>"


def _extract_confidence(summary):
    try:
        return float(re.search(r"Confidence:\*\* ([\d.]+)%", summary).group(1)) / 100
    except Exception:
        return -1.0


def dual_predict_and_log_html(image_left, image_right, task, model_name, age, sex, history):
    left_summary, right_summary, combined, updated, gallery, history_md = dual_predict_and_log(image_left, image_right, task, model_name, age, sex, history)
    left_card = _summary_to_card(left_summary, task, model_name)
    right_card = _summary_to_card(right_summary, task, model_name)
    combined_card = right_card if _extract_confidence(right_summary) > _extract_confidence(left_summary) else left_card
    return left_card, right_card, combined_card, updated, gallery, history_md


# ── Gradio UI ───────────────────────────────────────────────────────────────

def build_prediction_subtabs(task, session_history):
    slug = "cvd" if task == "CVD Risk" else "htn"
    btn_class = f"btn-{slug}"
    initial_choices = get_model_choices(task)
    initial_model = initial_choices[0] if initial_choices else None
    initial_desc = get_model_description(task, initial_model) if initial_model else ""
    components = {}
    task_state = gr.State(task)

    with gr.Tabs(elem_classes=["sub-tab-nav"]):
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_dd = gr.Dropdown(choices=initial_choices, value=initial_model, label="Model", info="Select which trained model to use")
                with gr.Column(scale=3):
                    model_desc = gr.Textbox(value=initial_desc, label="Model info", interactive=False, lines=1)
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Retinal fundus image", type="pil", sources=["upload", "clipboard"], height=380)
                    with gr.Accordion("Patient metadata (optional)", open=False):
                        age_input = gr.Number(label="Age", precision=0, minimum=0, maximum=120)
                        sex_input = gr.Dropdown(choices=[("Female", 0), ("Male", 1)], label="Sex", type="index")
                with gr.Column(scale=1):
                    result_output = gr.HTML(value=EMPTY_RESULT)
                    with gr.Accordion("FHIR R4 observation (JSON)", open=False):
                        fhir_output = gr.Code(language="json", label="FHIR output", lines=18)
            predict_btn = gr.Button("Run prediction", variant="primary", size="lg", elem_classes=btn_class)
            model_dd.change(fn=lambda m: get_model_description(task, m), inputs=[model_dd], outputs=[model_desc])

        with gr.Tab("Compare models"):
            gr.Markdown("Analyze the image with all available models and compare predictions side-by-side.")
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_image = gr.Image(label="Retinal fundus image", type="pil", sources=["upload", "clipboard"], height=280)
                    with gr.Accordion("Patient metadata (optional)", open=False):
                        cmp_age = gr.Number(label="Age", precision=0)
                        cmp_sex = gr.Dropdown(choices=[("Female", 0), ("Male", 1)], label="Sex", type="index")
                    cmp_btn = gr.Button("Compare all models", variant="primary", elem_classes=btn_class)
                with gr.Column(scale=1):
                    cmp_output = gr.Markdown(value="*Upload an image and click Compare all models.*")
            cmp_btn.click(fn=compare_models, inputs=[cmp_image, task_state, cmp_age, cmp_sex], outputs=[cmp_output])

        with gr.Tab("Dual eye"):
            gr.Markdown("Upload left and right eye images for a complete analysis.")
            with gr.Column():
                dual_model = gr.Dropdown(choices=initial_choices, value=initial_model, label="Model")
                with gr.Row():
                    left_eye = gr.Image(label="Left eye", type="pil", sources=["upload", "clipboard"], height=240)
                    right_eye = gr.Image(label="Right eye", type="pil", sources=["upload", "clipboard"], height=240)
                with gr.Accordion("Patient metadata (optional)", open=False):
                    dual_age = gr.Number(label="Age", precision=0)
                    dual_sex = gr.Dropdown(choices=[("Female", 0), ("Male", 1)], label="Sex", type="index")
                dual_btn = gr.Button("Analyse both eyes", variant="primary", elem_classes=btn_class)
                clear_dual_btn = gr.Button("Clear images", size="sm")
            clear_dual_btn.click(fn=lambda: (None, None), inputs=[], outputs=[left_eye, right_eye])
            dual_left_out = gr.HTML(visible=False)
            dual_right_out = gr.HTML(visible=False)
            dual_combined = gr.HTML(value="<div style='color:#9ca3af;font-size:0.82rem;padding:1rem'>Combined analysis will appear here.</div>")

        with gr.Tab("Saliency map"):
            gr.Markdown("See which retinal regions the model focuses on when making a prediction.")
            with gr.Row():
                with gr.Column():
                    sal_model = gr.Dropdown(choices=initial_choices, value=initial_model, label="Model")
                    sal_image = gr.Image(label="Retinal fundus image", type="pil", sources=["upload", "clipboard"], height=280)
                    sal_btn = gr.Button("Generate saliency map", variant="primary", elem_classes=btn_class)
                with gr.Column():
                    sal_display = gr.Image(label="Saliency overlay", height=280)
                    sal_toggle = gr.Button("Show original", size="sm", visible=False)
                    sal_info = gr.Markdown()
            sal_overlay_state = gr.State(None)
            sal_original_state = gr.State(None)
            sal_showing = gr.State(True)
            sal_btn.click(fn=generate_saliency_and_store, inputs=[sal_image, task_state, sal_model], outputs=[sal_display, sal_info, sal_overlay_state, sal_original_state, sal_showing, sal_toggle])
            sal_toggle.click(fn=toggle_saliency_view, inputs=[sal_showing, sal_overlay_state, sal_original_state], outputs=[sal_display, sal_showing, sal_toggle])

    components.update({
        "predict_btn": predict_btn, "image_input": image_input, "model_dd": model_dd,
        "age_input": age_input, "sex_input": sex_input, "result_output": result_output,
        "fhir_output": fhir_output, "task_state": task_state, "session_history": session_history,
        "dual_btn": dual_btn, "left_eye": left_eye, "right_eye": right_eye,
        "dual_model": dual_model, "dual_age": dual_age, "dual_sex": dual_sex,
        "dual_left_out": dual_left_out, "dual_right_out": dual_right_out, "dual_combined": dual_combined,
    })
    return components

def build_app():
    with gr.Blocks(title="Project Retinexis", css=CUSTOM_CSS, theme=gr.themes.Soft()) as app:
        def _logo_uri(filename, media_type):
            path = os.path.join(_HERE, "images", filename)
            with open(path, "rb") as f:
                return f"data:{media_type};base64," + base64.b64encode(f.read()).decode()

        kcl_uri = _logo_uri("kingslogo.png", "image/png")
        sdm_uri = _logo_uri("SDM.jpeg", "image/jpeg")

        gr.HTML(f"""
            <div class="retinexis-header-wrap">
                <div style="display:flex;justify-content:center;align-items:center;gap:2rem;margin-bottom:0.5rem">
                    <img src="{kcl_uri}" alt="King's College London" style="height:45px">
                    <img src="{sdm_uri}" alt="SDM" style="height:45px">
                </div>
                <div class="retinexis-eyebrow">Retinal AI diagnostics</div>
                <h1 class="retinexis-title">Retinexis</h1>
                <p class="retinexis-subtitle">The eye knows more than it sees</p>
            </div>
        """)
        session_history = gr.State([])
        with gr.Tabs(elem_id="disease-tabs"):
            with gr.Tab("❤  Cardiovascular risk", id="cvd"):
                cvd = build_prediction_subtabs("CVD Risk", session_history)
            with gr.Tab("🩺  Hypertension", id="htn"):
                htn = build_prediction_subtabs("Hypertension", session_history)
            with gr.Tab("History", id="history"):
                gr.Markdown("### Session history\nAll predictions made during this session. Resets on page refresh.")
                history_gallery = gr.Gallery(label="Uploaded images", columns=3, height=250, object_fit="cover")
                history_md = gr.Markdown(value="*No predictions yet in this session.*")
                clear_btn = gr.Button("Clear history", variant="stop", size="sm")
                clear_btn.click(fn=clear_history, inputs=[], outputs=[session_history, history_gallery, history_md])

        for tab in (cvd, htn):
            tab["predict_btn"].click(fn=predict_and_log_html, inputs=[tab["image_input"], tab["task_state"], tab["model_dd"], tab["age_input"], tab["sex_input"], session_history], outputs=[tab["result_output"], tab["fhir_output"], session_history, history_gallery, history_md])
            tab["dual_btn"].click(fn=dual_predict_and_log_html, inputs=[tab["left_eye"], tab["right_eye"], tab["task_state"], tab["dual_model"], tab["dual_age"], tab["dual_sex"], session_history], outputs=[tab["dual_left_out"], tab["dual_right_out"], tab["dual_combined"], session_history, history_gallery, history_md])
    return app


if __name__ == "__main__":
   build_app().launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=[os.path.join(_HERE, "images")])