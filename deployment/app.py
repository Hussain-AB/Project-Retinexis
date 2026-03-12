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
from pathlib import Path
# from src.models import create_model  # uncomment when deploying with full pipeline
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
    #look up the archectrure and variant from the TASK_MODELS dict
    model_info = TASK_MODELS.get(task, {}).get(model_name, {})
    architecture = model_info.get("architecture")
    variant = model_info.get("variant")

    #2. create model from the factory function
    if variant and variant != "default":
        model = create_model(architecture, variant=variant, pretrained=True)
    else:
        model = create_model(architecture, pretrained=True)

    #3. load the trained checkpoint weights
    # if model trained with meta data yiu need to load that to 
    checkpoint_path = Path(f"outputs/{task}/{model_name}/fold_0/checkpoints/best.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise gr.Error(f"Checkpoint not found: {checkpoint_path}")

    #4. set model to eval mode
    model.eval()

    #5. return model and preprocessing config
    return model, model.preprocessing_config



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
    # Read the model's required preprocessing settings
    input_size = preprocessing_config.input_size
    mean = preprocessing_config.mean
    std = preprocessing_config.std

    # Resize the image to the model's expected input size
    image = cv2.resize(image, (input_size, input_size))
    # Scale pixel values from 0-255 to 0-1
    image = image.astype(np.float32) / 255.0
    # Normalize using the model's expected mean and std
    image = (image - np.array(mean)) / np.array(std)

    # Convert to tensor: (H, W, C) → (1, C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    return tensor




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

    # 1. Pick device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)


    # 2. Run the model (no gradient needed for inference)
    with torch.no_grad():
        logits = model(image_tensor)

     # 3. Apply sigmoid to convert logits to probability (0 to 1)
    prob = torch.sigmoid(logits).item()

    # 4. Return results
    return {
        "probability": prob,
        "prediction": int(prob > 0.5),
        "confidence": abs(prob - 0.5) * 2,
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


def generate_saliency_map(image, task, model_name):
    """
    Generate a GradCAM or attention rollout heatmap overlay.

    TODO [Person 3]: Implement saliency map generation:
        - For CNNs (EfficientNet, ConvNeXt, SE-ResNeXt): use GradCAM on last conv layer
        - For ViTs (RETFound, DINOv2, DINOv3): use attention rollout from last attention block
        - Overlay heatmap on original image with adjustable opacity
        - Return both the overlay image and a raw heatmap

    Libraries to consider:
        - pytorch-grad-cam (pip install grad-cam)
        - Custom attention rollout for ViTs

    Returns:
        tuple: (overlay_image, raw_heatmap)
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    # STUB — returns placeholder. Remove when implementing.
    placeholder = np.zeros_like(np.array(image))
    return Image.fromarray(placeholder), "Saliency map not yet implemented."



# [EXTRAS] Model comparison


def compare_models(image, task, age, sex):
    """
    Run all available models for a task on the same image and compare.

    TODO [Person 3]: Implement model comparison:
        1. Get all models for the selected task from TASK_MODELS
        2. Run predict() for each model
        3. Build a comparison table with columns: Model, Prediction, Probability, Confidence
        4. Return as a Markdown table or gr.Dataframe

    Returns:
        str: Markdown-formatted comparison table
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    models = get_model_choices(task)
    rows = []
    for name in models:
        # TODO: Replace with actual inference per model
        dummy_prob = float(np.random.uniform(0.2, 0.8))
        pred = "Positive" if dummy_prob > 0.5 else "Negative"
        rows.append(f"| {name} | {pred} | {dummy_prob:.1%} | {abs(dummy_prob - 0.5)*2:.1%} |")

    table = (
        "| Model | Prediction | Probability | Confidence |\n"
        "|-------|-----------|-------------|------------|\n"
        + "\n".join(rows)
    )
    return table



# [EXTRAS] Dual eye input


def predict_dual_eye(image_left, image_right, task, model_name, age, sex):
    """
    Run prediction on both eyes and display side-by-side results.

    TODO [Person 3]: Implement dual eye prediction.
        The label is patient-level (not per-eye), so inference only runs
        on one eye. The dual eye upload reflects the clinical workflow
        where both eyes are captured during examination.
    
    Returns:
        tuple: (left_summary, right_summary, combined_summary)
    """
    results = []
    for label, img in [("Left Eye", image_left), ("Right Eye", image_right)]:
        if img is None:
            results.append(f"**{label}:** No image uploaded")
        else:
            summary, _ = predict(img, task, model_name, age, sex)
            results.append(f"### {label}\n{summary}")

    combined = "\n\n---\n\n".join(results)
    return combined



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
                predict_btn.click(
                    fn=predict,
                    inputs=[image_input, task_dropdown, model_dropdown, age_input, sex_input],
                    outputs=[result_output, fhir_output],
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

                dual_output = gr.Markdown(
                    value="*Upload images for both eyes and click 'Analyse Both Eyes'.*"
                )

                # Update model dropdown when task changes
                dual_task.change(
                    fn=lambda t: gr.update(choices=get_model_choices(t), value=get_model_choices(t)[0]),
                    inputs=[dual_task],
                    outputs=[dual_model],
                )

                dual_btn.click(
                    fn=predict_dual_eye,
                    inputs=[left_eye_input, right_eye_input, dual_task, dual_model, dual_age, dual_sex],
                    outputs=[dual_output],
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
                        sal_overlay = gr.Image(
                            label="Saliency Overlay",
                            height=300,
                        )
                        sal_info = gr.Markdown()

                # Update model dropdown when task changes
                sal_task.change(
                    fn=lambda t: gr.update(choices=get_model_choices(t), value=get_model_choices(t)[0]),
                    inputs=[sal_task],
                    outputs=[sal_model],
                )

                sal_btn.click(
                    fn=generate_saliency_map,
                    inputs=[sal_image, sal_task, sal_model],
                    outputs=[sal_overlay, sal_info],
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