# Project Retinexis — UI Specification Document



---

## 1. Overview

The Project Retinexis web application allows users to upload a retinal fundus image, select a pretrained model, and receive a prediction for either cardiovascular disease (CVD) risk or hypertension. The uploaded image is automatically resized to the resolution required by the selected pretrained model. The UI should be clean, intuitive, and accessible to both clinical and research users.

---

## 2. Core Features

These are the minimum features required for a functional application.

### 2.1 Image Upload

- Users can upload a single retinal fundus image (JPEG or PNG) via drag-and-drop or file picker.
- A thumbnail preview of the uploaded image is displayed before running the model.
- Basic validation: file type check, file size limit (e.g. max 10MB), and a clear error message if invalid.
- The uploaded image is automatically resized to the resolution required by the selected model (e.g. 224px for ViT models, 260px for EfficientNet-B2) using the model's `preprocessing_config`.

### 2.2 Model Selection

- A dropdown or card selector lets the user choose which model to run (e.g. EfficientNet-B2, RETFound-DINOv2, DINOv2-Large, etc.).
- Each model option shows its name and a brief one-line description (e.g. "RETFound-DINOv2 — Retinal specialist, top performer").
- Default selection should be the best-performing model.
- The system will include three models per disease, allowing comparison of predictions across different architectures.

### 2.3 Task Selection

- Users choose the prediction task: CVD Risk (CIMT dataset) or Hypertension (BRSET dataset).
- Each disease has its own set of 3 deployed models, selected based on which architectures performed best for that specific task during training.

### 2.4 Prediction & Results Display

- A "Run Prediction" button sends the image and selected model/task to the backend.
- A loading spinner or progress indicator is shown while the model runs inference.
- Results are displayed clearly: the predicted risk score or classification, with a confidence level if available.

### 2.5 Error Handling

- Clear error messages for: invalid image format, server errors, model loading failures, and timeout.
- Users should always be able to retry or upload a new image without refreshing the page.

### 2.6 Patient Metadata Input

- Input fields for patient metadata used during model training (e.g. age, sex) to match the data the model was trained on.
- The required metadata fields will depend on the dataset: CIMT and BRSET each include different clinical variables.
- These fields are required if the model was trained with metadata as part of its input. If the model is image-only, these fields become optional context.

### 2.7 FHIR-Formatted Output

- Prediction results are formatted as FHIR Observation resources (JSON), enabling compatibility with hospital electronic health record systems.
- Users can export the FHIR-formatted output via a download button.
- This supports the client's requirement for healthcare data interoperability using either FHIR or HL7 standards.

---

## 3. Extra Features

These are additional features that would enhance the user experience, improve clinical usability, and make the application more practical for real-world use. They are not required for the initial version but can be added iteratively.

### 3.1 Attention / Saliency Map Visualisation

- Overlay a heatmap on the retinal image showing which regions the model focused on (e.g. GradCAM for CNNs, attention rollout for Vision Transformers).
- Toggle between the original image and the heatmap overlay.

### 3.2 Model Comparison

- Run multiple models on the same image and display results side by side.
- Show a comparison table with each model's prediction and confidence.

### 3.3 Dual Eye Input

- Option to upload both left and right eye images from the same patient.
- Run predictions on both images and display results side by side.
- This reflects the clinical workflow where clinicians capture fundus images of both eyes during routine examinations.

### 3.4 History & Session Persistence

- Keep a history of uploaded images and predictions within a session so users can compare results without re-uploading.

---

## 4. Library Recommendation

### Streamlit

Streamlit can be deployed on Streamlit Community Cloud, which provides a shared CPU environment with approximately 2.7 GB of RAM per app. Compared to Gradio, Streamlit offers more advanced interface features, including better support for dashboard-style layouts, improved layout control (such as columns, tabs, and sidebars), and greater flexibility for creating a customized user interface. For visualization, Streamlit also includes built-in theming and layout components (such as sidebars, columns, tabs, and metrics), allowing applications to look visually clean and organized out of the box without requiring additional CSS styling.

### Gradio

The main advantage of Gradio is that it is faster to implement. We define the model function and Gradio automatically manages the input/output interaction, whereas Streamlit requires more manual control but offers greater flexibility. Gradio can be easily deployed on Hugging Face Spaces. In terms of visualization, Gradio by default has a simpler interface, but it can be customized using a custom CSS file if needed. A typical Spaces environment provides approximately 16 GB of RAM, 2 CPU cores, and around 50 GB of temporary (non-persistent) disk storage by default. This makes Hugging Face Spaces a suitable option for deployment, as it allows us to load and run all three models within the same application without major memory constraints.

### Flask

Flask is different from Gradio and Streamlit because it is a general web framework rather than a specialized ML interface tool. It provides full control over the application's design, layout, and interactions. Using Flask would involve writing a Python backend with API endpoints to receive images and return predictions, creating a separate frontend using HTML, CSS, and JavaScript (or frameworks such as React), and manually connecting the two components by handling file uploads, sending requests to the backend, and displaying the prediction results in the user interface.

### Recommended: Streamlit

#### 4.1 Library Comparison

| Criteria | Streamlit | Gradio | Flask + React |
|---|---|---|---|
| Setup time | Very fast (minutes) | Fast (minutes) | Slow (days) |
| Python-only | Yes | Yes | No (JS + Python) |
| ML integration | Native (PyTorch, etc.) | Native | Manual API setup |
| Image upload | Built-in widget | Built-in widget | Custom component |
| Customisation | Moderate | Limited | Full control |
| Free deployment | Streamlit Cloud | HuggingFace Spaces | Requires hosting |
| Team familiarity | Python only needed | Python only needed | Needs frontend dev |

#### 4.2 Why Streamlit?

1. **Python-only:** Streamlit requires no JavaScript, HTML, or CSS knowledge. The built-in layout components are sufficient to create the user interface.
2. **Fast prototyping:** Image upload, model selection dropdowns, and result displays are all built-in widgets.
3. **Direct ML integration:** Streamlit integrates directly with PyTorch and our model pipeline. We can call `create_model()` and run inference without building a separate API layer.
4. **Free deployment:** Streamlit Community Cloud provides free hosting for public apps.
5. **Sufficient for our needs:** While Flask + React offers more customisation, we don't need a production-grade frontend. Streamlit handles image upload, dropdowns, progress bars, and result displays out of the box.

---

## 5. User Flow Summary

1. User opens the app and selects a task (CVD Risk or Hypertension).
2. User selects a model from the available options for that task.
3. User uploads a retinal fundus image.
4. User clicks "Run Prediction".
5. The app preprocesses the image, runs inference, and displays the result.
6. User can upload another image or change model/task and run again.


