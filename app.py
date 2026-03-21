"""
SpotSeg — Text-Guided Object Segmentation
Gradio app for HuggingFace Spaces deployment.
Highlights objects in images using natural language prompts.
Optimized for free-tier CPU inference.
"""

import os
import time
import gradio as gr
import numpy as np
from PIL import Image

from seg_models import ObjectSegmentor
from seg_utils import (
    create_highlight_overlay,
    create_blur_background,
    create_detection_visualization,
    create_contour_outline,
)
from download_examples import download_examples


# ──────────────────────────────────────────────
# Download example images on startup
# ──────────────────────────────────────────────
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
download_examples(EXAMPLES_DIR)

# ──────────────────────────────────────────────
# Load models once at startup
# ──────────────────────────────────────────────
print("Loading SpotSeg models...")
segmentor = ObjectSegmentor()
print("Models loaded successfully.")


def predict(
    image: Image.Image,
    text_query: str,
    mode: str,
    highlight_color: str,
    threshold: float,
) -> tuple:
    """
    Main prediction endpoint.

    Args:
        image: Input PIL image
        text_query: Object to find (ignored in Auto-Detect mode)
        mode: "Highlight Object" | "Blur Background" | "Contour Outline" | "Auto-Detect All"
        highlight_color: Hex color for highlight overlay
        threshold: Confidence threshold (0.0 - 1.0)

    Returns:
        (result_image, stats_string, detected_objects_text)
    """
    if image is None:
        raise gr.Error("Please upload an image first.")

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    start = time.time()

    if mode == "Auto-Detect All":
        # Use YOLOv8 to detect all objects
        detections = segmentor.detect_all_objects(image, conf=threshold)
        elapsed = time.time() - start

        result_img = create_detection_visualization(image, detections, highlight_color)

        # Build object list
        if detections:
            obj_list = []
            for d in detections:
                obj_list.append(f"{d['label']} ({d['confidence']:.0%})")
            objects_text = ", ".join(obj_list)
            unique_classes = len(set(d["label"] for d in detections))
            stats = (
                f"{image.size[0]}×{image.size[1]} · "
                f"{elapsed:.2f}s · "
                f"{len(detections)} objects · "
                f"{unique_classes} classes"
            )
        else:
            objects_text = "No objects detected — try lowering the threshold."
            stats = (
                f"{image.size[0]}×{image.size[1]} · "
                f"{elapsed:.2f}s · "
                f"0 objects"
            )

        return result_img, stats, objects_text

    else:
        # Text-guided segmentation with CLIPSeg
        if not text_query or text_query.strip() == "":
            raise gr.Error("Please enter an object to find (e.g., 'dog', 'car', 'person').")

        # Support comma-separated queries → pick the one with highest confidence
        queries = [q.strip() for q in text_query.split(",") if q.strip()]

        masks = []
        for q in queries:
            mask, score = segmentor.segment_object(image, q, threshold=threshold)
            if mask is not None:
                masks.append((mask, q, score))

        elapsed = time.time() - start

        if not masks:
            raise gr.Error(
                f"Could not find '{text_query}' in the image. "
                "Try a different query or lower the threshold."
            )

        # Combine all masks
        combined_mask = np.zeros_like(masks[0][0], dtype=np.float32)
        for mask, _, _ in masks:
            combined_mask = np.maximum(combined_mask, mask)

        found_labels = [f"{q} ({s:.0%})" for _, q, s in masks]

        if mode == "Highlight Object":
            result_img = create_highlight_overlay(image, combined_mask, highlight_color)
        elif mode == "Blur Background":
            result_img = create_blur_background(image, combined_mask)
        elif mode == "Contour Outline":
            result_img = create_contour_outline(image, combined_mask, highlight_color)
        else:
            result_img = create_highlight_overlay(image, combined_mask, highlight_color)

        stats = (
            f"{image.size[0]}×{image.size[1]} · "
            f"{elapsed:.2f}s · "
            f"CLIPSeg"
        )
        objects_text = "Found: " + ", ".join(found_labels)

        return result_img, stats, objects_text


# ──────────────────────────────────────────────
# Build Gradio Interface
# ──────────────────────────────────────────────

example_images = []
for fname in sorted(os.listdir(EXAMPLES_DIR)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        example_images.append(os.path.join(EXAMPLES_DIR, fname))

with gr.Blocks(
    title="SpotSeg — Text-Guided Object Segmentation",
    theme=gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 960px !important; }
    .gr-button-primary { background: #4fd1c5 !important; color: #0b0f14 !important; }
    """,
) as demo:
    gr.Markdown(
        """
        # 🎯 SpotSeg — Text-Guided Object Segmentation
        Type what you're looking for, and SpotSeg will highlight it in the image.
        Use **Auto-Detect All** to find every object automatically.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image", height=320)
            text_query = gr.Textbox(
                label="What to find",
                placeholder="e.g. dog, car, person (comma-separate for multiple)",
                info="Leave empty for Auto-Detect mode",
            )
            mode = gr.Dropdown(
                choices=[
                    "Highlight Object",
                    "Blur Background",
                    "Contour Outline",
                    "Auto-Detect All",
                ],
                value="Highlight Object",
                label="Mode",
            )
            with gr.Row():
                highlight_color = gr.ColorPicker(
                    value="#4fd1c5",
                    label="Highlight Color",
                )
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.35,
                    step=0.05,
                    label="Threshold",
                )
            run_btn = gr.Button("Find Objects", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Result", height=320)
            stats_text = gr.Textbox(label="Stats", interactive=False)
            objects_text = gr.Textbox(label="Detected Objects", interactive=False)

    run_btn.click(
        fn=predict,
        inputs=[input_image, text_query, mode, highlight_color, threshold],
        outputs=[output_image, stats_text, objects_text],
    )

    if example_images:
        gr.Examples(
            examples=[
                [example_images[0], "car", "Highlight Object", "#4fd1c5", 0.35],
                [example_images[1], "person", "Blur Background", "#f6ad55", 0.35],
                [example_images[2], "dog", "Highlight Object", "#68d391", 0.30],
            ],
            inputs=[input_image, text_query, mode, highlight_color, threshold],
            outputs=[output_image, stats_text, objects_text],
            fn=predict,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
