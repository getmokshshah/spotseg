---
title: SpotSeg
emoji: 🎯
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.23.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# SpotSeg — Text-Guided Object Segmentation

Find and highlight any object in an image using natural language. Type "dog" and the dog lights up. Type "car, person" and both are highlighted. Switch to Auto-Detect to find everything at once.

**[Try the Live Demo →](https://huggingface.co/spaces/getmokshshah/spotseg)**

---

## What It Does

SpotSeg combines two powerful vision models into a single tool:

1. **Text-Guided Mode** — Describe what you're looking for in plain English. CLIPSeg (a CLIP-based zero-shot segmentation model) produces a per-pixel probability mask showing where that object is. No retraining needed — it understands any noun out of the box.

2. **Auto-Detect Mode** — YOLOv8 scans the entire image and draws labeled bounding boxes around every recognized object with confidence scores.

Both modes run entirely on CPU, making the tool deployable on free-tier cloud infrastructure.

## 📁 Project Structure

```
spotseg/
├── app.py                    # Gradio web app (for HuggingFace Spaces)
├── inference.py              # Standalone CLI inference script
├── requirements.txt          # Python dependencies
├── models/
│   └── segmentor.py          # CLIPSeg + YOLOv8 model wrapper
├── utils/
│   └── visualization.py      # Overlays, blur, contours, detection boxes
├── examples/                 # Sample images (downloaded at runtime)
└── .github/workflows/
    └── sync-to-hf.yml        # Auto-deploy to HuggingFace Spaces
```

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/getmokshshah/spotseg.git
cd spotseg
pip install -r requirements.txt
```

### 2. Run the Web App Locally

```bash
python app.py
```

Opens a Gradio interface at `http://localhost:7860` where you can upload images, type object names, and see highlighted results.

### 3. Run Inference from the Command Line

```bash
# Highlight a specific object
python inference.py --input photo.jpg --output result.png --query "dog"

# Blur everything except the subject
python inference.py --input photo.jpg --output result.png --query "person" --mode blur

# Draw contour outlines
python inference.py --input photo.jpg --output result.png --query "car" --mode contour

# Auto-detect all objects
python inference.py --input photo.jpg --output result.png --mode detect

# Process a folder of images
python inference.py --input ./photos/ --output ./results/ --query "cat" --batch

# Save raw segmentation mask as NumPy array
python inference.py --input photo.jpg --output mask.npy --query "dog" --save-raw
```

## Models

| Model | Task | Speed (CPU) | Memory | Source |
|-------|------|-------------|--------|--------|
| CLIPSeg (`clipseg-rd64-refined`) | Text-guided segmentation | ~1.5s/image | ~600MB | CIDAS / HuggingFace |
| YOLOv8n | Object detection (80 classes) | ~0.3s/image | ~50MB | Ultralytics |

**CLIPSeg** uses a CLIP vision-language backbone with a lightweight decoder head that produces pixel-level segmentation masks from arbitrary text prompts. It requires no fine-tuning — any English noun or phrase works as a query.

**YOLOv8 Nano** is a real-time object detector trained on COCO (80 common object classes). It's used exclusively in Auto-Detect mode for fast, comprehensive scene analysis.

## Visualization Modes

SpotSeg offers four output styles:

- **Highlight Object** — Semi-transparent color overlay on the detected object with the background dimmed. Includes a subtle edge glow for visual polish.
- **Blur Background** — Keeps the object sharp while applying a strong Gaussian blur to the background, creating a portrait / bokeh effect.
- **Contour Outline** — Draws a glowing contour line around the object boundary with a faint filled overlay. Clean and technical.
- **Auto-Detect All** — Switches to YOLOv8 and draws labeled bounding boxes with confidence scores around every detected object.

## How It Works

1. **Text Encoding** — Your text query is tokenized and encoded by CLIP's text encoder into a 512-dimensional embedding that captures semantic meaning.
2. **Image Encoding** — The input image is processed by CLIP's vision encoder (ViT backbone) to produce dense visual features at multiple spatial scales.
3. **Cross-Modal Fusion** — CLIPSeg's decoder compares text and image embeddings at each spatial location, producing a heatmap of per-pixel relevance scores.
4. **Mask Thresholding** — Raw probabilities are sigmoid-activated and thresholded to create a binary segmentation mask, then resized to the original image dimensions.
5. **Visualization** — The mask is composited with the original image using the selected visualization mode (highlight, blur, contour, or detection boxes).

### Architecture Details

CLIPSeg extends the standard CLIP model with a **FPN-style decoder** that takes multi-scale features from the vision transformer and a text conditioning vector. The decoder outputs a single-channel logit map at 352×352 resolution, which is upsampled to the input size. This architecture enables zero-shot segmentation — the model generalizes to any object described in natural language without class-specific training.

For Auto-Detect mode, YOLOv8 Nano uses a **CSPDarknet** backbone with a **PANet** feature pyramid and decoupled detection heads, optimized for speed on edge devices.

## Understanding the Output

- **Colored regions** → detected object location
- **Dimmed / blurred areas** → background
- **Confidence scores** → model certainty (higher = more reliable)
- **Multiple queries** → comma-separate to find several objects at once (e.g., "dog, frisbee")

## Performance

Benchmarked on a 2-core CPU (HuggingFace Spaces free tier):

| Task | Resolution | Inference Time | Peak RAM |
|------|-----------|---------------|----------|
| CLIPSeg (single query) | 640×480 | ~1.5s | ~800MB |
| CLIPSeg (3 queries) | 640×480 | ~3.5s | ~900MB |
| YOLOv8n Auto-Detect | 640×480 | ~0.3s | ~400MB |

## Configuration

### Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Path to image or folder |
| `--output` | required | Output path for results |
| `--query` | `""` | Object to find (comma-separated for multiple) |
| `--mode` | `highlight` | Mode: `highlight`, `blur`, `contour`, `detect` |
| `--color` | `#4fd1c5` | Highlight/contour color (hex) |
| `--threshold` | `0.35` | Confidence threshold (0.0–1.0) |
| `--batch` | `False` | Process all images in a folder |
| `--save-raw` | `False` | Save raw mask as .npy file |

## Use Cases

- **Image Editing** — Isolate subjects for compositing, background replacement, or selective adjustments
- **Accessibility** — Help visually impaired users understand what's in an image via object labeling
- **Content Moderation** — Detect and locate specific objects or categories in user-uploaded photos
- **Retail / E-commerce** — Automatically segment products from background for catalog images
- **Robotics & AR** — Zero-shot object localization for manipulation or scene understanding
- **Education** — Interactive tool for teaching computer vision concepts

## Limitations

- CLIPSeg produces **soft masks** — edges may not be pixel-perfect on complex shapes (hair, fur, thin objects)
- Performance depends on how well CLIP's training data represents the query — unusual or very specific objects may have lower accuracy
- YOLOv8n is limited to **80 COCO classes** in Auto-Detect mode; CLIPSeg has no class limit
- Very small objects (< 32px) may not be reliably segmented
- Overlapping objects of the same type may be merged into a single mask

## License

MIT License — free to use for research or commercial projects.

## Credits

- **CLIPSeg**: Lüddecke & Ecker, "Image Segmentation Using Text and Image Prompts" (CVPR 2022)
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (OpenAI, 2021)
- **YOLOv8**: Ultralytics (2023)
- **Built with**: PyTorch, Transformers, OpenCV, Gradio, NumPy, Ultralytics
