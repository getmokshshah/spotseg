"""
Visualization Utilities

Produces publication-quality overlays, highlights, and detection
visualizations for the SpotSeg pipeline.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import colorsys


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_label_colors(n: int, base_hex: str = "#4fd1c5") -> list:
    """
    Generate n visually distinct colors starting from a base hue.
    Returns list of (R, G, B) tuples.
    """
    base_rgb = hex_to_rgb(base_hex)
    base_hsv = colorsys.rgb_to_hsv(
        base_rgb[0] / 255, base_rgb[1] / 255, base_rgb[2] / 255
    )
    colors = []
    for i in range(n):
        hue = (base_hsv[0] + i * 0.618033988749895) % 1.0  # golden ratio
        sat = 0.7 + (i % 3) * 0.1
        val = 0.85
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def create_highlight_overlay(
    image: Image.Image,
    mask: np.ndarray,
    highlight_color: str = "#4fd1c5",
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a colored, semi-transparent highlight on the detected object.
    Areas outside the mask are slightly dimmed for emphasis.

    Args:
        image: Original PIL Image (RGB)
        mask: Float32 array [0, 1] same size as image
        highlight_color: Hex color for the highlight
        alpha: Opacity of the highlight overlay

    Returns:
        PIL Image with highlight applied
    """
    img_array = np.array(image).astype(np.float32)
    color = hex_to_rgb(highlight_color)

    # Expand mask to 3 channels
    mask_3d = np.stack([mask] * 3, axis=-1)

    # Create highlight layer
    highlight = np.full_like(img_array, color, dtype=np.float32)

    # Dim background slightly
    dimmed = img_array * 0.4

    # Blend: highlighted object + dimmed background
    result = np.where(
        mask_3d > 0,
        img_array * (1 - alpha) + highlight * alpha,
        dimmed,
    )

    # Add subtle glow at mask edges
    from PIL import ImageFilter as IF

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    edge_glow = mask_pil.filter(IF.GaussianBlur(radius=8))
    edge_glow = np.array(edge_glow).astype(np.float32) / 255.0
    edge_only = np.clip(edge_glow - mask, 0, 1)
    edge_3d = np.stack([edge_only] * 3, axis=-1)
    glow = np.full_like(img_array, color, dtype=np.float32)
    result = result + edge_3d * glow * 0.3

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def create_blur_background(
    image: Image.Image,
    mask: np.ndarray,
    blur_radius: int = 25,
) -> Image.Image:
    """
    Keep the detected object sharp and blur the background.
    Creates a portrait-mode / bokeh effect.

    Args:
        image: Original PIL Image
        mask: Float32 array [0, 1]
        blur_radius: Gaussian blur strength for background

    Returns:
        PIL Image with blurred background
    """
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_array = np.array(image).astype(np.float32)
    blur_array = np.array(blurred).astype(np.float32)

    # Smooth the mask edges for a natural transition
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_smooth = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    mask_smooth = np.array(mask_smooth).astype(np.float32) / 255.0
    mask_3d = np.stack([mask_smooth] * 3, axis=-1)

    result = img_array * mask_3d + blur_array * (1 - mask_3d)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def create_contour_outline(
    image: Image.Image,
    mask: np.ndarray,
    outline_color: str = "#4fd1c5",
    thickness: int = 3,
) -> Image.Image:
    """
    Draw a glowing contour outline around the detected object.

    Args:
        image: Original PIL Image
        mask: Float32 array [0, 1]
        outline_color: Hex color for the contour
        thickness: Line thickness in pixels

    Returns:
        PIL Image with contour overlay
    """
    import cv2

    color = hex_to_rgb(outline_color)
    binary = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.array(image).copy()

    # Draw filled semi-transparent overlay first
    overlay = result.copy()
    cv2.drawContours(overlay, contours, -1, color, -1)
    alpha = 0.15
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # Draw contour outlines
    cv2.drawContours(result, contours, -1, color, thickness)

    # Add outer glow effect
    glow_mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(glow_mask, contours, -1, 255, thickness + 4)
    glow_mask = cv2.GaussianBlur(glow_mask, (15, 15), 0)
    glow_3d = np.stack([glow_mask] * 3, axis=-1).astype(np.float32) / 255.0
    glow_color = np.full_like(result, color, dtype=np.float32)
    result = result.astype(np.float32) + glow_3d * glow_color * 0.3
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def create_detection_visualization(
    image: Image.Image,
    detections: list,
    highlight_color: str = "#4fd1c5",
) -> Image.Image:
    """
    Draw bounding boxes and labels for all detected objects.

    Args:
        image: Original PIL Image
        detections: List of dicts with 'label', 'confidence', 'bbox'
        highlight_color: Base color for generating palette

    Returns:
        PIL Image with detection boxes drawn
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)

    # Get distinct colors per unique class
    unique_labels = list(set(d["label"] for d in detections))
    colors = get_label_colors(len(unique_labels), highlight_color)
    label_color_map = dict(zip(unique_labels, colors))

    # Try to load a clean font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = label_color_map[label]

        # Draw box with slight transparency
        for offset in range(2):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
                width=1,
            )

        # Label background
        text = f"{label} {conf:.0%}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(y1 - th - 8, 0)

        draw.rectangle(
            [x1, label_y, x1 + tw + 10, label_y + th + 6],
            fill=color,
        )
        draw.text(
            (x1 + 5, label_y + 2),
            text,
            fill=(0, 0, 0),
            font=font,
        )

    return result
