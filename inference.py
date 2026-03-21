"""
SpotSeg — Standalone Inference Script

Run text-guided object segmentation or auto-detection from the command line.

Usage:
    # Text-guided highlight
    python inference.py --input photo.jpg --output result.png --query "dog"

    # Blur background around object
    python inference.py --input photo.jpg --output result.png --query "person" --mode blur

    # Auto-detect all objects
    python inference.py --input photo.jpg --output result.png --mode detect

    # Process a folder
    python inference.py --input ./photos/ --output ./results/ --query "car" --batch

    # Save raw segmentation mask
    python inference.py --input photo.jpg --output mask.npy --query "cat" --save-raw
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from models import ObjectSegmentor
from utils import (
    create_highlight_overlay,
    create_blur_background,
    create_detection_visualization,
    create_contour_outline,
)


def process_single(
    segmentor: ObjectSegmentor,
    image_path: str,
    output_path: str,
    query: str,
    mode: str,
    color: str,
    threshold: float,
    save_raw: bool,
):
    """Process a single image."""
    image = Image.open(image_path).convert("RGB")
    print(f"  Image: {image.size[0]}×{image.size[1]}")

    start = time.time()

    if mode == "detect":
        detections = segmentor.detect_all_objects(image, conf=threshold)
        elapsed = time.time() - start
        result = create_detection_visualization(image, detections, color)
        print(f"  Found {len(detections)} objects in {elapsed:.2f}s")
        for d in detections:
            print(f"    - {d['label']}: {d['confidence']:.1%}")
    else:
        if not query:
            print("  Error: --query is required for highlight/blur/contour modes")
            return

        queries = [q.strip() for q in query.split(",") if q.strip()]
        masks = []
        for q in queries:
            mask, score = segmentor.segment_object(image, q, threshold=threshold)
            if mask is not None:
                masks.append((mask, q, score))
                print(f"  Found '{q}' with {score:.1%} confidence")
            else:
                print(f"  Could not find '{q}'")

        if not masks:
            print("  No objects found. Try a different query or lower threshold.")
            return

        elapsed = time.time() - start
        combined_mask = np.zeros_like(masks[0][0], dtype=np.float32)
        for mask, _, _ in masks:
            combined_mask = np.maximum(combined_mask, mask)

        if save_raw:
            np.save(output_path, combined_mask)
            print(f"  Saved raw mask to {output_path} ({elapsed:.2f}s)")
            return

        if mode == "blur":
            result = create_blur_background(image, combined_mask)
        elif mode == "contour":
            result = create_contour_outline(image, combined_mask, color)
        else:
            result = create_highlight_overlay(image, combined_mask, color)

        print(f"  Inference: {elapsed:.2f}s")

    result.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpotSeg — Text-Guided Object Segmentation"
    )
    parser.add_argument("--input", required=True, help="Image path or folder")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--query", default="", help="Object to find (comma-separated)")
    parser.add_argument(
        "--mode",
        default="highlight",
        choices=["highlight", "blur", "contour", "detect"],
        help="Visualization mode",
    )
    parser.add_argument("--color", default="#4fd1c5", help="Highlight color (hex)")
    parser.add_argument(
        "--threshold", type=float, default=0.35, help="Confidence threshold"
    )
    parser.add_argument("--batch", action="store_true", help="Process folder of images")
    parser.add_argument(
        "--save-raw", action="store_true", help="Save raw mask as .npy"
    )

    args = parser.parse_args()

    print("Loading models...")
    segmentor = ObjectSegmentor()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [
            f
            for f in sorted(os.listdir(args.input))
            if os.path.splitext(f)[1].lower() in extensions
        ]
        print(f"Processing {len(files)} images...")

        for fname in files:
            print(f"\n{fname}:")
            ext = ".npy" if args.save_raw else os.path.splitext(fname)[1]
            out_name = os.path.splitext(fname)[0] + f"_spotseg{ext}"
            process_single(
                segmentor,
                os.path.join(args.input, fname),
                os.path.join(args.output, out_name),
                args.query,
                args.mode,
                args.color,
                args.threshold,
                args.save_raw,
            )
    else:
        print(f"\n{args.input}:")
        process_single(
            segmentor,
            args.input,
            args.output,
            args.query,
            args.mode,
            args.color,
            args.threshold,
            args.save_raw,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
