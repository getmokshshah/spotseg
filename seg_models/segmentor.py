"""
Object Segmentation Models

Wraps CLIPSeg for text-guided segmentation and YOLOv8 for
automatic object detection. Both optimized for CPU inference.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from ultralytics import YOLO


class ObjectSegmentor:
    """
    Combined model wrapper for text-guided segmentation and object detection.

    - CLIPSeg (CIDAS/clipseg-rd64-refined): CLIP-based zero-shot segmentation.
      Takes a text prompt + image and produces a per-pixel probability mask
      indicating where the described object is located.

    - YOLOv8n: Ultralytics YOLOv8 nano model for fast multi-object detection.
      Used in Auto-Detect mode to find all recognizable objects.
    """

    def __init__(self):
        # ── CLIPSeg for text-guided segmentation ──
        print("  Loading CLIPSeg processor...")
        self.clip_processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        print("  Loading CLIPSeg model...")
        self.clip_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.clip_model.eval()

        # ── YOLOv8 nano for auto-detection ──
        print("  Loading YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        print("  All models loaded.")

    @torch.no_grad()
    def segment_object(
        self,
        image: Image.Image,
        query: str,
        threshold: float = 0.35,
    ) -> tuple:
        """
        Segment a specific object described by text.

        Args:
            image: PIL Image (RGB)
            query: Natural language description of the object
            threshold: Minimum probability to include in mask (0-1)

        Returns:
            (mask, confidence) where mask is a float32 numpy array [0,1]
            of the same size as the input image, or (None, 0.0) if not found.
        """
        inputs = self.clip_processor(
            text=[query],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        outputs = self.clip_model(**inputs)

        # CLIPSeg outputs logits of shape (batch, height, width)
        logits = outputs.logits[0]  # (H, W) — typically 352x352
        probs = torch.sigmoid(logits).cpu().numpy()

        # Resize to original image size
        from PIL import Image as PILImage

        prob_img = PILImage.fromarray((probs * 255).astype(np.uint8))
        prob_img = prob_img.resize(image.size, PILImage.BILINEAR)
        mask = np.array(prob_img).astype(np.float32) / 255.0

        # Compute confidence as mean probability in the thresholded region
        binary = mask > threshold
        if binary.sum() < 50:
            # Too few pixels — object not found
            return None, 0.0

        confidence = float(mask[binary].mean())
        # Zero out below threshold for clean mask
        mask[~binary] = 0.0

        return mask, confidence

    def detect_all_objects(
        self,
        image: Image.Image,
        conf: float = 0.30,
    ) -> list:
        """
        Detect all objects in the image using YOLOv8.

        Args:
            image: PIL Image (RGB)
            conf: Minimum confidence threshold

        Returns:
            List of dicts with keys: label, confidence, bbox (x1, y1, x2, y2)
        """
        results = self.yolo(image, conf=conf, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                label = result.names[cls_id]
                confidence = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    }
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
