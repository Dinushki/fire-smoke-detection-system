#!/usr/bin/env python3
"""
YOLO-based Fire and Smoke Detection Pipeline
Uses Ultralytics YOLOv8 for object detection.

This module provides a class-based wrapper around YOLO models,
allowing easy inference on images with configuration options.
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class YOLODetectionResult:
    """
    Stores results from YOLO detection for a single image.
    
    Attributes:
        image_path (str): Path to the input image.
        detections (List[Dict]): List of detection dictionaries, each containing
                                  'bbox', 'confidence', 'class_id', 'class_name'.
        processing_time (float): Inference time in seconds.
        annotated_image (Optional[np.ndarray]): Image with bounding boxes drawn (if requested).
    """
    image_path: str
    detections: List[Dict[str, Any]]
    processing_time: float
    annotated_image: Optional[np.ndarray] = None


class YOLOPipeline:
    """
    YOLO-based detection pipeline.
    
    Handles model loading, inference, result parsing, and visualization.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", config: Optional[Dict] = None):
        """
        Initialize the YOLO pipeline.
        
        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
                              If not found, it will be downloaded automatically.
            config (dict, optional): Configuration dictionary to override defaults.
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'conf_threshold': 0.25,        # Confidence threshold
            'iou_threshold': 0.45,          # IoU threshold for NMS
            'classes': None,                 # List of class IDs to filter (None = all)
            'device': 'cpu',                  # 'cpu', 'cuda', 'mps', etc.
            'save_annotated': True,           # Whether to save annotated images
            'output_dir': 'outputs',           # Directory for saving results
            'show_display': False,             # Display image window (requires GUI)
        }
        if config:
            self.config.update(config)
        
        # Load model
        self.logger.info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.logger.info("Model loaded successfully.")
    
    def detect_single_image(self, image_path: str, return_annotated: bool = True) -> Optional[YOLODetectionResult]:
        """
        Run YOLO detection on a single image.
        
        Args:
            image_path (str): Path to the input image.
            return_annotated (bool): If True, generate and store annotated image.
        
        Returns:
            YOLODetectionResult or None if inference fails.
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                source=image_path,
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                classes=self.config['classes'],
                device=self.config['device'],
                verbose=False  # Reduce logging clutter
            )
        except Exception as e:
            self.logger.error(f"YOLO inference failed: {e}")
            return None
        
        processing_time = time.time() - start_time
        
        # Parse detections from the first (and only) image result
        result = results[0]
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()          # (N, 4) in xyxy format
            confs = result.boxes.conf.cpu().numpy()          # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                # Convert xyxy to xywh (optional, but common)
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(w), int(h)],  # x, y, w, h
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': result.names[cls_id]
                })
        
        # Generate annotated image if requested
        annotated_image = None
        if return_annotated:
            # result.plot() returns a BGR numpy array with annotations
            annotated_image = result.plot()
        
        return YOLODetectionResult(
            image_path=image_path,
            detections=detections,
            processing_time=processing_time,
            annotated_image=annotated_image
        )
    
    def process_and_save(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Convenience method: detect, annotate, and save the result.
        
        Args:
            image_path (str): Input image path.
            output_dir (str, optional): Override default output directory.
        
        Returns:
            str or None: Path to saved annotated image, or None if failed.
        """
        result = self.detect_single_image(image_path, return_annotated=True)
        if result is None or result.annotated_image is None:
            return None
        
        # Determine output directory
        out_dir = output_dir or self.config['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate output filename
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"yolo_{name}{ext}")
        
        # Save annotated image
        cv2.imwrite(out_path, result.annotated_image)
        
        self.logger.info(f"Saved annotated image to {out_path}")
        self.logger.info(f"Detected {len(result.detections)} objects in {result.processing_time:.3f}s")
        for det in result.detections:
            self.logger.info(f"  {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")
        
        return out_path
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                      show: bool = False) -> Optional[str]:
        """
        Process a video file frame by frame.
        
        Args:
            video_path (str): Path to input video.
            output_path (str, optional): Path to save output video.
                                         If None, a default name in output_dir is used.
            show (bool): Whether to display the video in a window.
        
        Returns:
            str or None: Path to saved output video, or None if failed.
        """
        # (Implementation will be added later, as per your request)
        self.logger.info("Video processing not yet implemented.")
        return None


def main():
    """Command-line interface for YOLO image detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv8 Fire and Smoke Detection")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image or video")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Directory to save results (default: outputs)")
    parser.add_argument("--model", "-m", default="yolov8n.pt",
                        help="YOLO model file (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--classes", nargs="+", type=int, default=None,
                        help="Filter by class IDs (e.g., 0 for person)")
    parser.add_argument("--device", default="cpu",
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--show", action="store_true",
                        help="Display result window")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'conf_threshold': args.conf,
        'classes': args.classes,
        'device': args.device,
        'output_dir': args.output,
        'show_display': args.show,
    }
    
    # Create pipeline
    pipeline = YOLOPipeline(model_path=args.model, config=config)
    
    # Process based on file extension (simple heuristic)
    if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video mode (not yet fully implemented, but we call placeholder)
        out_video = pipeline.process_video(args.input, show=args.show)
        if out_video:
            print(f"Video saved to {out_video}")
        else:
            print("Video processing not available yet.")
    else:
        # Image mode
        out_path = pipeline.process_and_save(args.input, args.output)
        if out_path and args.show:
            img = cv2.imread(out_path)
            cv2.imshow("YOLO Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()