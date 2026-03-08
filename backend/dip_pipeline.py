#!/usr/bin/env python3
"""
Digital Image Processing Pipeline for Fire and Smoke Detection
Author: [Your Name]
Date: 2025-03-01

This module provides a class-based implementation of a fire and smoke detector
using classical computer vision techniques (color segmentation, morphological ops,
feature validation). It includes configurable parameters, logging, and a command-line
interface for processing single images.
"""

import cv2
import numpy as np
import os
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class DetectionResult:
    """
    Data class to store the results of a detection.
    
    Attributes:
        detected_type (str): "Fire", "Smoke", or "Normal"
        detected (bool): True if fire or smoke was found
        confidence (float): Approximate confidence score (0-1)
        fire_regions (List[Tuple[int,int,int,int]]): List of bounding boxes (x,y,w,h) for fire
        smoke_regions (List[Tuple[int,int,int,int]]): List of bounding boxes for smoke
        processing_time (float): Time taken for detection in seconds
    """
    detected_type: str
    detected: bool
    confidence: float
    fire_regions: List[Tuple[int, int, int, int]]
    smoke_regions: List[Tuple[int, int, int, int]]
    processing_time: float


class FireSmokeDetector:
    """
    Main detector class that encapsulates the entire image processing pipeline.
    
    The pipeline includes:
        - Image loading and resizing
        - Preprocessing (CLAHE, bilateral filter, HSV conversion)
        - Fire detection based on color and brightness variance
        - Smoke detection based on low saturation and low edge density
        - Drawing results and saving output
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector with optional configuration overrides.
        
        Args:
            config (dict, optional): Dictionary of configuration parameters.
                                     If None, default values are used.
        """
        # Default configuration parameters
        self.config = {
            # Image dimensions after resizing
            'resize_width': 640,
            'resize_height': 480,
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            
            # Bilateral filter parameters (noise reduction while preserving edges)
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            
            # Fire detection parameters
            'fire_hue_range': [(10, 35)],          # Hue range(s) for fire (can add more)
            'fire_saturation_range': (120, 255),   # Min/max saturation
            'fire_value_range': (150, 255),        # Min/max value (brightness)
            'fire_min_area': 1000,                  # Minimum contour area to consider
            'fire_brightness_variance_thresh': 200, # Minimum variance in V channel (flicker)
            'fire_kernel_size': (5, 5),             # Morphological kernel size
            
            # Smoke detection parameters
            'smoke_hue_range': (0, 180),            # Full hue range (smoke is gray)
            'smoke_saturation_range': (0, 60),       # Low saturation
            'smoke_value_range': (100, 230),         # Medium to high brightness
            'smoke_min_area': 1500,                   # Minimum contour area
            'smoke_max_edge_density': 8000,           # Maximum number of edge pixels in ROI
            'smoke_kernel_size': (7, 7),              # Morphological kernel size
        }
        
        # Update with user-provided configuration
        if config:
            self.config.update(config)
        
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk, resize it to the configured dimensions.
        
        Args:
            img_path (str): Path to the image file.
        
        Returns:
            np.ndarray or None: Loaded BGR image, or None if loading fails.
        """
        if not os.path.exists(img_path):
            self.logger.error(f"Image not found: {img_path}")
            return None
        
        image = cv2.imread(img_path)
        if image is None:
            self.logger.error(f"Failed to load image: {img_path}")
            return None
        
        # Resize to standard dimensions
        image = cv2.resize(image, (self.config['resize_width'], self.config['resize_height']))
        return image
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing steps to enhance image and prepare for detection.
        
        Steps:
            1. Convert BGR to LAB and apply CLAHE on L-channel for contrast enhancement.
            2. Convert back to BGR and apply bilateral filter to reduce noise while keeping edges.
            3. Convert to HSV color space for easier color-based segmentation.
        
        Args:
            image (np.ndarray): Input BGR image.
        
        Returns:
            Tuple containing:
                - filtered (np.ndarray): Bilateral filtered BGR image.
                - hsv (np.ndarray): HSV version of the filtered image.
                - v_channel (np.ndarray): Value (brightness) channel from HSV.
        """
        # Convert to LAB color space (better for lighting adjustments)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_grid_size']
        )
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Bilateral filter: preserves edges while smoothing
        filtered = cv2.bilateralFilter(
            enhanced,
            self.config['bilateral_d'],
            self.config['bilateral_sigma_color'],
            self.config['bilateral_sigma_space']
        )
        
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        
        # Extract the Value (brightness) channel for later analysis
        _, _, v = cv2.split(hsv)
        
        return filtered, hsv, v
    
    def detect_fire(self, hsv: np.ndarray, v_channel: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect fire regions using color thresholding and brightness variance validation.
        
        Args:
            hsv (np.ndarray): HSV image.
            v_channel (np.ndarray): Value channel (brightness) of the HSV image.
        
        Returns:
            List of bounding boxes (x, y, w, h) for detected fire regions.
        """
        fire_regions = []
        
        # Build a combined mask from all defined fire hue ranges
        fire_mask = None
        for hue_range in self.config['fire_hue_range']:
            lower = np.array([
                hue_range[0],
                self.config['fire_saturation_range'][0],
                self.config['fire_value_range'][0]
            ])
            upper = np.array([
                hue_range[1],
                self.config['fire_saturation_range'][1],
                self.config['fire_value_range'][1]
            ])
            mask = cv2.inRange(hsv, lower, upper)
            fire_mask = mask if fire_mask is None else cv2.bitwise_or(fire_mask, mask)
    
        # If no hue ranges were defined, return empty list
        if fire_mask is None:
            return fire_regions
        
        # Morphological opeaning to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config['fire_kernel_size'])
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of potential fire regions
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = self.config['fire_min_area']
        var_thresh = self.config['fire_brightness_variance_thresh']
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Extract the corresponding region from the value channel
                roi_v = v_channel[y:y+h, x:x+w]
                # Fire flickers → high variance in brightness
                if np.var(roi_v) > var_thresh:
                    fire_regions.append((x, y, w, h))
        
        return fire_regions
    
    def detect_smoke(self, hsv: np.ndarray, filtered: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect smoke regions using low saturation and low edge density validation.
        
        Args:
            hsv (np.ndarray): HSV image.
            filtered (np.ndarray): Bilateral filtered BGR image (used for edge detection).
        
        Returns:
            List of bounding boxes (x, y, w, h) for detected smoke regions.
        """
        smoke_regions = []
        
        # Smoke: low saturation, medium to high brightness
        lower = np.array([
            self.config['smoke_hue_range'][0],
            self.config['smoke_saturation_range'][0],
            self.config['smoke_value_range'][0]
        ])
        upper = np.array([
            self.config['smoke_hue_range'][1],
            self.config['smoke_saturation_range'][1],
            self.config['smoke_value_range'][1]
        ])
        smoke_mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological closing to connect nearby smoke regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config['smoke_kernel_size'])
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = self.config['smoke_min_area']
        max_edge_density = self.config['smoke_max_edge_density']
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Extract the corresponding region from the filtered BGR image
                roi = filtered[y:y+h, x:x+w]
                # Compute edge density (Canny edges)
                edges = cv2.Canny(roi, 50, 150)
                edge_count = np.sum(edges > 0)
                # Smoke has a soft, low-texture appearance → low edge density
                if edge_count < max_edge_density:
                    smoke_regions.append((x, y, w, h))
        
        return smoke_regions
    
    def draw_results(self, image: np.ndarray,
                     fire_regions: List[Tuple[int, int, int, int]],
                     smoke_regions: List[Tuple[int, int, int, int]],
                     detected_type: str) -> np.ndarray:
        """
        Annotate the image with bounding boxes and status text.
        
        Args:
            image (np.ndarray): Original BGR image (will be copied).
            fire_regions: List of fire bounding boxes.
            smoke_regions: List of smoke bounding boxes.
            detected_type: "Fire", "Smoke", or "Normal".
        
        Returns:
            np.ndarray: Annotated image.
        """
        output = image.copy()
        
        # Draw fire boxes in red
        for (x, y, w, h) in fire_regions:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, "Fire", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw smoke boxes in gray
        for (x, y, w, h) in smoke_regions:
            cv2.rectangle(output, (x, y), (x + w, y + h), (200, 200, 200), 2)
            cv2.putText(output, "Smoke", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Add overall status at top-left corner
        if not fire_regions and not smoke_regions:
            cv2.putText(output, "Normal Image", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            # Choose color based on what was detected (red for fire, gray for smoke only)
            color = (0, 0, 255) if fire_regions else (200, 200, 200)
            cv2.putText(output, f"ALERT: {detected_type}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return output
    
    def detect(self, img_path: str) -> Optional[DetectionResult]:
        """
        Run the full detection pipeline on a single image.
        
        Args:
            img_path (str): Path to the input image.
        
        Returns:
            DetectionResult object or None if image loading failed.
        """
        start_time = time.time()
        
        # Load image
        image = self.load_image(img_path)
        if image is None:
            return None
        
        # Preprocess
        filtered, hsv, v_channel = self.preprocess(image)
        
        # Detect fire and smoke
        fire_regions = self.detect_fire(hsv, v_channel)
        smoke_regions = self.detect_smoke(hsv, filtered)
        
        # Determine overall result
        if fire_regions:
            detected_type = "Fire"
            confidence = 0.85  # You could compute a more sophisticated score
        elif smoke_regions:
            detected_type = "Smoke"
            confidence = 0.75
        else:
            detected_type = "Normal"
            confidence = 0.95
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            detected_type=detected_type,
            detected=bool(fire_regions or smoke_regions),
            confidence=confidence,
            fire_regions=fire_regions,
            smoke_regions=smoke_regions,
            processing_time=processing_time
        )
    
    def process_and_save(self, img_path: str, output_dir: str = "outputs") -> Optional[str]:
        """
        Convenience method: detect, annotate, and save the result.
        
        Args:
            img_path (str): Input image path.
            output_dir (str): Directory to save the output image.
        
        Returns:
            str or None: Path to the saved output image, or None if failed.
        """
        # Run detection
        result = self.detect(img_path)
        if result is None:
            return None
        
        # Load image again for drawing (or you could store the original in detect)
        image = self.load_image(img_path)
        if image is None:
            return None
        
        # Draw results
        output_img = self.draw_results(image, result.fire_regions, result.smoke_regions, result.detected_type)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output filename
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, f"result_{base_name}")
        cv2.imwrite(out_path, output_img)
        
        self.logger.info(f"Saved result to {out_path}")
        self.logger.info(f"Detection: {result.detected_type} (conf: {result.confidence:.2f})")
        self.logger.info(f"Processing time: {result.processing_time:.3f}s")
        
        return out_path


def main():
    """Command-line interface for processing a single image."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fire and Smoke Detection using DIP")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input image")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Directory to save output image (default: outputs)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML configuration file (optional)")
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = None
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    
    # Create detector instance
    detector = FireSmokeDetector(config)
    
    # Process and save
    out_path = detector.process_and_save(args.input, args.output)
    
    if out_path:
        # Optionally display the result
        img = cv2.imread(out_path)
        cv2.imshow("Detection Result", img)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Processing failed. Check the log for details.")


if __name__ == "__main__":
    main()