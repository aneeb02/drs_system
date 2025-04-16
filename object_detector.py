#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Detector Module

This module is responsible for detecting cricket-related objects in video frames,
including the ball, stumps, batsman, and bat. It uses a combination of traditional
computer vision techniques and deep learning approaches.

Team Member Responsibilities:
----------------------------
Member 3: Object detection implementation, model training/integration, and detection optimization
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple

class ObjectDetector:
    """
    Detects cricket-related objects in frames.
    
    This class implements detection algorithms for cricket balls, stumps,
    batsmen, and bats using a hybrid approach combining traditional CV
    techniques with deep learning.
    
    Team Member Responsibilities:
    ----------------------------
    Member 3: Implementation of detection algorithms and model integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Object Detector with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.detection_method = config.get("detection_method", "hybrid")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Initialize detection models based on method
        if self.detection_method in ["deep_learning", "hybrid"]:
            self._init_deep_learning_models()
        
        # Initialize traditional CV parameters
        self._init_traditional_cv_params()
    
    def _init_deep_learning_models(self):
        """Initialize deep learning models for object detection."""
        # In a real implementation, this would load pre-trained models
        # For this prototype, we'll use placeholders
        
        # YOLO or similar model could be used here
        # self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # For now, we'll use a placeholder
        self.use_deep_learning = True
        print("Deep learning models would be initialized here")
        
        # Define class names for the model
        self.classes = ["cricket_ball", "stumps", "batsman", "bat"]
    
    def _init_traditional_cv_params(self):
        """Initialize parameters for traditional computer vision approaches."""
        # Ball detection parameters
        self.ball_color_lower = np.array([0, 0, 100])  # HSV lower bound for red cricket ball
        self.ball_color_upper = np.array([10, 255, 255])  # HSV upper bound for red cricket ball
        self.ball_size_range = (5, 30)  # Min and max radius in pixels
        
        # Stump detection parameters
        self.stump_color_lower = np.array([20, 0, 200])  # HSV lower bound for white/cream stumps
        self.stump_color_upper = np.array([40, 30, 255])  # HSV upper bound for white/cream stumps
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.get("bg_history", 120),
            varThreshold=self.config.get("bg_threshold", 16)
        )
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect objects in the frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing detection results for each object type
        """
        results = {
            "ball": [],
            "stumps": [],
            "batsman": [],
            "bat": []
        }
        
        # Choose detection method based on configuration
        if self.detection_method == "deep_learning":
            detections = self._detect_with_deep_learning(frame)
            self._process_dl_detections(detections, results)
        elif self.detection_method == "traditional":
            self._detect_with_traditional_cv(frame, results)
        else:  # hybrid approach
            # Use deep learning for batsman and stumps
            detections = self._detect_with_deep_learning(frame)
            self._process_dl_detections(detections, results)
            
            # If ball not detected with deep learning, try traditional methods
            if not results["ball"]:
                ball_detections = self._detect_ball_traditional(frame)
                results["ball"] = ball_detections
        
        return results
    
    def _detect_with_deep_learning(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using deep learning models.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection results
        """
        # In a real implementation, this would run inference with the model
        # For this prototype, we'll return placeholder detections
        
        # Placeholder for demonstration - in a real implementation, this would be model output
        # This would be replaced with actual model inference
        
        # For testing purposes, let's create some dummy detections
        # In the real implementation, these would come from the model
        
        # Detect a ball in the center-right area of the frame
        h, w = frame.shape[:2]
        detections = [
            {
                "class_id": 0,  # cricket_ball
                "confidence": 0.85,
                "bbox": (int(w * 0.7), int(h * 0.5), 20, 20)  # x, y, width, height
            }
        ]
        
        # Detect stumps in the center of the frame
        detections.append({
            "class_id": 1,  # stumps
            "confidence": 0.92,
            "bbox": (int(w * 0.5) - 15, int(h * 0.4), 30, 150)  # x, y, width, height
        })
        
        # Detect batsman on the left side
        detections.append({
            "class_id": 2,  # batsman
            "confidence": 0.88,
            "bbox": (int(w * 0.3) - 50, int(h * 0.3), 100, 200)  # x, y, width, height
        })
        
        # Detect bat near the batsman
        detections.append({
            "class_id": 3,  # bat
            "confidence": 0.78,
            "bbox": (int(w * 0.4) - 20, int(h * 0.4), 40, 100)  # x, y, width, height
        })
        
        return detections
    
    def _process_dl_detections(self, detections: List[Dict[str, Any]], results: Dict[str, List[Dict[str, Any]]]):
        """
        Process deep learning detections into the results dictionary.
        
        Args:
            detections: List of detection results from deep learning model
            results: Dictionary to store processed results
        """
        for detection in detections:
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            
            if confidence < self.confidence_threshold:
                continue
            
            if class_id == 0:  # cricket_ball
                results["ball"].append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "center": (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
                    "radius": max(bbox[2], bbox[3]) // 2
                })
            elif class_id == 1:  # stumps
                results["stumps"].append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "top": (bbox[0] + bbox[2] // 2, bbox[1]),
                    "bottom": (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3])
                })
            elif class_id == 2:  # batsman
                results["batsman"].append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "center": (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                })
            elif class_id == 3:  # bat
                results["bat"].append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "handle": (bbox[0] + bbox[2] // 2, bbox[1]),
                    "edge": (bbox[0] + bbox[2], bbox[1] + bbox[3] // 2),
                    "tip": (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3])
                })
    
    def _detect_with_traditional_cv(self, frame: np.ndarray, results: Dict[str, List[Dict[str, Any]]]):
        """
        Detect objects using traditional computer vision techniques.
        
        Args:
            frame: Input frame
            results: Dictionary to store detection results
        """
        # Detect ball using color and shape
        ball_detections = self._detect_ball_traditional(frame)
        results["ball"] = ball_detections
        
        # Detect stumps using edge detection and Hough lines
        stump_detections = self._detect_stumps_traditional(frame)
        results["stumps"] = stump_detections
        
        # For batsman and bat, traditional methods are less reliable
        # In a real implementation, these would use more sophisticated techniques
        # For now, we'll use placeholders
    
    def _detect_ball_traditional(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect cricket ball using traditional CV techniques.
        
        Args:
            frame: Input frame
            
        Returns:
            List of ball detection results
        """
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red cricket ball
        mask1 = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)
        mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))  # Upper red hue range
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_detections = []
        for contour in contours:
            # Approximate the contour to a circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # Filter by size
            if self.ball_size_range[0] <= radius <= self.ball_size_range[1]:
                # Calculate circularity
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * area / (2 * np.pi * radius) ** 2 if radius > 0 else 0
                
                # If the contour is approximately circular
                if circularity > 0.7:
                    confidence = circularity  # Use circularity as confidence
                    ball_detections.append({
                        "bbox": (int(x - radius), int(y - radius), int(2 * radius), int(2 * radius)),
                        "confidence": confidence,
                        "center": (int(x), int(y)),
                        "radius": int(radius)
                    })
        
        # Sort by confidence and return the best detection
        ball_detections.sort(key=lambda x: x["confidence"], reverse=True)
        return ball_detections[:1]  # Return only the best detection
    
    def _detect_stumps_traditional(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect stumps using traditional CV techniques.
        
        Args:
            frame: Input frame
            
        Returns:
            List of stump detection results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Filter vertical lines (potential stumps)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is approximately vertical
            if abs(x2 - x1) < 20:  # Small horizontal difference
                vertical_lines.append((x1, y1, x2, y2))
        
        # Group nearby vertical lines (stumps are usually close together)
        # This is a simplified approach; a real implementation would use clustering
        if not vertical_lines:
            return []
        
        # For simplicity, just return the first set of vertical lines as stumps
        # In a real implementation, this would use more sophisticated grouping
        x_values = [line[0] for line in vertical_lines[:3]]
        y_min = min([min(line[1], line[3]) for line in vertical_lines[:3]])
        y_max = max([max(line[1], line[3]) for line in vertical_lines[:3]])
        
        # Calculate average x position and width
        x_avg = sum(x_values) / len(x_values)
        width = max(x_values) - min(x_values) + 10  # Add padding
        
        return [{
            "bbox": (int(x_avg - width/2), y_min, int(width), y_max - y_min),
            "confidence": 0.7,  # Placeholder confidence
            "top": (int(x_avg), y_min),
            "bottom": (int(x_avg), y_max)
        }]
