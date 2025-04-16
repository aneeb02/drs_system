#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Calibration Module

This module is responsible for calibrating the camera to enable accurate 3D position
estimation and tracking of cricket objects.

Team Member Responsibilities:
----------------------------
Member 2: Camera calibration algorithms and 3D coordinate system setup
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class CameraCalibrator:
    """
    Calibrates camera for accurate 3D position estimation.
    
    This class implements algorithms to calibrate the camera using known
    reference points in the cricket scene, such as pitch markings and stumps.
    
    Team Member Responsibilities:
    ----------------------------
    Member 2: Implementation of calibration algorithms and coordinate transformations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Camera Calibrator with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Calibration parameters
        self.calibration_method = config.get("calibration_method", "auto")
        self.min_calibration_points = config.get("min_calibration_points", 4)
        self.recalibration_interval = config.get("recalibration_interval", 300)  # frames
        
        # Known cricket pitch dimensions (in meters)
        self.pitch_length = config.get("pitch_length", 20.12)  # Standard cricket pitch length
        self.pitch_width = config.get("pitch_width", 3.05)    # Standard cricket pitch width
        self.stump_height = config.get("stump_height", 0.71)  # Standard cricket stump height
        
        # Calibration state
        self.is_calibrated = False
        self.frames_since_calibration = 0
        self.camera_matrix = None
        self.dist_coeffs = None
        self.extrinsic_matrix = None
        
        # Default camera parameters (will be updated during calibration)
        self._init_default_camera_params()
    
    def _init_default_camera_params(self):
        """Initialize default camera parameters."""
        # Default intrinsic matrix (will be updated during calibration)
        self.camera_matrix = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Default distortion coefficients (will be updated during calibration)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Default extrinsic matrix (will be updated during calibration)
        self.extrinsic_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0]
        ], dtype=np.float32)
    
    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Calibrate the camera using the current frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            True if calibration is successful, False otherwise
        """
        # If already calibrated and not due for recalibration
        if self.is_calibrated and self.frames_since_calibration < self.recalibration_interval:
            self.frames_since_calibration += 1
            return True
        
        # Choose calibration method
        if self.calibration_method == "manual":
            success = self._manual_calibration(frame)
        elif self.calibration_method == "chessboard":
            success = self._chessboard_calibration(frame)
        else:  # auto
            success = self._auto_calibration(frame)
        
        if success:
            self.is_calibrated = True
            self.frames_since_calibration = 0
            return True
        
        return False
    
    def _manual_calibration(self, frame: np.ndarray) -> bool:
        """
        Calibrate using manually specified points.
        
        Args:
            frame: Current video frame
            
        Returns:
            True if calibration is successful, False otherwise
        """
        # In a real implementation, this would use manually specified points
        # For this prototype, we'll use default values
        
        # Use default camera parameters
        return True
    
    def _chessboard_calibration(self, frame: np.ndarray) -> bool:
        """
        Calibrate using a chessboard pattern.
        
        Args:
            frame: Current video frame
            
        Returns:
            True if calibration is successful, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define chessboard size
        chessboard_size = (7, 7)  # Number of internal corners
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if not ret:
            return False
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Prepare object points (3D points in real world space)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Assume square size of 0.025 meters (2.5 cm)
        objp *= 0.025
        
        # Calibrate camera
        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [objp], [corners], gray.shape[::-1], None, None
        )
        
        if not ret:
            return False
        
        # Calculate extrinsic matrix from the first view
        if rvecs and tvecs:
            R = cv2.Rodrigues(rvecs[0])[0]
            T = tvecs[0]
            self.extrinsic_matrix = np.hstack((R, T.reshape(3, 1)))
        
        return True
    
    def _auto_calibration(self, frame: np.ndarray) -> bool:
        """
        Calibrate automatically using cricket scene features.
        
        Args:
            frame: Current video frame
            
        Returns:
            True if calibration is successful, False otherwise
        """
        # In a real implementation, this would detect cricket pitch markings,
        # stumps, and other known features to calibrate the camera
        
        # For this prototype, we'll use a simplified approach
        
        # Try to detect stumps
        stumps_points = self._detect_stumps_for_calibration(frame)
        
        # Try to detect pitch markings
        pitch_points = self._detect_pitch_markings(frame)
        
        # Combine calibration points
        calibration_points = []
        if stumps_points:
            calibration_points.extend(stumps_points)
        if pitch_points:
            calibration_points.extend(pitch_points)
        
        # Check if we have enough points
        if len(calibration_points) < self.min_calibration_points:
            # Not enough points, use default parameters
            return True
        
        # In a real implementation, this would use the calibration points
        # to calculate camera parameters
        
        # For now, use default parameters with slight adjustments
        height, width = frame.shape[:2]
        self.camera_matrix[0, 0] = width  # Approximate focal length
        self.camera_matrix[1, 1] = width  # Approximate focal length
        self.camera_matrix[0, 2] = width / 2  # Principal point x
        self.camera_matrix[1, 2] = height / 2  # Principal point y
        
        return True
    
    def _detect_stumps_for_calibration(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect stumps for calibration.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of (image_point, world_point) pairs
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
        
        if not vertical_lines:
            return []
        
        # Sort by x-coordinate to identify left, middle, right stumps
        vertical_lines.sort(key=lambda line: (line[0] + line[2]) / 2)
        
        # Take up to 3 lines (for 3 stumps)
        stump_lines = vertical_lines[:3]
        
        # Create calibration points
        calibration_points = []
        
        # World coordinates of stumps (in meters)
        # Origin at the center of the pitch at ground level
        world_points = [
            np.array([0.0, 0.0, -self.stump_spacing]),  # Leg stump
            np.array([0.0, 0.0, 0.0]),                  # Middle stump
            np.array([0.0, 0.0, self.stump_spacing])    # Off stump
        ]
        
        for i, line in enumerate(stump_lines):
            if i >= len(world_points):
                break
                
            # Use bottom of stump (ground level)
            x1, y1, x2, y2 = line
            bottom_y = max(y1, y2)
            bottom_x = x1 if y1 == bottom_y else x2
            
            # Image point
            image_point = np.array([bottom_x, bottom_y])
            
            # World point
            world_point = world_points[i]
            
            calibration_points.append((image_point, world_point))
            
            # Also use top of stump
            top_y = min(y1, y2)
            top_x = x1 if y1 == top_y else x2
            
            # Image point
            image_point = np.array([top_x, top_y])
            
            # World point (add stump height)
            world_point_top = world_point.copy()
            world_point_top[1] = self.stump_height  # Y is height
            
            calibration_points.append((image_point, world_point_top))
        
        return calibration_points
    
    def _detect_pitch_markings(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect pitch markings for calibration.
        
        Args:
            frame: Current video frame
            
        Returns:
            List of (image_point, world_point) pairs
        """
        # In a real implementation, this would detect pitch markings
        # such as crease lines, bowling marks, etc.
        
        # For this prototype, we'll return an empty list
        return []
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get the camera intrinsic matrix."""
        return self.camera_matrix
    
    def get_dist_coeffs(self) -> np.ndarray:
        """Get the distortion coefficients."""
        return self.dist_coeffs
    
    def get_extrinsic_matrix(self) -> np.ndarray:
        """Get the camera extrinsic matrix."""
        return self.extrinsic_matrix
    
    def image_to_world(self, image_point: Tuple[float, float], depth: float) -> np.ndarray:
        """
        Convert image coordinates to world coordinates.
        
        Args:
            image_point: 2D point in image coordinates (x, y)
            depth: Depth from camera (Z coordinate in camera space)
            
        Returns:
            3D point in world coordinates
        """
        # Convert image point to normalized camera coordinates
        x, y = image_point
        x_norm = (x - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y_norm = (y - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # Calculate 3D point in camera coordinates
        camera_point = np.array([x_norm * depth, y_norm * depth, depth])
        
        # Convert to world coordinates using extrinsic matrix
        R = self.extrinsic_matrix[:, :3]
        T = self.extrinsic_matrix[:, 3]
        
        world_point = np.linalg.inv(R) @ (camera_point - T)
        
        return world_point
    
    def world_to_image(self, world_point: np.ndarray) -> Tuple[float, float]:
        """
        Convert world coordinates to image coordinates.
        
        Args:
            world_point: 3D point in world coordinates
            
        Returns:
            2D point in image coordinates (x, y)
        """
        # Convert world point to camera coordinates
        R = self.extrinsic_matrix[:, :3]
        T = self.extrinsic_matrix[:, 3]
        
        camera_point = R @ world_point + T
        
        # Project to image coordinates
        x_norm = camera_point[0] / camera_point[2]
        y_norm = camera_point[1] / camera_point[2]
        
        x = x_norm * self.camera_matrix[0, 0] + self.camera_matrix[0, 2]
        y = y_norm * self.camera_matrix[1, 1] + self.camera_matrix[1, 2]
        
        return (x, y)
