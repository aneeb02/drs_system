#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ball Tracker Module

This module is responsible for tracking the cricket ball across video frames,
calculating its trajectory, velocity, acceleration, and spin.

Team Member Responsibilities:
----------------------------
Member 4: Ball tracking algorithms, trajectory calculation, and physics modeling
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import math

class BallTracker:
    """
    Tracks cricket ball across frames and calculates trajectory data.
    
    This class implements algorithms to track the ball's position across frames,
    calculate its 3D coordinates, velocity, acceleration, and spin.
    
    Team Member Responsibilities:
    ----------------------------
    Member 4: Implementation of tracking algorithms and physics calculations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ball Tracker with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Kalman filter parameters
        self.use_kalman = config.get("use_kalman", True)
        if self.use_kalman:
            self._init_kalman_filter()
        
        # Tracking parameters
        self.max_tracking_distance = config.get("max_tracking_distance", 100)
        self.min_detection_confidence = config.get("min_detection_confidence", 0.5)
        
        # Physics calculation parameters
        self.gravity = config.get("gravity", 9.8)  # m/s²
        self.air_resistance_coef = config.get("air_resistance", 0.1)
        self.frame_rate = config.get("frame_rate", 30)  # fps
        
        # 3D reconstruction parameters
        self.camera_matrix = None  # Will be set during calibration
        self.dist_coeffs = None    # Will be set during calibration
        
        # State variables
        self.is_tracking = False
        self.last_position = None
        self.last_velocity = None
        self.last_acceleration = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = config.get("max_lost_frames", 10)
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for ball tracking."""
        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        self.kalman = cv2.KalmanFilter(9, 3)
        
        # Measurement matrix (we only measure position)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], np.float32)
        
        # Transition matrix
        # x' = x + vx*dt + 0.5*ax*dt²
        # vx' = vx + ax*dt
        # ax' = ax
        dt = 1.0 / self.frame_rate
        dt2 = 0.5 * dt * dt
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0, dt2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, dt2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, dt2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(9, dtype=np.float32) * 0.01
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(9, dtype=np.float32)
    
    def set_calibration(self, camera_matrix, dist_coeffs):
        """
        Set camera calibration parameters for 3D reconstruction.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def track(self, frame: np.ndarray, detections: Dict[str, List[Dict[str, Any]]], 
              historical_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Track the ball in the current frame and calculate trajectory data.
        
        Args:
            frame: Current video frame
            detections: Object detection results
            historical_positions: Previous ball positions
            
        Returns:
            Dictionary containing ball trajectory data
        """
        # Extract ball detections
        ball_detections = detections.get("ball", [])
        
        # If no ball detected
        if not ball_detections:
            return self._handle_missing_detection(historical_positions)
        
        # Get the most confident ball detection
        ball_detection = max(ball_detections, key=lambda x: x["confidence"])
        
        # Check if confidence is high enough
        if ball_detection["confidence"] < self.min_detection_confidence:
            return self._handle_missing_detection(historical_positions)
        
        # Extract ball position
        center = ball_detection["center"]
        radius = ball_detection.get("radius", 10)
        
        # Convert to 3D coordinates
        position_3d = self._estimate_3d_position(center, radius, frame.shape)
        
        # Update tracking state
        if not self.is_tracking:
            self._initialize_tracking(position_3d)
        else:
            self._update_tracking(position_3d)
        
        # Calculate trajectory data
        return self._calculate_trajectory_data(position_3d, ball_detection["confidence"], 
                                              historical_positions)
    
    def _handle_missing_detection(self, historical_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle the case when ball is not detected in the current frame.
        
        Args:
            historical_positions: Previous ball positions
            
        Returns:
            Estimated ball trajectory data or None
        """
        if not self.is_tracking:
            return None
        
        self.tracking_lost_frames += 1
        
        # If tracking is lost for too many frames, reset tracking
        if self.tracking_lost_frames > self.max_lost_frames:
            self.is_tracking = False
            return None
        
        # Predict position using Kalman filter or physics model
        if self.use_kalman:
            predicted_state = self.kalman.predict()
            predicted_position = predicted_state[:3].flatten()
        else:
            # Simple physics prediction
            predicted_position = self._predict_position_physics()
        
        # Update state variables
        self.last_position = predicted_position
        
        # Calculate trajectory data with lower confidence
        confidence = max(0.1, 0.9 - 0.08 * self.tracking_lost_frames)
        return self._calculate_trajectory_data(predicted_position, confidence, historical_positions)
    
    def _initialize_tracking(self, position: np.ndarray):
        """
        Initialize tracking with the first detected position.
        
        Args:
            position: 3D position of the ball
        """
        self.is_tracking = True
        self.last_position = position
        self.last_velocity = np.zeros(3)
        self.last_acceleration = np.array([0, -self.gravity, 0])
        self.tracking_lost_frames = 0
        
        if self.use_kalman:
            # Initialize Kalman filter state
            self.kalman.statePost = np.zeros((9, 1), np.float32)
            self.kalman.statePost[:3] = position.reshape(3, 1)
            self.kalman.statePost[6:] = np.array([[0], [-self.gravity], [0]], np.float32)
    
    def _update_tracking(self, position: np.ndarray):
        """
        Update tracking with a new detected position.
        
        Args:
            position: 3D position of the ball
        """
        dt = 1.0 / self.frame_rate
        
        if self.use_kalman:
            # Update Kalman filter
            measurement = position.reshape(3, 1).astype(np.float32)
            self.kalman.correct(measurement)
            
            # Get updated state
            state = self.kalman.statePost
            self.last_position = state[:3].flatten()
            self.last_velocity = state[3:6].flatten()
            self.last_acceleration = state[6:].flatten()
        else:
            # Calculate velocity and acceleration
            new_velocity = (position - self.last_position) / dt
            new_acceleration = (new_velocity - self.last_velocity) / dt
            
            # Apply smoothing
            alpha = 0.7  # Smoothing factor
            self.last_velocity = alpha * new_velocity + (1 - alpha) * self.last_velocity
            self.last_acceleration = alpha * new_acceleration + (1 - alpha) * self.last_acceleration
            self.last_position = position
        
        self.tracking_lost_frames = 0
    
    def _predict_position_physics(self) -> np.ndarray:
        """
        Predict next position using physics model.
        
        Returns:
            Predicted 3D position
        """
        dt = 1.0 / self.frame_rate
        
        # Apply physics equations of motion
        # x = x0 + v0*t + 0.5*a*t²
        predicted_position = (
            self.last_position + 
            self.last_velocity * dt + 
            0.5 * self.last_acceleration * dt * dt
        )
        
        # Update velocity for next prediction
        # v = v0 + a*t
        self.last_velocity = self.last_velocity + self.last_acceleration * dt
        
        # Apply air resistance (simplified)
        speed = np.linalg.norm(self.last_velocity)
        if speed > 0:
            drag = self.air_resistance_coef * speed * speed
            drag_acceleration = -drag * self.last_velocity / speed
            self.last_acceleration = np.array([
                drag_acceleration[0],
                -self.gravity + drag_acceleration[1],
                drag_acceleration[2]
            ])
        
        return predicted_position
    
    def _estimate_3d_position(self, center: Tuple[int, int], radius: float, 
                             frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Estimate 3D position from 2D detection.
        
        In a real implementation, this would use camera calibration and
        possibly multiple camera views for accurate 3D reconstruction.
        
        Args:
            center: 2D center of the ball in the image
            radius: Radius of the ball in pixels
            frame_shape: Shape of the frame
            
        Returns:
            Estimated 3D position [x, y, z]
        """
        # If camera is calibrated, use proper 3D reconstruction
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # This would use proper 3D reconstruction techniques
            # For now, we'll use a simplified approach
            pass
        
        # Simplified approach: use image coordinates and estimated depth
        # This is a placeholder for demonstration
        x_image, y_image = center
        height, width = frame_shape[:2]
        
        # Normalize image coordinates to [-1, 1]
        x_norm = (x_image - width/2) / (width/2)
        y_norm = (y_image - height/2) / (height/2)
        
        # Estimate depth from ball size (inverse relationship)
        # Assuming a standard cricket ball size and camera parameters
        # This is highly simplified and would be replaced with proper depth estimation
        z_depth = 20.0 / radius if radius > 0 else 10.0
        
        # Convert to world coordinates (simplified)
        # In a real implementation, this would use proper camera extrinsics
        x_world = x_norm * z_depth
        y_world = -y_norm * z_depth  # Invert y-axis
        
        return np.array([x_world, y_world, z_depth])
    
    def _calculate_trajectory_data(self, position: np.ndarray, confidence: float,
                                  historical_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate complete ball trajectory data.
        
        Args:
            position: Current 3D position
            confidence: Detection confidence
            historical_positions: Previous ball positions
            
        Returns:
            Dictionary containing ball trajectory data
        """
        # Calculate spin (in a real implementation, this would use more sophisticated techniques)
        spin_axis, spin_rate = self._estimate_spin(historical_positions)
        
        return {
            "current_position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2])
            },
            "velocity": {
                "x": float(self.last_velocity[0]),
                "y": float(self.last_velocity[1]),
                "z": float(self.last_velocity[2])
            },
            "acceleration": {
                "x": float(self.last_acceleration[0]),
                "y": float(self.last_acceleration[1]),
                "z": float(self.last_acceleration[2])
            },
            "spin": {
                "axis": {
                    "x": float(spin_axis[0]),
                    "y": float(spin_axis[1]),
                    "z": float(spin_axis[2])
                },
                "rate": float(spin_rate)
            },
            "detection_confidence": float(confidence),
            "historical_positions": historical_positions[-10:] if historical_positions else []
        }
    
    def _estimate_spin(self, historical_positions: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
        """
        Estimate ball spin axis and rate from historical positions.
        
        In a real implementation, this would use more sophisticated techniques
        such as analyzing ball texture or markings across frames.
        
        Args:
            historical_positions: Previous ball positions
            
        Returns:
            Tuple of (spin_axis, spin_rate)
        """
        # Placeholder implementation
        # In a real system, this would use computer vision to track ball rotation
        
        # Default values
        spin_axis = np.array([0.1, 0.8, 0.2])
        spin_rate = 25.5
        
        # Normalize spin axis
        norm = np.linalg.norm(spin_axis)
        if norm > 0:
            spin_axis = spin_axis / norm
        
        return spin_axis, spin_rate
