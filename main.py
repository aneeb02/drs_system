# -*- coding: utf-8 -*-

"""
Ball and Object Tracking Module - Main Entry Point

This module serves as the main entry point for the Ball and Object Tracking Module
of the DRS (Decision Review System). It orchestrates the workflow between different
components and manages the overall processing pipeline.

The module is designed to be run on a mobile device, processing video frames to track
cricket balls, stumps, batsman, and bat positions in real-time.

Team Member Responsibilities:
----------------------------
This file should be maintained collaboratively, with the Team Lead (Member 1)
having primary responsibility for the overall architecture and integration.
"""

import os
import time
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from frame_processor import FrameProcessor
from object_detector import ObjectDetector
from ball_tracker import BallTracker
from player_tracker import PlayerTracker
from stump_detector import StumpDetector
from calibration import CameraCalibrator
from data_models import TrackingData, BallTrajectory, BatsmanData, BatData, StumpsData, PitchData
from utils import setup_logging, calculate_fps, visualize_results

# Setup logging
logger = logging.getLogger(__name__)

class BallTrackingModule:
    """
    Main class for the Ball and Object Tracking Module.
    
    This class orchestrates the entire tracking process, from receiving input frames
    to producing the final tracking output data.
    
    Team Member Responsibilities:
    ----------------------------
    Team Lead (Member 1): Overall architecture, integration, and performance optimization
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Ball Tracking Module with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        setup_logging(self.config.get("logging", {}))
        logger.info("Initializing Ball Tracking Module")
        
        # Initialize components
        self.frame_processor = FrameProcessor(self.config.get("frame_processor", {}))
        self.object_detector = ObjectDetector(self.config.get("object_detector", {}))
        self.ball_tracker = BallTracker(self.config.get("ball_tracker", {}))
        self.player_tracker = PlayerTracker(self.config.get("player_tracker", {}))
        self.stump_detector = StumpDetector(self.config.get("stump_detector", {}))
        self.calibrator = CameraCalibrator(self.config.get("calibration", {}))
        
        # Module state
        self.is_calibrated = False
        self.frame_count = 0
        self.processing_times = []
        self.historical_ball_positions = []
        
        logger.info("Ball Tracking Module initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return {}
    
    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single frame and return tracking data.
        
        Args:
            frame_data: Dictionary containing frame data and metadata
            
        Returns:
            Dictionary containing tracking results
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Extract frame and metadata
        frame = self.frame_processor.decode_frame(frame_data)
        
        # Calibrate camera if needed
        if not self.is_calibrated:
            self.is_calibrated = self.calibrator.calibrate(frame)
            if self.is_calibrated:
                logger.info("Camera calibration completed successfully")
        
        # Detect objects in the frame
        detections = self.object_detector.detect(frame)
        
        # Track ball
        ball_data = self.ball_tracker.track(frame, detections, self.historical_ball_positions)
        if ball_data and 'current_position' in ball_data:
            self.historical_ball_positions.append({
                'frame_id': frame_data.get('frame_id'),
                'position': ball_data['current_position'],
                'timestamp': frame_data.get('timestamp')
            })
            # Keep only recent positions
            max_history = self.config.get("max_ball_history", 30)
            if len(self.historical_ball_positions) > max_history:
                self.historical_ball_positions = self.historical_ball_positions[-max_history:]
        
        # Track batsman and bat
        batsman_data = self.player_tracker.track_batsman(frame, detections)
        bat_data = self.player_tracker.track_bat(frame, detections, batsman_data)
        
        # Detect stumps
        stumps_data = self.stump_detector.detect(frame, detections)
        
        # Calculate pitch data
        pitch_data = self._calculate_pitch_data(frame, detections, ball_data)
        
        # Prepare tracking data
        tracking_data = self._prepare_tracking_data(
            frame_data, ball_data, batsman_data, bat_data, stumps_data, pitch_data
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Log performance metrics periodically
        if self.frame_count % 100 == 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            logger.info(f"Processed {self.frame_count} frames. Avg time: {avg_time:.4f}s, FPS: {fps:.2f}")
            self.processing_times = self.processing_times[-100:]  # Keep only recent times
        
        return tracking_data
    
    def _calculate_pitch_data(self, frame, detections, ball_data):
        """Calculate pitch-related data including bounce points."""
        # This would include logic to determine the pitch dimensions and bounce points
        # For now, return a placeholder
        return {
            "pitch_map": {
                "length": 20.12,
                "width": 3.05,
                "center": {"x": 0.0, "y": 0.0, "z": 0.0}
            },
            "bounce_point": None  # Will be populated when a bounce is detected
        }
    
    def _prepare_tracking_data(self, frame_data, ball_data, batsman_data, bat_data, stumps_data, pitch_data):
        """Prepare the final tracking data output."""
        return {
            "tracking_data": {
                "frame_id": frame_data.get("frame_id"),
                "timestamp": frame_data.get("timestamp"),
                "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "delivery_id": frame_data.get("sequence_metadata", {}).get("delivery_id"),
                "match_id": frame_data.get("sequence_metadata", {}).get("match_id"),
                "confidence_score": self._calculate_overall_confidence(
                    ball_data, batsman_data, bat_data, stumps_data
                )
            },
            "ball_trajectory": ball_data,
            "batsman_data": batsman_data,
            "bat_data": bat_data,
            "stumps_data": stumps_data,
            "pitch_data": pitch_data
        }
    
    def _calculate_overall_confidence(self, ball_data, batsman_data, bat_data, stumps_data):
        """Calculate overall confidence score based on individual detection confidences."""
        confidences = []
        
        if ball_data and 'detection_confidence' in ball_data:
            confidences.append(ball_data['detection_confidence'])
        
        if batsman_data and 'detection_confidence' in batsman_data:
            confidences.append(batsman_data['detection_confidence'])
        
        if bat_data and 'detection_confidence' in bat_data:
            confidences.append(bat_data['detection_confidence'])
        
        if stumps_data and 'detection_confidence' in stumps_data:
            confidences.append(stumps_data['detection_confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ball and Object Tracking Module')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Path to input video file (for testing)')
    parser.add_argument('--output', type=str, help='Path to output file for tracking data')
    parser.add_argument('--visualize', action='store_true', help='Visualize tracking results')
    return parser.parse_args()

def main():
    """Main entry point for the module."""
    args = parse_args()
    
    # Initialize the module
    tracker = BallTrackingModule(args.config)
    
    # For testing with a video file
    if args.input:
        import cv2
        
        cap = cv2.VideoCapture(args.input)
        output_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create frame data dictionary
            frame_data = {
                "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
                "resolution": {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                },
                "sequence_metadata": {
                    "delivery_id": "test-delivery",
                    "match_id": "test-match"
                }
            }
            
            # Process the frame
            result = tracker.process_frame(frame_data)
            output_data.append(result)
            
            # Visualize if requested
            if args.visualize:
                visualize_results(frame, result)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save output data if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    else:
        logger.info("No input specified. In a real deployment, this would connect to the camera stream.")
        # In a real deployment, this would connect to the camera stream or message queue

if __name__ == "__main__":
    main()
