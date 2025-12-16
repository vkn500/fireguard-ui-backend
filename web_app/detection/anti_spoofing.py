"""
anti_spoofing.py - Detect real fire vs video/image spoofing
Add this to detection/ folder
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TemporalConsistencyFilter:
    """Distinguishes real fire from photos/videos"""
    
    def __init__(self, window_size=30, min_frames_for_alert=10):
        self.window_size = window_size
        self.min_frames_for_alert = min_frames_for_alert
        self.detection_history = deque(maxlen=window_size)
        self.last_fire_frame_idx = None
        
    def add_detection(self, label, confidence):
        """Add detection to history"""
        self.detection_history.append({
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
    
    def is_real_fire(self):
        """
        Check if REAL fire or video/image
        Returns: (is_real: bool, reason: str, confidence: float)
        """
        
        if len(self.detection_history) < 3:
            return False, "Insufficient history", 0.0
        
        recent = list(self.detection_history)[-30:]
        fire_count = sum(1 for d in recent if d['label'] == 'fire')
        
        # TEST 1: Sudden appearance = FAKE
        first_fire_idx = None
        for i, detection in enumerate(recent):
            if detection['label'] == 'fire':
                first_fire_idx = i
                break
        
        if first_fire_idx is not None:
            if first_fire_idx < 3 and self.last_fire_frame_idx is None:
                return False, "Fire appeared suddenly (video/image)", 0.3
        
        # TEST 2: Persistence check
        if fire_count < self.min_frames_for_alert:
            return False, "Fire detection too brief", 0.4
        
        # TEST 3: Confidence stability
        fire_confidences = [d['confidence'] for d in recent if d['label'] == 'fire']
        
        if fire_confidences:
            avg_conf = np.mean(fire_confidences)
            conf_std = np.std(fire_confidences)
            
            if conf_std < 0.03 and avg_conf > 0.75:
                return False, "Confidence too consistent (video)", 0.4
            
            if avg_conf > 0.5:
                self.last_fire_frame_idx = len(recent) - 1
        
        # TEST 4: Duration pattern
        fire_frame_indices = [i for i, d in enumerate(recent) if d['label'] == 'fire']
        
        if len(fire_frame_indices) > 5:
            gaps = []
            for i in range(1, len(fire_frame_indices)):
                gap = fire_frame_indices[i] - fire_frame_indices[i-1]
                if gap > 2:
                    gaps.append(gap)
            
            if len(gaps) > 2:
                return False, "Fire detection interrupted (image)", 0.35
        
        # Passed all tests
        if fire_count >= self.min_frames_for_alert:
            confidence = 0.7 + (fire_count / len(recent)) * 0.3
            return True, "Real fire verified", confidence
        
        return False, "Insufficient fire detections", 0.4
    
    def reset(self):
        """Reset history"""
        self.detection_history.clear()
        self.last_fire_frame_idx = None


class MotionAnalyzer:
    """Detect motion pattern - for production"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
    
    def analyze(self, frame):
        """Check motion pattern"""
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) < 2:
            return None, "Insufficient frames"
        
        try:
            gray1 = cv2.cvtColor(self.frame_buffer[-2], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_motion = np.mean(mag)
            
            if mean_motion < 0.05:
                return False, "Static image"
            elif mean_motion < 0.1:
                return False, "Low motion"
            else:
                return True, "Good motion"
        
        except Exception as e:
            logger.warning(f"Motion analysis error: {e}")
            return None, "Error"


class BrightnessAnalyzer:
    """Real fire increases brightness"""
    
    def __init__(self, window_size=15):
        self.brightness_history = deque(maxlen=window_size)
    
    def analyze(self, frame):
        """Check brightness change"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        
        if len(self.brightness_history) < 5:
            return None, "Insufficient history"
        
        recent = list(self.brightness_history)
        avg_recent = np.mean(recent[-3:])
        avg_earlier = np.mean(recent[:3])
        brightness_change = avg_recent - avg_earlier
        
        if brightness_change > 8:
            return True, "Brightness increasing"
        elif brightness_change < -5:
            return False, "Brightness decreasing"
        else:
            return None, "Inconclusive"


class RobustFireDetector:
    """PRODUCTION: All methods combined"""
    
    def __init__(self):
        self.temporal_filter = TemporalConsistencyFilter()
        self.motion_analyzer = MotionAnalyzer()
        self.brightness_analyzer = BrightnessAnalyzer()
        
        self.REAL_FIRE_THRESHOLD = 0.65
        self.WEIGHTS = {
            'temporal': 0.50,
            'motion': 0.25,
            'brightness': 0.25
        }
    
    def should_alert(self, frame, yolo_label, yolo_confidence):
        """Should we send alert?"""
        
        self.temporal_filter.add_detection(yolo_label, yolo_confidence)
        
        if yolo_label == 'no_fire':
            return False, {'reason': 'No fire detected', 'scores': {}, 'final_confidence': 0.0}
        
        scores = {}
        
        # Temporal
        is_temporal_valid, _, _ = self.temporal_filter.is_real_fire()
        scores['temporal'] = 1.0 if is_temporal_valid else 0.0
        
        # Motion
        has_motion, _ = self.motion_analyzer.analyze(frame)
        scores['motion'] = 1.0 if has_motion else (0.5 if has_motion is None else 0.0)
        
        # Brightness
        brightness_inc, _ = self.brightness_analyzer.analyze(frame)
        scores['brightness'] = 1.0 if brightness_inc else (0.5 if brightness_inc is None else 0.0)
        
        final_confidence = (
            scores['temporal'] * self.WEIGHTS['temporal'] +
            scores['motion'] * self.WEIGHTS['motion'] +
            scores['brightness'] * self.WEIGHTS['brightness']
        )
        
        should_alert = final_confidence >= self.REAL_FIRE_THRESHOLD
        
        return should_alert, {
            'scores': scores,
            'final_confidence': final_confidence
        }
