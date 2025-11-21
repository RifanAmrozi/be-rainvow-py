import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import math
import json
import os
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from pathlib import Path
from app.service.storage import upload_video_to_supabase, upload_photo_to_supabase

class DetectionPhase(Enum):
    """Fase deteksi shoplifting"""
    IDLE = "idle"
    WALKING = "walking"
    REACHING_SHELF = "reaching_shelf"
    GRABBING = "grabbing"
    SUSPICIOUS_MOVEMENT = "suspicious"
    ALERT = "alert"

class SuspiciousPose(Enum):
    BENDING_DOWN = "bending_down"
    CROUCHING = "crouching"
    HIDING_UNDER_CLOTHING = "hiding_under_clothing"
    REACHING_POCKET = "reaching_pocket"
    HANDS_NEAR_BODY = "hands_near_body"
    PUTTING_IN_PANTS_POCKET = "putting_in_pants_pocket"
    HANDS_BEHIND_BACK = "hands_behind_back"
    SQUATTING_LOW = "squatting_low"
    ZONE_PANTS_POCKET_LEFT = "zone_pants_pocket_left"
    ZONE_PANTS_POCKET_RIGHT = "zone_pants_pocket_right"
    ZONE_JACKET_POCKET_LEFT = "zone_jacket_pocket_left"
    ZONE_JACKET_POCKET_RIGHT = "zone_jacket_pocket_right"
    HIDING_IN_HAT = "hiding_in_hat"
    HAND_ON_HEAD = "hand_on_head"
    HIDING_IN_SHOE_LEFT = "hiding_in_shoe_left"     
    HIDING_IN_SHOE_RIGHT = "hiding_in_shoe_right"  
    REACHING_SHOE_AREA = "reaching_shoe_area"       

class PocketZone:
    
    def __init__(self, zone_type, left_point, right_point, width_factor=0.35, depth_factor=0.3):
        self.zone_type = zone_type
        self.left_point = left_point
        self.right_point = right_point
        self.width_factor = width_factor
        self.depth_factor = depth_factor
        self.zone_box = None
    
    def calculate_zone(self, shoulder_width):
        """Hitung bounding box zona kantong"""
        if not self.left_point or not self.right_point:
            return None
        
        zone_width = shoulder_width * self.width_factor
        
        if 'pants' in self.zone_type:
            zone_height = shoulder_width * 0.5        
            x_center = (self.left_point[0] + self.right_point[0]) / 2
            y_center = (self.left_point[1] + self.right_point[1]) / 2 - shoulder_width * 0.2 
        else:
            zone_height = shoulder_width * 0.5       
            x_center = (self.left_point[0] + self.right_point[0]) / 2
            y_center = (self.left_point[1] + self.right_point[1]) / 2 + shoulder_width * 0.15  
        
        if 'left' in self.zone_type:
            x1 = x_center - zone_width * 1.0      
            x2 = x_center + zone_width * 0.6      
        else:
            x1 = x_center - zone_width * 0.6     
            x2 = x_center + zone_width * 1.0     
        
        y1 = y_center
        y2 = y_center + zone_height
        
        self.zone_box = (int(x1), int(y1), int(x2), int(y2))
        return self.zone_box


    
    def is_point_in_zone(self, point):
        """Cek apakah point masuk ke zona"""
        if not self.zone_box or not point:
            return False
        
        x1, y1, x2, y2 = self.zone_box
        x, y = point[0], point[1]
        
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_penetration_depth(self, point):
        """Hitung seberapa dalam tangan masuk ke zona (0-1)"""
        if not self.zone_box or not point:
            return 0
        
        x1, y1, x2, y2 = self.zone_box
        x, y = point[0], point[1]
        
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return 0
        
        if 'left' in self.zone_type:
            depth = (x - x1) / (x2 - x1) if (x2 - x1) > 0 else 0
        else:
            depth = (x2 - x) / (x2 - x1) if (x2 - x1) > 0 else 0
        
        return depth

class ThreadedRTSPCapture:
    """
    RTSP Capture dengan threading untuk eliminate frame delays
    Frame baru langsung di-grab di background thread
    """
    
    def __init__(self, rtsp_url, buffer_size=1, name="RTSPCapture"):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.name = name
        
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = None
        self.cap = None
        
        self.frames_read = 0
        self.frames_dropped = 0
        self.last_frame_time = time.time()
        
    def start(self):
        """Start capture thread"""
        print(f"🎬 Starting threaded RTSP capture: {self.name}")
        

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP: {self.rtsp_url}")
        
        self.stopped = False
        self.thread = threading.Thread(target=self._update, name=self.name, daemon=True)
        self.thread.start()
        
        print(f"✅ Threaded capture started: {self.name}")
        return self
        
    def _update(self):
        """Background thread untuk grab frames"""
        consecutive_failures = 0
        max_failures = 30
        
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"⚠️  [{self.name}] Frame read failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print(f"❌ [{self.name}] Too many failures, reconnecting...")
                    self._reconnect()
                    consecutive_failures = 0
                
                time.sleep(0.05)
                continue
            
            consecutive_failures = 0
            self.frames_read += 1
            self.last_frame_time = time.time()
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                self.frames_dropped += 1
    
    def _reconnect(self):
        """Reconnect RTSP stream"""
        if self.cap is not None:
            self.cap.release()
        
        time.sleep(2)
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                print(f"✅ [{self.name}] Reconnected")
            else:
                print(f"❌ [{self.name}] Reconnection failed")
        except Exception as e:
            print(f"❌ [{self.name}] Reconnection error: {e}")
    
    def read(self):
        """Read latest frame from queue"""
        if self.frame_queue.empty():
            return False, None
        
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop capture thread"""
        print(f"🛑 Stopping {self.name}...")
        self.stopped = True
        
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        if self.cap is not None:
            self.cap.release()
        
        print(f"✅ {self.name} stopped")
        print(f"   Frames read: {self.frames_read}")
        print(f"   Frames dropped: {self.frames_dropped}")
        print(f"   Drop rate: {(self.frames_dropped / max(self.frames_read, 1) * 100):.1f}%")
    
    def get_stats(self):
        """Get capture statistics"""
        return {
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / max(self.frames_read, 1),
            'is_alive': self.thread.is_alive() if self.thread else False,
            'time_since_last_frame': time.time() - self.last_frame_time
        }

class ShopliftingPoseDetectorWithGrab:
    def __init__(self, pose_model="yolo11m-pose.pt", debug_mode=False, high_angle_mode=False):
        print("🚀 Initializing Shoplifting Detector...")
        print("   [SHOPLIFTING DETECTION]   ")
        
        try:
            self.pose_model = YOLO(pose_model)
            print("✅ Pose detection model loaded")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        self.frame_count = 0
        self.debug_mode = debug_mode
        self.high_angle_mode = high_angle_mode  
    
        if self.high_angle_mode:
            print("   [HIGH-ANGLE CCTV MODE ENABLED]")
            print("   - Relaxed horizontal extension requirements")
            print("   - Focus on downward + forward movement")
            print("   - Lower confidence thresholds")

        self.inference_size = 416
        
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        self.person_tracks = defaultdict(lambda: {
            'phase': DetectionPhase.IDLE,
            'phase_start_frame': 0,

            # WALKING DETECTION
            'position_history': deque(maxlen=10), 
            'is_walking': False,
            'walking_frames': 0,
            'stationary_frames': 0,
            'last_movement_speed': 0.0,
            'walking_direction': None, 
            'walking_with_item': False,  
            'pre_walking_phase': None,   

            
            # Grabbing detection
            'wrist_positions': deque(maxlen=20),
            'hand_extended': False,
            'hand_extended_frames': 0,
            'grab_detected': False,
            'grab_frame': 0,
            'grabbed_hand': None,
            'reach_type': None,

            'head_pose_frames': 0,            
            'head_pose_detected': False,        
            'head_pose_first_frame': 0,         
            'head_pose_max_confidence': 0.0,    
            
            # Suspicious pose tracking
            'suspicious_poses': deque(maxlen=30),
            'pose_counts': defaultdict(int),
            'alert_triggered': False,
            'last_alert_frame': 0,
            'suspicion_score': 0.0,
            'consecutive_suspicious': 0,
            'total_frames_tracked': 0,
            'suspicious_ratio': 0.0,
            'first_seen': time.time(),

            # Suspicious validation
            'suspicious_buffer': deque(maxlen=15),
            'suspicious_frame_count': 0,
            'last_normal_frame': 0,
            
            'zone_penetration_detected': False,
            'zone_penetration_frames': 0,
            'zone_penetration_zones': [],

            # History
            'phase_history': [],
            
            # Pocket zone tracking
            'pocket_zones': {},
            'wrist_in_zone_frames': defaultdict(int),
            'zone_entry_frames': defaultdict(int),
            'max_zone_depth': defaultdict(float),
            'zone_detections': defaultdict(list),
            'current_keypoints': None,

            'keypoint_history': deque(maxlen=10), 
            'stability_scores': defaultdict(lambda: deque(maxlen=10)),  
            'is_stable': True,
            'unstable_frames': 0,
            'last_stability_check': 0,

            'shoulder_orientation': deque(maxlen=10),  
            'hip_orientation': deque(maxlen=10),      
            'is_rotating': False,
            'rotation_frames': 0,
            'last_rotation_check': 0,
        })
        
        self.GRAB_THRESHOLDS = {
            'hand_extension_threshold': 75,              
            'hand_height_tolerance': 150,                 
            'min_extension_frames': 3,                  
            'grab_timeout': 120,
            'hand_close_distance': 60,
            'elbow_angle_extended': 70,                  
            'elbow_angle_grab': 140,
            'distance_reduction_threshold': 30,
            'velocity_threshold': 8,
            'reaching_down_horizontal_min': 10,         
            'reaching_down_position_ratio_min': 0.02,   
            'reaching_down_position_ratio_max': 3.0,     
            'reaching_down_hip_distance': 15,           
            'reaching_down_knee_proximity': 150,        
            'reaching_down_angle_threshold': 60,
            'squat_partial_hip_horizontal_min': 10, 
            'squat_ partial_confidence_min': 0.30,             
        }

        self.SUSPICIOUS_VALIDATION = {
            'min_suspicious_frames': 5,                    
            'suspicious_confidence_threshold': 0.70, 
            'head_pose_threshold': 0.50,
            'pose_consistency_window': 15,
            'min_unique_poses': 1,               
            'high_severity_poses': [
                SuspiciousPose.HIDING_UNDER_CLOTHING,
                SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
                SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
                SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
                SuspiciousPose.ZONE_JACKET_POCKET_RIGHT,
                SuspiciousPose.HIDING_IN_HAT,
                SuspiciousPose.HAND_ON_HEAD,
                SuspiciousPose.HIDING_IN_SHOE_LEFT,
                SuspiciousPose.HIDING_IN_SHOE_RIGHT,             
            ],
            'timeout_normal_behavior': 100
        }

        self.REQUIRED_KEYPOINTS = {
            'reaching': ['shoulder', 'elbow', 'wrist', 'hip'],  
            'grabbing': ['shoulder', 'elbow', 'wrist', 'hip'],
            'suspicious': ['shoulder', 'elbow', 'wrist', 'hip', 'knee'],
            'zone_detection': ['shoulder', 'elbow', 'wrist', 'hip']
        }

        self.KEYPOINT_QUALITY_THRESHOLDS = {
            'min_confidence': 0.65, 
            'min_visibility_ratio': 0.75,  
            'critical_keypoints': [ 
                'left_shoulder', 'right_shoulder',
                'left_hip', 'right_hip'
            ]
        }

        self.SUSPICIOUS_THRESHOLDS = {
            'bending_threshold': 0.55,
            'crouch_knee_angle': 110,
            'suspicious_frame_count': 10,
            'alert_cooldown': 90,
            'high_confidence_threshold': 0.90,
            'score_threshold': 75.0,
            'score_decay': 3.0,
            'continuous_pose_bonus': 15,
            'min_consecutive_for_bonus': 12,
            'min_tracking_frames': 15,
            'suspicious_ratio_threshold': 0.35
        }
        
        # Zone thresholds
        self.ZONE_THRESHOLDS = {
            'min_frames_in_zone': 1,              
            'min_penetration_depth': 0.15,        
            'high_confidence_depth': 0.30,        
            'zone_based_alert_score': 30,
            'immediate_suspicious_depth': 0.25,  
            'immediate_suspicious_frames': 1,    
        }

        self.NATURAL_POSITION_THRESHOLDS = {
            'max_horizontal_distance_from_hip': 70,   
            'min_elbow_angle_straight': 160,            
            'max_wrist_to_hip_distance_relax': 80,    
            'max_horizontal_offset_relax': 50,         
            'vertical_ratio_threshold': 0.25,            
        }
        
        self.alert_log = []
        self.session_start = datetime.now()
        self.frame_buffer = deque(maxlen=150)
        self.alert_clips_saved = []
        self.fps = 30
        
        self.recording_alerts = {}
        
        print("✅ Initialization complete")
    
    def get_keypoint(self, keypoints, name):
        """Get keypoint by name dengan confidence check"""
        idx = self.KEYPOINTS[name]
        if idx < len(keypoints):
            x, y, conf = keypoints[idx]
            return (float(x), float(y), float(conf)) if conf > 0.6 else None
        return None
    
    def calculate_angle(self, p1, p2, p3):
        """Hitung sudut antara 3 titik"""
        if not all([p1, p2, p3]):
            return None
        
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def smooth_keypoints(self, track_id, keypoints):
        """
        Smooth keypoints menggunakan moving average (reduce jitter)
        """
        track = self.person_tracks[track_id]
        
        if len(track['keypoint_history']) < 3:
            return keypoints 
        
        smoothed = keypoints.copy()
        
        # Apply moving average untuk setiap keypoint
        for kp_idx in range(len(keypoints)):
            if keypoints[kp_idx][2] > 0.6:  
                # Get last 3 positions
                positions = []
                for hist_kp in list(track['keypoint_history'])[-3:]:
                    if kp_idx < len(hist_kp) and hist_kp[kp_idx][2] > 0.6:
                        positions.append([hist_kp[kp_idx][0], hist_kp[kp_idx][1]])
                
                if len(positions) >= 2:
                    # Moving average
                    avg_pos = np.mean(positions, axis=0)
                    smoothed[kp_idx][0] = avg_pos[0]
                    smoothed[kp_idx][1] = avg_pos[1]
        
        return smoothed

    def calculate_keypoint_stability(self, track_id, keypoints):
        """
        Hitung stabilitas keypoints - apakah jittery/noisy?
        
        Returns:
            (is_stable, stability_score, jittery_keypoints)
        """
        track = self.person_tracks[track_id]
        
        # Save current keypoints
        track['keypoint_history'].append(keypoints.copy())
        
        # Need at least 5 frames untuk calculate stability
        if len(track['keypoint_history']) < 5:
            return True, 1.0, []  # Assume stable initially
        
        jittery_keypoints = []
        stability_scores = []
        
        # Check critical keypoints only (shoulders, elbows, wrists, hips)
        critical_indices = [
            self.KEYPOINTS['left_shoulder'], self.KEYPOINTS['right_shoulder'],
            self.KEYPOINTS['left_elbow'], self.KEYPOINTS['right_elbow'],
            self.KEYPOINTS['left_wrist'], self.KEYPOINTS['right_wrist'],
            self.KEYPOINTS['left_hip'], self.KEYPOINTS['right_hip']
        ]
        
        for kp_idx in critical_indices:
            # Get positions from last 5 frames
            positions = []
            for hist_kp in list(track['keypoint_history'])[-5:]:
                if kp_idx < len(hist_kp) and hist_kp[kp_idx][2] > 0.6:
                    positions.append([hist_kp[kp_idx][0], hist_kp[kp_idx][1]])
            
            if len(positions) < 3:
                continue
            
            # Calculate movement variance
            positions = np.array(positions)
            variance = np.var(positions, axis=0)
            total_variance = np.sum(variance)
            
            # Calculate movement velocity
            velocities = []
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[i-1])
                velocities.append(dist)
            
            avg_velocity = np.mean(velocities) if velocities else 0
                        
            is_kp_stable = total_variance < 50 and avg_velocity < 20
            
            if not is_kp_stable:
                kp_name = [k for k, v in self.KEYPOINTS.items() if v == kp_idx][0]
                jittery_keypoints.append(kp_name)
                stability_scores.append(0)
            else:
                stability_scores.append(1)
        
        # Overall stability score
        overall_stability = np.mean(stability_scores) if stability_scores else 0
        is_stable = overall_stability >= 0.50 
        return is_stable, overall_stability, jittery_keypoints
    
    def distance(self, p1, p2):
        """Euclidean distance"""
        if not all([p1, p2]):
            return None
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def detect_body_rotation(self, keypoints, track_id):
        """
        Deteksi apakah orang sedang merotasi badan
        Returns: (is_rotating, rotation_degree, confidence)
        """
        track = self.person_tracks[track_id]

        if not track['grab_detected']:
            return False, 0, 0.0
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return False, 0, 0.0
        
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        hip_width = abs(right_hip[0] - left_hip[0])
        
        shoulder_ratio = shoulder_width / (abs(right_shoulder[1] - left_shoulder[1]) + 1)
        hip_ratio = hip_width / (abs(right_hip[1] - left_hip[1]) + 1)
        
        # Simpan history
        track['shoulder_orientation'].append(shoulder_width)
        track['hip_orientation'].append(hip_width)
        
        # Deteksi perubahan orientasi yang cepat
        is_rotating = False
        rotation_confidence = 0.0
        
        if len(track['shoulder_orientation']) >= 5:
            recent_shoulders = list(track['shoulder_orientation'])[-5:]
            shoulder_variance = np.std(recent_shoulders)
            shoulder_mean = np.mean(recent_shoulders)
            
            if shoulder_mean > 0:
                cv = shoulder_variance / shoulder_mean
                
                if cv > 0.08:
                    is_rotating = True
                    rotation_confidence = min(1.0, cv / 0.3)
                    
                    if recent_shoulders[-1] < recent_shoulders[0] * 0.8:
                        rotation_type = "turning_sideways"
                    elif recent_shoulders[-1] > recent_shoulders[0] * 1.2:
                        rotation_type = "turning_front"
                    else:
                        rotation_type = "rotating"
        
        # Update tracking state
        if is_rotating:
            track['is_rotating'] = True
            track['rotation_frames'] += 1
        else:
            # Decay rotation state 
            if track['rotation_frames'] > 2: 
                track['rotation_frames'] -= 1
            elif track['rotation_frames'] > 0:
                track['rotation_frames'] = 0  
            
            if track['rotation_frames'] == 0:
                track['is_rotating'] = False
        
        track['last_rotation_check'] = self.frame_count
        
        return is_rotating, 0, rotation_confidence
    
    def check_keypoint_quality(self, keypoints, phase=None):
        """
        Check kualitas keypoints - RELAXED untuk squatting posture
        
        Returns:
            (is_valid, confidence, missing_keypoints)
        """
        # 🆕 STEP 1: PRE-CHECK for squatting
        is_squatting, squat_conf, squat_keypoints = self.detect_squatting_posture(keypoints)
        
        if is_squatting and squat_conf >= 0.60:
            # SQUATTING MODE: Only require lower body keypoints
            if self.debug_mode:
                print(f"  🧎 SQUATTING DETECTED (conf: {squat_conf:.2f})")
                print(f"     Available keypoints: {squat_keypoints}")
            
            # Check if we have MINIMUM required keypoints for squatting detection
            required_lower = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
            available_lower = [kp for kp in required_lower if kp in squat_keypoints]
            
            if len(available_lower) >= 3:  # Need at least 3/4 lower body keypoints
                visibility = len(available_lower) / len(required_lower)
                
                if self.debug_mode:
                    print(f"  ✅ SQUATTING: Keypoint quality OK (lower body: {visibility:.1%})")
                
                return True, visibility, [] 
            else:
                if self.debug_mode:
                    missing = [kp for kp in required_lower if kp not in squat_keypoints]
                    print(f"  ❌ SQUATTING: Insufficient lower body keypoints")
                    print(f"     Missing: {missing}")
                
                return False, 0.0, available_lower
        
        # NORMAL MODE: Original full keypoint check
        total_keypoints = len(self.KEYPOINTS)
        visible_count = 0
        missing_critical = []
        
        # COUNT visible keypoints (confidence > threshold)
        for kp_name, kp_idx in self.KEYPOINTS.items():
            if kp_idx < len(keypoints):
                conf = keypoints[kp_idx][2]
                if conf > self.KEYPOINT_QUALITY_THRESHOLDS['min_confidence']:
                    visible_count += 1
        
        visibility_ratio = visible_count / total_keypoints
        
        # CHECK critical keypoints (shoulders & hips HARUS ada)
        for critical_kp in self.KEYPOINT_QUALITY_THRESHOLDS['critical_keypoints']:
            kp = self.get_keypoint(keypoints, critical_kp)
            if not kp:
                missing_critical.append(critical_kp)
        
        # DECISION LOGIC
        if missing_critical:
            return False, 0.0, missing_critical
        
        if visibility_ratio < self.KEYPOINT_QUALITY_THRESHOLDS['min_visibility_ratio']:
            return False, visibility_ratio, []
        
        return True, visibility_ratio, []

    def check_phase_keypoint_requirements(self, keypoints, phase, hand_side=None):
        """
        Check apakah keypoints mencukupi untuk fase tertentu
        
        Args:
            keypoints: raw keypoints data
            phase: DetectionPhase enum
            hand_side: 'left' or 'right' (for hand-specific checks)
        
        Returns:
            (is_sufficient, confidence)
        """
        required = []

        if phase in [DetectionPhase.REACHING_SHELF, DetectionPhase.GRABBING]:
            if not hand_side or hand_side not in ['left', 'right']:
                # Invalid hand_side untuk fase ini
                if self.debug_mode:
                    print(f"⚠️ Invalid hand_side='{hand_side}' for phase {phase.value}")
                return False, 0.0

        is_squatting, squat_conf, _ = self.detect_squatting_posture(keypoints)
        is_squat_mode = is_squatting and squat_conf >= 0.60
        
        if self.debug_mode and is_squat_mode:
            print(f"  🧎 SQUAT MODE in keypoint check (conf: {squat_conf:.2f})")
        
        if phase == DetectionPhase.REACHING_SHELF:
            if hand_side:
                if is_squat_mode:
                    # SQUAT MODE: Only lower body required (upper body sering occluded)
                    required = [
                        f'{hand_side}_hip',
                        f'{hand_side}_knee',
                        'left_hip', 'right_hip'  # For stability check
                    ]
                    # Bonus: add shoulder if visible (not mandatory)
                    if self.get_keypoint(keypoints, f'{hand_side}_shoulder'):
                        required.append(f'{hand_side}_shoulder')
                else:
                    # NORMAL MODE: Full upper body required
                    required = [
                        f'{hand_side}_shoulder',
                        f'{hand_side}_elbow', 
                        f'{hand_side}_wrist',
                        f'{hand_side}_hip'
                    ]
        
        elif phase == DetectionPhase.GRABBING:
            if hand_side:
                if is_squat_mode:
                    # 🆕 SQUAT MODE: Minimal lower body + optional upper body
                    required = [
                        f'{hand_side}_hip',
                        f'{hand_side}_knee',
                        'left_hip', 'right_hip'
                    ]
                    # Add upper body keypoints if available (bonus points)
                    for kp_name in [f'{hand_side}_shoulder', f'{hand_side}_elbow', f'{hand_side}_wrist']:
                        if self.get_keypoint(keypoints, kp_name):
                            required.append(kp_name)
                else:
                    # NORMAL MODE: Full keypoints required
                    required = [
                        f'{hand_side}_shoulder',
                        f'{hand_side}_elbow',
                        f'{hand_side}_wrist',
                        f'{hand_side}_hip'
                    ]
        
        elif phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
            # Need more keypoints for pose detection
            required = [
                'nose',
                'left_shoulder', 'right_shoulder',
                'left_hip', 'right_hip',
                'left_knee', 'right_knee'
            ]

            if hand_side and hand_side in ['left', 'right']:
                        required.extend([
                            f'{hand_side}_shoulder',
                            f'{hand_side}_elbow',
                            f'{hand_side}_wrist',
                            f'{hand_side}_hip'
                        ])
                    
        # CHECK availability
        available_count = 0
        for kp_name in required:
            kp = self.get_keypoint(keypoints, kp_name)
            if kp:
                available_count += 1
        
        confidence = available_count / len(required) if required else 0

        if is_squat_mode:
            threshold = 0.60 
            is_sufficient = confidence >= threshold
            
            if self.debug_mode:
                print(f"  🧎 SQUAT threshold: {threshold:.0%} | Achieved: {confidence:.0%} | Pass: {is_sufficient}")
        else:
            threshold = 0.85  
            is_sufficient = confidence >= threshold
        
        return is_sufficient, confidence

    def initialize_pocket_zones(self, track_id, keypoints):
        """Inisialisasi zona kantong untuk person baru"""
        track = self.person_tracks[track_id]
        
        if track['pocket_zones']:
            return
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        left_ankle = self.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint(keypoints, 'right_ankle')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return
        
        shoulder_width = self.distance(left_shoulder, right_shoulder)
        if not shoulder_width:
            return
        
        track['pocket_zones'] = {
            'pants_pocket_left': PocketZone(
                'pants_pocket_left',
                left_hip, left_hip,
                width_factor=0.2,      
                depth_factor=0.35
            ),
            'pants_pocket_right': PocketZone(
                'pants_pocket_right',
                right_hip, right_hip,
                width_factor=0.2,      
                depth_factor=0.35
            ),
            'jacket_pocket_left': PocketZone(
                'jacket_pocket_left',
                left_shoulder, left_hip,
                width_factor=0.2,    
                depth_factor=0.3
            ),
            'jacket_pocket_right': PocketZone(
                'jacket_pocket_right',
                right_shoulder, right_hip,
                width_factor=0.2,     
                depth_factor=0.3
            ),
            'shoe_left': PocketZone(
                'shoe_left',
                left_ankle, left_ankle,
                width_factor=0.25,     
                depth_factor=0.20
            ) if left_ankle else None,
                'shoe_right': PocketZone(
                    'shoe_right',
                    right_ankle, right_ankle,
                    width_factor=0.25,
                    depth_factor=0.20
            ) if right_ankle else None
        }
        
        if self.debug_mode:
            print(f"  Initialize pocket zones for Track {track_id} (shoulder_width: {shoulder_width:.0f}px)")
    
    def update_pocket_zones(self, track_id, keypoints):
        """Update zona kantong berdasarkan posisi body terbaru"""
        track = self.person_tracks[track_id]
        
        if not track['pocket_zones']:
            return
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return
        
        shoulder_width = self.distance(left_shoulder, right_shoulder)
        if not shoulder_width:
            return
        
        track['pocket_zones']['pants_pocket_left'].left_point = left_hip
        track['pocket_zones']['pants_pocket_left'].right_point = left_hip
        track['pocket_zones']['pants_pocket_left'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['pants_pocket_right'].left_point = right_hip
        track['pocket_zones']['pants_pocket_right'].right_point = right_hip
        track['pocket_zones']['pants_pocket_right'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['jacket_pocket_left'].left_point = left_shoulder
        track['pocket_zones']['jacket_pocket_left'].right_point = left_hip
        track['pocket_zones']['jacket_pocket_left'].calculate_zone(shoulder_width)
        
        track['pocket_zones']['jacket_pocket_right'].left_point = right_shoulder
        track['pocket_zones']['jacket_pocket_right'].right_point = right_hip
        track['pocket_zones']['jacket_pocket_right'].calculate_zone(shoulder_width)
      
    def detect_zone_penetration(self, track_id, keypoints):
        """
        🆕 CRITICAL FIX: HANYA track zona untuk grabbed_hand side!
        """
        track = self.person_tracks[track_id]
        zone_poses = [] 
        
        zone_to_pose = {
            'pants_pocket_left': SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
            'pants_pocket_right': SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
            'jacket_pocket_left': SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
            'jacket_pocket_right': SuspiciousPose.ZONE_JACKET_POCKET_RIGHT
        }

        if not track['grab_detected']:
            return False, [], 0.0, []
        
        grabbed_hand = track['grabbed_hand']  

        if not grabbed_hand or grabbed_hand not in ['left', 'right']:
            if self.debug_mode:
                print(f"⚠️ Track {track_id}: Invalid grabbed_hand='{grabbed_hand}'")
            return False, [], 0.0, []
        
        # VALIDATION: Check grabbed_hand validity
        if not grabbed_hand or grabbed_hand not in ['left', 'right']:
            if self.debug_mode:
                print(f"⚠️ Track {track_id}: Invalid grabbed_hand='{grabbed_hand}'")
            return False, [], 0.0, []

        # GET WRIST KEYPOINT - Hanya untuk grabbed_hand
        wrist_key = f'{grabbed_hand}_wrist'
        wrist = self.get_keypoint(keypoints, wrist_key)
        
        if not wrist:
            return False, [], 0.0, []
        
        # CRITICAL FIX: FILTER ZONA berdasarkan grabbed_hand
        relevant_zones = {}
        for zone_name, zone in track['pocket_zones'].items():
            if grabbed_hand in zone_name:  
                relevant_zones[zone_name] = zone
        
        if not relevant_zones:
            if self.debug_mode:
                print(f"  ⚠️ No relevant zones for grabbed_hand={grabbed_hand}")
            return False, [], 0.0, []
        
        # Natural position check 
        shoulder_key = f'{grabbed_hand}_shoulder'
        elbow_key = f'{grabbed_hand}_elbow'
        hip_key = f'{grabbed_hand}_hip'
        
        shoulder = self.get_keypoint(keypoints, shoulder_key)
        elbow = self.get_keypoint(keypoints, elbow_key)
        hip = self.get_keypoint(keypoints, hip_key)
        
        if all([shoulder, elbow, wrist, hip]):
            if wrist[1] > hip[1]:
                horizontal_dist = abs(wrist[0] - hip[0])
                
                if horizontal_dist < 70:
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    wrist_to_hip = self.distance(wrist, hip)
                    
                    is_straight_arm = elbow_angle and elbow_angle >= 160
                    is_relax_position = wrist_to_hip and wrist_to_hip < 80 and horizontal_dist < 50
                    
                    if is_straight_arm or is_relax_position:
                        for zone_name in track['wrist_in_zone_frames'].keys():
                            if track['wrist_in_zone_frames'][zone_name] > 0:
                                track['wrist_in_zone_frames'][zone_name] = max(0, track['wrist_in_zone_frames'][zone_name] - 2)
                        
                        if self.debug_mode:
                            print(f"  ✅ Natural position - SKIP zone detection")
                        
                        return False, [], 0.0, []
        
        if not track['pocket_zones']:
            self.initialize_pocket_zones(track_id, keypoints)
            return False, [], 0.0, []
        
        self.update_pocket_zones(track_id, keypoints)
        
        zone_details = []
        confidence = 0.0
        
        # LOOP HANYA pada relevant_zones
        for zone_name, zone in relevant_zones.items():
            if zone is None: 
                continue
                
            if zone.is_point_in_zone(wrist):
                depth = zone.get_penetration_depth(wrist)
                
                track['wrist_in_zone_frames'][zone_name] += 1
                track['max_zone_depth'][zone_name] = max(
                    track['max_zone_depth'][zone_name],
                    depth
                )
                
                if track['wrist_in_zone_frames'][zone_name] >= self.ZONE_THRESHOLDS['min_frames_in_zone']:
                    depth_conf = min(1.0, depth / self.ZONE_THRESHOLDS['high_confidence_depth'])
                    frame_conf = min(1.0, track['wrist_in_zone_frames'][zone_name] / 10)
                    zone_conf = (depth_conf + frame_conf) / 2 * 0.95
                    
                    zone_details.append({
                        'zone': zone_name,
                        'hand': grabbed_hand,
                        'depth': depth,
                        'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                        'confidence': zone_conf,
                        'severity': 'high' if depth > self.ZONE_THRESHOLDS['high_confidence_depth'] else 'medium'
                    })

                    pose_type = zone_to_pose[zone_name]
                    zone_poses.append((
                        pose_type,
                        zone_conf,
                        f"Tangan {grabbed_hand.upper()} masuk {zone_name.replace('_', ' ')} (depth: {depth:.1%})"
                    ))
                    
                    confidence = max(confidence, zone_conf)
            else:
                # Zone exit logic
                if track['wrist_in_zone_frames'][zone_name] > 0:
                    if track['max_zone_depth'][zone_name] > self.ZONE_THRESHOLDS['min_penetration_depth']:
                        track['zone_detections'][zone_name].append({
                            'frame': self.frame_count,
                            'max_depth': track['max_zone_depth'][zone_name],
                            'frames_in_zone': track['wrist_in_zone_frames'][zone_name],
                            'hand': grabbed_hand
                        })
                    
                    track['wrist_in_zone_frames'][zone_name] = 0
                    track['max_zone_depth'][zone_name] = 0
        
        has_penetration = len(zone_details) > 0
        
        if self.debug_mode and has_penetration:
            print(f"  🎯 ZONE PENETRATION ({grabbed_hand.upper()} hand):")
            for z in zone_details:
                print(f"     - {z['zone']}: depth={z['depth']:.1%}, frames={z['frames_in_zone']}")
        
        return has_penetration, zone_details, confidence, zone_poses

    def validate_lower_reach_context(self, keypoints, wrist, elbow, shoulder, hip, knee, hand_side):
        """
        Validasi khusus untuk lower reach - pastikan bukan gerakan natural
        Returns: (is_valid, confidence_modifier)
        """
        # Cek 1: Wrist harus JAUH dari posisi relax 
        horizontal_dist_from_hip = abs(wrist[0] - hip[0])
        
        if horizontal_dist_from_hip < 50:  
            return False, 0.0
        
        # Cek 2: Elbow harus menunjukkan "reaching" motion 
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if elbow_angle and elbow_angle < 90:
            return False, 0.0
        
        # Cek 3: Wrist harus lebih rendah dari knee 
        if wrist[1] <= knee[1]:
            return False, 0.0
        
        # Cek 4: Movement vector harus menunjukkan "reaching forward and down"
        wrist_to_shoulder_horizontal = abs(wrist[0] - shoulder[0])
        wrist_to_shoulder_vertical = abs(wrist[1] - shoulder[1])
        
        if wrist_to_shoulder_horizontal < wrist_to_shoulder_vertical * 0.3:
            return False, 0.0
        
        # Cek 5: Wrist tidak boleh terlalu dekat dengan knee 
        wrist_to_knee = self.distance(wrist, knee)
        if wrist_to_knee and wrist_to_knee < 80: 
            return False, 0.0
        
        horizontal_conf = min(1.0, horizontal_dist_from_hip / 100)
        angle_conf = min(1.0, (elbow_angle - 90) / 90) if elbow_angle else 0.5
        
        confidence_modifier = (horizontal_conf + angle_conf) / 2
        
        return True, confidence_modifier

    def validate_reaching_context(self, keypoints, wrist, shoulder, hip, hand_side):
        """
        Validasi apakah benar-benar reaching ke rak (bukan gerakan random)
        Returns: (is_valid, confidence_modifier)
        """
        # Cek 1: Wrist harus di atas hip 
        if wrist[1] >= hip[1]:
            return False, 0.0
        
        # Cek 2: Shoulder seharusnya stabil 
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return False, 0.0
        
        # Body harus cukup vertikal 
        shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
        torso_height = abs(hip_midpoint_y - shoulder_midpoint_y)
        
        if torso_height < 100:  
            return False, 0.0
        
        # Cek 3: Wrist seharusnya extended secara horizontal 
        horizontal_extension = abs(wrist[0] - shoulder[0])
        vertical_movement = abs(shoulder[1] - wrist[1])
        
        if horizontal_extension < 40:  
            return False, 0.0
        
        if vertical_movement > horizontal_extension * 1.5:  
            return False, 0.0
        
        confidence_modifier = min(1.0, horizontal_extension / 150)
        
        return True, confidence_modifier
        
    def detect_hand_reaching(self, keypoints, track_id):
        """FASE 1: Deteksi tangan meraih - SUPPORT HIGH-ANGLE CCTV"""
        track = self.person_tracks[track_id]
        
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        left_knee = self.get_keypoint(keypoints, 'left_knee')
        right_knee = self.get_keypoint(keypoints, 'right_knee')
        
        # Save wrist positions
        if left_wrist:
            track['wrist_positions'].append(('left', left_wrist, self.frame_count))
        if right_wrist:
            track['wrist_positions'].append(('right', right_wrist, self.frame_count))
        
        reaching_detected = False
        hand_side = None
        confidence = 0.0
        reach_type = None

        # ═══════════════════════════════════════════════════════════
        # STEP 0: DETECT SQUATTING POSTURE
        # ═══════════════════════════════════════════════════════════
        is_squatting, squat_conf, _ = self.detect_squatting_posture(keypoints)
        squat_mode = is_squatting and squat_conf >= 0.60

        # ═══════════════════════════════════════════════════════════
        # SQUAT MODE: SPECIAL REACHING DETECTION dengan FALLBACK
        # ═══════════════════════════════════════════════════════════
        if squat_mode:
            if self.debug_mode:
                print(f"    🧎 SQUAT MODE ACTIVATED (conf: {squat_conf:.2f})")
            
            # ────────────────────────────────────────────────────────
            # Check LEFT hand - SQUAT LOGIC dengan FALLBACK
            # ────────────────────────────────────────────────────────
            # Get keypoints (bisa None)
            left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
            left_elbow = self.get_keypoint(keypoints, 'left_elbow')
            left_wrist = self.get_keypoint(keypoints, 'left_wrist')
            left_knee = self.get_keypoint(keypoints, 'left_knee')
            left_hip = self.get_keypoint(keypoints, 'left_hip')
            
            # COUNT available keypoints
            left_available = sum([
                left_shoulder is not None,
                left_elbow is not None,
                left_wrist is not None,
                left_knee is not None,
                left_hip is not None
            ])
            
            # FALLBACK
            if left_available >= 3:
                # CASE 1: FULL detection 
                if all([left_shoulder, left_elbow, left_wrist, left_knee]):
                    wrist_to_shoulder_dist = self.distance(left_wrist, left_shoulder)
                    is_wrist_extended = wrist_to_shoulder_dist and wrist_to_shoulder_dist > 45
                    is_reaching_down = left_wrist[1] > left_shoulder[1]
                    
                    elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                    is_elbow_bent = elbow_angle and 50 < elbow_angle < 160
                    
                    wrist_to_knee_horizontal = abs(left_wrist[0] - left_knee[0])
                    is_reaching_forward = wrist_to_knee_horizontal > 25
                    
                    if is_wrist_extended and is_reaching_down and is_elbow_bent and is_reaching_forward:
                        reaching_detected = True
                        hand_side = 'left'
                        reach_type = 'squat'
                        
                        dist_conf = min(1.0, wrist_to_shoulder_dist / 90)
                        angle_conf = 0.8 if is_elbow_bent else 0.5
                        forward_conf = min(1.0, wrist_to_knee_horizontal / 60)
                        
                        confidence = (dist_conf + angle_conf + forward_conf) / 3 * 0.85
                        
                        if squat_conf >= 0.75:
                            confidence = min(0.92, confidence * 1.15)
                
                # CASE 2: PARTIAL detection 
                elif left_hip and left_knee:
                    # Gunakan HIP sebagai proxy untuk hand position
                    hip_to_knee_horizontal = abs(left_hip[0] - left_knee[0])

                    if hip_to_knee_horizontal > self.GRAB_THRESHOLDS.get('squat_partial_hip_horizontal_min', 35) and \
                        squat_conf >= self.GRAB_THRESHOLDS.get('squat_partial_confidence_min', 0.75):
                  
                        reaching_detected = True
                        hand_side = 'left'
                        reach_type = 'squat_partial'
                        
                        # Confidence lebih rendah karena tidak ada wrist
                        confidence = squat_conf * 0.65  # Max 0.65 untuk partial
                        
                        if self.debug_mode:
                            print(f"    ⚠️ SQUAT PARTIAL LEFT (no wrist):")
                            print(f"       - hip_to_knee_horiz: {hip_to_knee_horizontal:.0f}px")
                            print(f"       - squat_conf: {squat_conf:.2f}")
                            print(f"       - confidence: {confidence:.2f}")
            
            # ────────────────────────────────────────────────────────
            # Check RIGHT hand - SAME LOGIC
            # ────────────────────────────────────────────────────────
            if not reaching_detected:
                right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
                right_elbow = self.get_keypoint(keypoints, 'right_elbow')
                right_wrist = self.get_keypoint(keypoints, 'right_wrist')
                right_knee = self.get_keypoint(keypoints, 'right_knee')
                right_hip = self.get_keypoint(keypoints, 'right_hip')
                
                right_available = sum([
                    right_shoulder is not None,
                    right_elbow is not None,
                    right_wrist is not None,
                    right_knee is not None,
                    right_hip is not None
                ])
                
                if right_available >= 3:
                    # FULL detection
                    if all([right_shoulder, right_elbow, right_wrist, right_knee]):
                        wrist_to_shoulder_dist = self.distance(right_wrist, right_shoulder)
                        is_wrist_extended = wrist_to_shoulder_dist and wrist_to_shoulder_dist > 45
                        is_reaching_down = right_wrist[1] > right_shoulder[1]
                        
                        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                        is_elbow_bent = elbow_angle and 50 < elbow_angle < 160
                        
                        wrist_to_knee_horizontal = abs(right_wrist[0] - right_knee[0])
                        is_reaching_forward = wrist_to_knee_horizontal > 25
                        
                        if is_wrist_extended and is_reaching_down and is_elbow_bent and is_reaching_forward:
                            reaching_detected = True
                            hand_side = 'right'
                            reach_type = 'squat'
                            
                            dist_conf = min(1.0, wrist_to_shoulder_dist / 90)
                            angle_conf = 0.8 if is_elbow_bent else 0.5
                            forward_conf = min(1.0, wrist_to_knee_horizontal / 60)
                            
                            confidence = (dist_conf + angle_conf + forward_conf) / 3 * 0.85
                            
                            if squat_conf >= 0.75:
                                confidence = min(0.92, confidence * 1.15)
                    
                    # PARTIAL detection
                    elif right_hip and right_knee:
                        hip_to_knee_horizontal = abs(right_hip[0] - right_knee[0])
                        
                        if hip_to_knee_horizontal > 40 and squat_conf >= 0.80:
                            reaching_detected = True
                            hand_side = 'right'
                            reach_type = 'squat_partial'
                            confidence = squat_conf * 0.65
                            
                            if self.debug_mode:
                                print(f"    ⚠️ SQUAT PARTIAL RIGHT (no wrist):")
                                print(f"       - hip_to_knee_horiz: {hip_to_knee_horizontal:.0f}px")
                                print(f"       - squat_conf: {squat_conf:.2f}")
                                print(f"       - confidence: {confidence:.2f}")
            
            # ────────────────────────────────────────────────────────
            # If detected in SQUAT mode, RETURN immediately
            # ────────────────────────────────────────────────────────
            if reaching_detected:
                if self.debug_mode:
                    print(f"  🎯 SQUAT REACHING: {hand_side} hand, type={reach_type}, conf: {confidence:.2f}")
                return reaching_detected, hand_side, confidence, reach_type
        
        # ═══════════════════════════════════════════════════════════
        # HIGH-ANGLE MODE: Simplified reaching detection
        # ═══════════════════════════════════════════════════════════
        elif self.high_angle_mode:
            # Check LEFT hand
            if all([left_shoulder, left_elbow, left_wrist, left_hip]):
                elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                wrist_to_shoulder = self.distance(left_wrist, left_shoulder)
                wrist_to_hip = self.distance(left_wrist, left_hip)
                
                is_arm_extended = elbow_angle and elbow_angle > 80
                is_away_from_body = (wrist_to_shoulder and wrist_to_shoulder > 50) or \
                                (wrist_to_hip and wrist_to_hip > 40)
                
                if is_arm_extended and is_away_from_body:
                    # Check movement 
                    has_movement = True
                    if len(track['wrist_positions']) >= 3:
                        past_left = [p for p in track['wrist_positions'] 
                                if p[0] == 'left' and self.frame_count - p[2] <= 5]
                        if past_left:
                            past_wrist = past_left[0][1]
                            movement = self.distance(left_wrist, past_wrist)
                            has_movement = movement and movement > 12  
                    
                    if has_movement:
                        reaching_detected = True
                        hand_side = 'left'
                        reach_type = 'high_angle'
                        
                        # Confidence based on elbow angle + distance
                        angle_conf = min(1.0, (elbow_angle - 80) / 100) if elbow_angle else 0.5
                        dist_conf = min(1.0, wrist_to_shoulder / 100) if wrist_to_shoulder else 0.5
                        confidence = (angle_conf + dist_conf) / 2 * 0.85
                        
                        if self.debug_mode:
                            print(f"    ✅ HIGH-ANGLE reach LEFT: angle={elbow_angle:.0f}°, "
                                f"dist={wrist_to_shoulder:.0f}px → conf={confidence:.2f}")
            
            # Check RIGHT hand (same logic)
            if not reaching_detected and all([right_shoulder, right_elbow, right_wrist, right_hip]):
                elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                wrist_to_shoulder = self.distance(right_wrist, right_shoulder)
                wrist_to_hip = self.distance(right_wrist, right_hip)
                
                is_arm_extended = elbow_angle and elbow_angle > 80
                is_away_from_body = (wrist_to_shoulder and wrist_to_shoulder > 50) or \
                                (wrist_to_hip and wrist_to_hip > 40)
                
                if is_arm_extended and is_away_from_body:
                    has_movement = True
                    if len(track['wrist_positions']) >= 3:
                        past_right = [p for p in track['wrist_positions'] 
                                    if p[0] == 'right' and self.frame_count - p[2] <= 5]
                        if past_right:
                            past_wrist = past_right[0][1]
                            movement = self.distance(right_wrist, past_wrist)
                            has_movement = movement and movement > 15
                    
                    if has_movement:
                        reaching_detected = True
                        hand_side = 'right'
                        reach_type = 'high_angle'
                        
                        angle_conf = min(1.0, (elbow_angle - 80) / 100) if elbow_angle else 0.5
                        dist_conf = min(1.0, wrist_to_shoulder / 100) if wrist_to_shoulder else 0.5
                        confidence = (angle_conf + dist_conf) / 2 * 0.85
                        
                        if self.debug_mode:
                            print(f"    ✅ HIGH-ANGLE reach RIGHT: angle={elbow_angle:.0f}°, "
                                f"dist={wrist_to_shoulder:.0f}px → conf={confidence:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # NORMAL MODE: Original complex logic 
        # ═══════════════════════════════════════════════════════════
        else:
            if all([left_shoulder, left_elbow, left_wrist, left_hip]):
                wrist_to_shoulder_dist = self.distance(left_wrist, left_shoulder)
                elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                wrist_to_hip_dist = self.distance(left_wrist, left_hip)
                
                # REACH TYPE 1: RAK ATAS/TENGAH
                if wrist_to_shoulder_dist and wrist_to_shoulder_dist > self.GRAB_THRESHOLDS['hand_extension_threshold']:
                    if elbow_angle and elbow_angle > self.GRAB_THRESHOLDS['elbow_angle_extended']:
                        height_diff = left_wrist[1] - left_shoulder[1]

                        is_valid_reach, conf_modifier = self.validate_reaching_context(
                            keypoints, left_wrist, left_shoulder, left_hip, 'left'
                        )
                        
                        if is_valid_reach:
                            if height_diff < self.GRAB_THRESHOLDS['hand_height_tolerance']:
                                if wrist_to_hip_dist and wrist_to_hip_dist > 60: 
                                    reaching_detected = True
                                    hand_side = 'left'
                                    reach_type = 'middle' if height_diff > -50 else 'upper'
                                    
                                    dist_conf = min(1.0, wrist_to_shoulder_dist / 200)  
                                    angle_conf = min(1.0, (elbow_angle - 120) / 60)      
                                    height_conf = 1.0 - (abs(height_diff) / self.GRAB_THRESHOLDS['hand_height_tolerance'])
                                    
                                    base_conf = (dist_conf + angle_conf + height_conf) / 3
                                    confidence = base_conf * 0.90 * conf_modifier  
                                    
                                    if self.debug_mode:
                                        print(f"    ✅ Upper/Middle reach LEFT: dist={wrist_to_shoulder_dist:.0f}px, "
                                            f"angle={elbow_angle:.0f}°, height_diff={height_diff:.0f}px → conf={confidence:.2f}")
                
                # REACH TYPE 2: RAK BAWAH 
                if not reaching_detected and all([left_wrist, left_knee, left_hip]):
                    wrist_y = left_wrist[1]
                    hip_y = left_hip[1]
                    knee_y = left_knee[1]
                    
                    # Tangan di bawah pinggul dengan posisi yang jelas
                    if wrist_y > hip_y + 50:  
                        hip_to_knee_dist = abs(knee_y - hip_y)
                        wrist_below_hip = wrist_y - hip_y
                        
                        if hip_to_knee_dist > 0:
                            position_ratio = wrist_below_hip / hip_to_knee_dist
                            
                            if 0.15 <= position_ratio <= 1.8:  
                                horizontal_dist = abs(left_wrist[0] - left_hip[0])
                                
                                if horizontal_dist > 25: 
                                    if elbow_angle and elbow_angle > 65:
                                        if wrist_to_hip_dist and wrist_to_hip_dist > 50:  
                                            reaching_detected = True
                                            hand_side = 'left'
                                            reach_type = 'lower'
                                            
                                            position_conf = min(1.0, position_ratio / 1.5)
                                            horizontal_conf = min(1.0, horizontal_dist / 120)
                                            dist_conf = min(1.0, wrist_to_hip_dist / 120)
                                            angle_bonus = min(0.20, (elbow_angle - 75) / 105 * 0.20)
                                            
                                            base_conf = (position_conf + horizontal_conf + dist_conf) / 3
                                            confidence = min(0.88, base_conf * 0.85 + angle_bonus)
                                            
                                            if self.debug_mode:
                                                print(f"    ✅ Lower reach LEFT: pos={position_ratio:.2f}, "
                                                    f"horiz={horizontal_dist:.0f}px, dist={wrist_to_hip_dist:.0f}px, "
                                                    f"angle={elbow_angle:.0f}° → conf={confidence:.2f}")
                    
            if all([right_shoulder, right_elbow, right_wrist, right_hip]) and not reaching_detected:
                wrist_to_shoulder_dist = self.distance(right_wrist, right_shoulder)
                elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                wrist_to_hip_dist = self.distance(right_wrist, right_hip)
                
                # REACH TYPE 1: RAK ATAS/TENGAH 
                if wrist_to_shoulder_dist and wrist_to_shoulder_dist > self.GRAB_THRESHOLDS['hand_extension_threshold']:
                    if elbow_angle and elbow_angle > self.GRAB_THRESHOLDS['elbow_angle_extended']:
                        height_diff = right_wrist[1] - right_shoulder[1]
                        

                        is_valid_reach, conf_modifier = self.validate_reaching_context(
                            keypoints, right_wrist, right_shoulder, right_hip, 'right'
                        )
                        
                        if is_valid_reach:
                            if height_diff < self.GRAB_THRESHOLDS['hand_height_tolerance']:
                                if wrist_to_hip_dist and wrist_to_hip_dist > 60:
                                    reaching_detected = True
                                    hand_side = 'right'
                                    reach_type = 'middle' if height_diff > -50 else 'upper'
                                    
                                    dist_conf = min(1.0, wrist_to_shoulder_dist / 200)
                                    angle_conf = min(1.0, (elbow_angle - 120) / 60)
                                    height_conf = 1.0 - (abs(height_diff) / self.GRAB_THRESHOLDS['hand_height_tolerance'])
                                    
                                    base_conf = (dist_conf + angle_conf + height_conf) / 3
                                    confidence = base_conf * 0.90 * conf_modifier
                                    
                                    if self.debug_mode:
                                        print(f"    ✅ Upper/Middle reach RIGHT: dist={wrist_to_shoulder_dist:.0f}px, "
                                            f"angle={elbow_angle:.0f}°, height_diff={height_diff:.0f}px → conf={confidence:.2f}")

                # REACH TYPE 2: RAK BAWAH - RIGHT HAND 
                if not reaching_detected and all([right_wrist, right_knee, right_hip]):
                    wrist_y = right_wrist[1]
                    hip_y = right_hip[1]
                    knee_y = right_knee[1]  
                    
                    # KONDISI 1: Tangan di bawah pinggul
                    if wrist_y > hip_y:
                        hip_to_knee_dist = abs(knee_y - hip_y)
                        wrist_below_hip = wrist_y - hip_y
                        
                        if hip_to_knee_dist > 0:
                            position_ratio = wrist_below_hip / hip_to_knee_dist
                            
                            if (self.GRAB_THRESHOLDS['reaching_down_position_ratio_min'] <= position_ratio <= 
                                self.GRAB_THRESHOLDS['reaching_down_position_ratio_max']):
                                
                                horizontal_dist = abs(right_wrist[0] - right_hip[0])
                                
                                if horizontal_dist > self.GRAB_THRESHOLDS['reaching_down_horizontal_min']:
                                    
                                    elbow_ok = True
                                    angle_bonus = 0.0
                                    
                                    if elbow_angle:
                                        if elbow_angle > self.GRAB_THRESHOLDS['reaching_down_angle_threshold']:
                                            angle_bonus = min(0.25, (elbow_angle - 85) / 95 * 0.25)
                                        elif elbow_angle < 70:
                                            elbow_ok = False
                                    
                                    if elbow_ok:
                                        if wrist_to_hip_dist and wrist_to_hip_dist > self.GRAB_THRESHOLDS['reaching_down_hip_distance']:
                                            reaching_detected = True
                                            hand_side = 'right'
                                            reach_type = 'lower'
                                            
                                            position_conf = min(1.0, position_ratio / 1.2)
                                            horizontal_conf = min(1.0, horizontal_dist / 100)
                                            dist_conf = min(1.0, wrist_to_hip_dist / 100)
                                            
                                            base_conf = (position_conf + horizontal_conf + dist_conf) / 3
                                            confidence = min(0.92, base_conf * 0.88 + angle_bonus)
                                            
                                            if self.debug_mode:
                                                print(f"    ✅ Lower reach RIGHT: pos={position_ratio:.2f}, "
                                                    f"horiz={horizontal_dist:.0f}px, dist={wrist_to_hip_dist:.0f}px, "
                                                    f"angle={elbow_angle:.0f}° → conf={confidence:.2f}")
    
            if reaching_detected and self.debug_mode and reach_type:
                print(f"  🎯 REACHING [{reach_type.upper()}]: {hand_side} hand, conf: {confidence:.2f}")
            elif self.debug_mode and not reaching_detected:
                if left_wrist and left_hip and left_knee:
                    horiz_l = abs(left_wrist[0] - left_hip[0])
                    y_diff_l = left_wrist[1] - left_hip[1]
                    
                    if y_diff_l > 0:  
                        hip_to_knee = abs(left_knee[1] - left_hip[1])
                        position_ratio = y_diff_l / hip_to_knee if hip_to_knee > 0 else 0
                        
                        wrist_to_hip_dist = self.distance(left_wrist, left_hip) if left_wrist and left_hip else 0
                        wrist_to_knee_dist = self.distance(left_wrist, left_knee) if left_wrist and left_knee else 0
                        
                        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
                        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
                        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist) if all([left_shoulder, left_elbow, left_wrist]) else 0
                        
                        print(f"  ❌ LEFT not reaching:")
                        print(f"     horiz={horiz_l:.0f}px (need >{self.GRAB_THRESHOLDS['reaching_down_horizontal_min']}px) {'✅' if horiz_l > self.GRAB_THRESHOLDS['reaching_down_horizontal_min'] else '❌'}")
                        print(f"     y_diff={y_diff_l:.0f}px, hip_knee={hip_to_knee:.0f}px")
                        print(f"     position_ratio={position_ratio:.2f} (need {self.GRAB_THRESHOLDS['reaching_down_position_ratio_min']:.2f}-{self.GRAB_THRESHOLDS['reaching_down_position_ratio_max']:.2f}) {'✅' if self.GRAB_THRESHOLDS['reaching_down_position_ratio_min'] <= position_ratio <= self.GRAB_THRESHOLDS['reaching_down_position_ratio_max'] else '❌'}")
                        print(f"     wrist_to_hip={wrist_to_hip_dist:.0f}px (need >{self.GRAB_THRESHOLDS['reaching_down_hip_distance']}px) {'✅' if wrist_to_hip_dist > self.GRAB_THRESHOLDS['reaching_down_hip_distance'] else '❌'}")
                        print(f"     wrist_to_knee={wrist_to_knee_dist:.0f}px (fallback <{self.GRAB_THRESHOLDS['reaching_down_knee_proximity']}px) {'✅ FALLBACK!' if wrist_to_knee_dist < self.GRAB_THRESHOLDS['reaching_down_knee_proximity'] else '❌'}")
                        print(f"     elbow_angle={elbow_angle:.0f}° (prefer >{self.GRAB_THRESHOLDS['reaching_down_angle_threshold']}°)")
                
                if right_wrist and right_hip and right_knee:
                    horiz_r = abs(right_wrist[0] - right_hip[0])
                    y_diff_r = right_wrist[1] - right_hip[1]
                    
                    if y_diff_r > 0:
                        hip_to_knee = abs(right_knee[1] - right_hip[1])
                        position_ratio = y_diff_r / hip_to_knee if hip_to_knee > 0 else 0
                        
                        wrist_to_hip_dist = self.distance(right_wrist, right_hip) if right_wrist and right_hip else 0
                        wrist_to_knee_dist = self.distance(right_wrist, right_knee) if right_wrist and right_knee else 0
                        
                        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
                        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
                        elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist) if all([right_shoulder, right_elbow, right_wrist]) else 0
                        
                        print(f"  ❌ RIGHT not reaching:")
                        print(f"     horiz={horiz_r:.0f}px (need >{self.GRAB_THRESHOLDS['reaching_down_horizontal_min']}px) {'✅' if horiz_r > self.GRAB_THRESHOLDS['reaching_down_horizontal_min'] else '❌'}")
                        print(f"     y_diff={y_diff_r:.0f}px, hip_knee={hip_to_knee:.0f}px")
                        print(f"     position_ratio={position_ratio:.2f} (need {self.GRAB_THRESHOLDS['reaching_down_position_ratio_min']:.2f}-{self.GRAB_THRESHOLDS['reaching_down_position_ratio_max']:.2f}) {'✅' if self.GRAB_THRESHOLDS['reaching_down_position_ratio_min'] <= position_ratio <= self.GRAB_THRESHOLDS['reaching_down_position_ratio_max'] else '❌'}")
                        print(f"     wrist_to_hip={wrist_to_hip_dist:.0f}px (need >{self.GRAB_THRESHOLDS['reaching_down_hip_distance']}px) {'✅' if wrist_to_hip_dist > self.GRAB_THRESHOLDS['reaching_down_hip_distance'] else '❌'}")
                        print(f"     wrist_to_knee={wrist_to_knee_dist:.0f}px (fallback <{self.GRAB_THRESHOLDS['reaching_down_knee_proximity']}px) {'✅ FALLBACK!' if wrist_to_knee_dist < self.GRAB_THRESHOLDS['reaching_down_knee_proximity'] else '❌'}")
                        print(f"     elbow_angle={elbow_angle:.0f}° (prefer >{self.GRAB_THRESHOLDS['reaching_down_angle_threshold']}°)")
            pass
        
        if reaching_detected and self.debug_mode and reach_type:
            print(f"  🎯 REACHING [{reach_type.upper()}]: {hand_side} hand, conf: {confidence:.2f}")
        
        return reaching_detected, hand_side, confidence, reach_type

    def detect_grabbing_motion(self, keypoints, track_id):
        """FASE 2: Deteksi gerakan menggenggam"""
        track = self.person_tracks[track_id]
        
        if not track['hand_extended']:
            return False, 0.0
        
        grabbed_hand = track['grabbed_hand']
        if not grabbed_hand:
            return False, 0.0
        
        shoulder_key = f'{grabbed_hand}_shoulder'
        wrist_key = f'{grabbed_hand}_wrist'
        elbow_key = f'{grabbed_hand}_elbow'
        hip_key = f'{grabbed_hand}_hip'
        
        shoulder = self.get_keypoint(keypoints, shoulder_key)
        wrist = self.get_keypoint(keypoints, wrist_key)
        elbow = self.get_keypoint(keypoints, elbow_key)
        hip = self.get_keypoint(keypoints, hip_key)
        
        if not all([shoulder, wrist, elbow, hip]):
            return False, 0.0
        
        current_dist_shoulder = self.distance(wrist, shoulder)
        current_dist_hip = self.distance(wrist, hip)
        
        grab_detected = False
        confidence = 0.0
        
        if len(track['wrist_positions']) >= 5:
            past_positions = [p for p in track['wrist_positions'] 
                            if p[0] == grabbed_hand and 
                            self.frame_count - p[2] >= 3 and 
                            self.frame_count - p[2] <= 6]
            
            if past_positions:
                past_wrist = past_positions[0][1]
                past_dist_shoulder = self.distance(past_wrist, shoulder)
                past_dist_hip = self.distance(past_wrist, hip)
                
                frame_diff = self.frame_count - past_positions[0][2]
                wrist_movement = self.distance(wrist, past_wrist)
                velocity = wrist_movement / max(frame_diff, 1) if wrist_movement else 0
                
                conditions_met = 0
                total_confidence = 0.0
                
                if past_dist_shoulder and current_dist_shoulder:
                    distance_reduction_shoulder = past_dist_shoulder - current_dist_shoulder
                    if distance_reduction_shoulder > self.GRAB_THRESHOLDS['distance_reduction_threshold']:
                        conditions_met += 1
                        total_confidence += min(0.35, distance_reduction_shoulder / 100)
                
                if past_dist_hip and current_dist_hip:
                    distance_reduction_hip = past_dist_hip - current_dist_hip
                    if distance_reduction_hip > self.GRAB_THRESHOLDS['distance_reduction_threshold']:
                        conditions_met += 1
                        total_confidence += min(0.35, distance_reduction_hip / 100)
                
                if velocity > self.GRAB_THRESHOLDS['velocity_threshold']:
                    conditions_met += 1
                    total_confidence += min(0.30, velocity / 20)
                
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                if elbow_angle and elbow_angle < self.GRAB_THRESHOLDS['elbow_angle_grab']:
                    conditions_met += 1
                    total_confidence += 0.30
                
                y_movement = wrist[1] - past_wrist[1]
                if y_movement > 20:
                    conditions_met += 1
                    total_confidence += 0.25
                
                if conditions_met >= 2:
                    grab_detected = True
                    confidence = min(0.95, total_confidence)
                    
                    if self.debug_mode:
                        print(f"  GRAB: {conditions_met}/4 conditions | conf: {confidence:.2f}")
        
        return grab_detected, confidence

    def detect_suspicious_poses(self, keypoints, grabbed_hand=None, is_rotating=False, is_walking=False):
        """
        FASE 3: Deteksi pose mencurigakan
        
        🆕 CRITICAL: HANYA deteksi untuk tangan yang melakukan GRAB!
        Tangan lain diabaikan untuk menghindari false positive.
        
        Args:
            grabbed_hand: 'left' or 'right' - tangan yang melakukan grab
            is_rotating: apakah body sedang rotating
            is_walking: apakah sedang berjalan dengan barang
        """
        suspicious_poses = []
        is_natural_position = False
        
        # ═══════════════════════════════════════════════════════════
        # 🆕 STEP 0: VALIDATE grabbed_hand FIRST
        # ═══════════════════════════════════════════════════════════
        if not grabbed_hand or grabbed_hand not in ['left', 'right']:
            if self.debug_mode:
                print(f"⚠️ Invalid grabbed_hand='{grabbed_hand}' in detect_suspicious_poses")
                print(f"   → SKIPPING ALL DETECTIONS (no grabbed_hand)")
            return []  # ❌ STOP - tidak ada deteksi sama sekali!
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: SQUATTING DETECTION 
        # ═══════════════════════════════════════════════════════════
        is_squatting, squat_conf, _ = self.detect_squatting_posture(keypoints)
        
        if is_squatting and squat_conf >= 0.60:
            left_hip = self.get_keypoint(keypoints, 'left_hip')
            right_hip = self.get_keypoint(keypoints, 'right_hip')
            left_knee = self.get_keypoint(keypoints, 'left_knee')
            right_knee = self.get_keypoint(keypoints, 'right_knee')
            left_ankle = self.get_keypoint(keypoints, 'left_ankle')
            right_ankle = self.get_keypoint(keypoints, 'right_ankle')
            
            if all([left_hip, right_hip, left_knee, right_knee]):
                left_angle = self.calculate_angle(left_hip, left_knee, left_ankle) if left_ankle else None
                right_angle = self.calculate_angle(right_hip, right_knee, right_ankle) if right_ankle else None
                
                angles = [a for a in [left_angle, right_angle] if a is not None]
                avg_angle = sum(angles) / len(angles) if angles else 90
                
                if avg_angle < 90:
                    pose_type = SuspiciousPose.SQUATTING_LOW
                    confidence = squat_conf * 0.95
                    description = f"🚨 Jongkok SANGAT RENDAH setelah ambil barang (sudut: {avg_angle:.0f}°)"
                elif avg_angle < 115:
                    pose_type = SuspiciousPose.SQUATTING_LOW
                    confidence = squat_conf * 0.88
                    description = f"⚠️ Jongkok rendah dengan barang (sudut: {avg_angle:.0f}°)"
                else:
                    pose_type = SuspiciousPose.CROUCHING
                    confidence = squat_conf * 0.80
                    description = f"Berjongkok dengan barang (sudut: {avg_angle:.0f}°)"
                
                suspicious_poses.append((pose_type, confidence, description))
                
                if self.debug_mode:
                    print(f"  🧎 SQUATTING POSE DETECTED:")
                    print(f"     - Avg angle: {avg_angle:.0f}°")
                    print(f"     - Confidence: {confidence:.2f}")
                    print(f"     - Type: {pose_type.value}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: GET KEYPOINTS 
        # ═══════════════════════════════════════════════════════════
        shoulder_key = f'{grabbed_hand}_shoulder'
        elbow_key = f'{grabbed_hand}_elbow'
        wrist_key = f'{grabbed_hand}_wrist'
        hip_key = f'{grabbed_hand}_hip'
        knee_key = f'{grabbed_hand}_knee'
        
        shoulder = self.get_keypoint(keypoints, shoulder_key)
        elbow = self.get_keypoint(keypoints, elbow_key)
        wrist = self.get_keypoint(keypoints, wrist_key)
        hip = self.get_keypoint(keypoints, hip_key)
        knee = self.get_keypoint(keypoints, knee_key)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: CHECK NATURAL POSITION 
        # ═══════════════════════════════════════════════════════════
        if all([shoulder, elbow, wrist, hip]):
            if wrist[1] > hip[1]:
                horizontal_dist = abs(wrist[0] - hip[0])
                
                if horizontal_dist < 70:
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    wrist_to_hip = self.distance(wrist, hip)
                    
                    is_straight_arm = elbow_angle and elbow_angle >= 160
                    is_relax_position = wrist_to_hip and wrist_to_hip < 80 and horizontal_dist < 50
                    
                    if is_straight_arm or is_relax_position:
                        is_natural_position = True
                        
                        if self.debug_mode:
                            print(f"  ✅ Hand {grabbed_hand} in NATURAL position - NO suspicious detection")
                            print(f"     - Elbow angle: {elbow_angle:.0f}°" if elbow_angle else "")
                            print(f"     - Wrist to hip: {wrist_to_hip:.0f}px" if wrist_to_hip else "")
                            print(f"     - Horizontal distance: {horizontal_dist:.0f}px")
                        
                        return suspicious_poses  
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: BODY POSE DETECTIONS 
        # ═══════════════════════════════════════════════════════════
        
        # 4.1: BENDING DOWN
        nose = self.get_keypoint(keypoints, 'nose')
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        
        if all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = abs(hip_y - shoulder_y)
            nose_to_hip = abs(nose[1] - hip_y)
            bend_ratio = nose_to_hip / (torso_height + 1e-6)
            
            if bend_ratio < 0.70:
                confidence = (1.0 - bend_ratio / 0.70) * 0.90
                suspicious_poses.append((
                    SuspiciousPose.BENDING_DOWN,
                    confidence,
                    f"Membungkuk setelah ambil barang"
                ))
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: HAND-SPECIFIC DETECTIONS 
        # ═══════════════════════════════════════════════════════════
        
        rotation_penalty = 0.6 if is_rotating else 1.0
        
        # 5.1: HIDING UNDER CLOTHING - STRICT FILTER
        if all([wrist, shoulder, elbow, left_shoulder, right_shoulder, left_hip, right_hip]):
            chest_y = (left_shoulder[1] + right_shoulder[1]) / 2
            belly_y = (left_hip[1] + right_hip[1]) / 2
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            torso_width = abs(left_shoulder[0] - right_shoulder[0])
            
            threshold_multiplier = 0.30 if is_rotating else 0.50
            
            wrist_at_torso = chest_y < wrist[1] < belly_y
            wrist_dist_from_center = abs(wrist[0] - torso_center_x)
            
            is_too_centered = wrist_dist_from_center < 30
            
            if wrist_at_torso and not is_too_centered and wrist_dist_from_center < torso_width * threshold_multiplier:
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                if not (elbow_angle and elbow_angle > 165):
                    confidence = 0.95 * rotation_penalty
                    if not is_rotating or confidence > 0.70:
                        suspicious_poses.append((
                            SuspiciousPose.HIDING_UNDER_CLOTHING,
                            confidence,
                            f"🚨 Memasukkan barang ke baju ({grabbed_hand.upper()} hand)"
                        ))
                        
                        if self.debug_mode:
                            print(f"  👕 HIDING_UNDER_CLOTHING detected:")
                            print(f"     - Hand: {grabbed_hand.upper()}")
                            print(f"     - Wrist at torso: Y={wrist[1]:.0f}")
                            print(f"     - Distance from center: {wrist_dist_from_center:.0f}px")
        
        # 5.2: PUTTING IN PANTS POCKET 
        if all([wrist, shoulder, elbow, hip, knee]):
            def is_hand_straight_down(shoulder, elbow, wrist, hip):
                if not all([shoulder, elbow, wrist, hip]):
                    return False
                
                if wrist[1] <= hip[1]:
                    return False
                
                horizontal_dist = abs(wrist[0] - shoulder[0])
                shoulder_to_hip_width = abs(shoulder[0] - hip[0])
                
                if horizontal_dist > shoulder_to_hip_width * 1.3:
                    return False
                
                arm_angle = self.calculate_angle(shoulder, elbow, wrist)
                if arm_angle and arm_angle >= 160:
                    return True
                
                wrist_to_hip = self.distance(wrist, hip)
                if wrist_to_hip and wrist_to_hip < 90:
                    horizontal_offset = abs(wrist[0] - hip[0])
                    if horizontal_offset < 50:
                        return True
                
                return False
            
            is_straight = is_hand_straight_down(shoulder, elbow, wrist, hip)
            
            if not is_straight:
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                if wrist[1] > hip[1] and wrist[1] < knee[1]:
                    wrist_to_hip_x = abs(wrist[0] - hip[0])
                    wrist_to_hip_y = abs(wrist[1] - hip[1])
                    
                    if elbow_angle and elbow_angle < 120:
                        horizontal_movement = abs(wrist[0] - hip[0])
                        
                        if wrist_to_hip_x < 70 and wrist_to_hip_y < 110 and horizontal_movement > 20:
                            confidence = 0.88
                            suspicious_poses.append((
                                SuspiciousPose.PUTTING_IN_PANTS_POCKET,
                                confidence,
                                f"🚨 Memasukkan ke kantong celana ({grabbed_hand.upper()})"
                            ))
                            
                            if self.debug_mode:
                                print(f"  👖 PANTS_POCKET detected:")
                                print(f"     - Hand: {grabbed_hand.upper()}")
                                print(f"     - Wrist to hip: X={wrist_to_hip_x:.0f}px, Y={wrist_to_hip_y:.0f}px")
        
        # 5.3: HEAD POSES 
        if all([wrist, shoulder, elbow]):
            if nose:
                head_y = nose[1]
                head_x = nose[0]
            elif left_shoulder and right_shoulder:
                left_eye = self.get_keypoint(keypoints, 'left_eye')
                right_eye = self.get_keypoint(keypoints, 'right_eye')
                if left_eye and right_eye:
                    head_y = (left_eye[1] + right_eye[1]) / 2
                    head_x = (left_eye[0] + right_eye[0]) / 2
                else:
                    head_y = None
                    head_x = None
            else:
                head_y = None
                head_x = None
            
            if head_y is not None and head_x is not None:
                vertical_distance = wrist[1] - head_y
                horizontal_dist_to_head = abs(wrist[0] - head_x)
                
                if vertical_distance > 30:
                    if self.debug_mode:
                        print(f"  ✅ HEAD CHECK SKIPPED: Wrist {vertical_distance:.0f}px BELOW head (safe)")
                else:
                    wrist_to_shoulder_dist = self.distance(wrist, shoulder)
                    is_extended = wrist_to_shoulder_dist and wrist_to_shoulder_dist > 50
                    
                    wrist_to_shoulder_horizontal = abs(wrist[0] - shoulder[0])
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist) if elbow else None
                    
                    is_reaching_motion = False
                    
                    if wrist_to_shoulder_horizontal > 80:
                        if elbow_angle and elbow_angle > 100:
                            if horizontal_dist_to_head > 80:
                                is_reaching_motion = True
                                
                                if self.debug_mode:
                                    print(f"  ✅ REACHING MOTION DETECTED:")
                                    print(f"     - Wrist horizontal ext: {wrist_to_shoulder_horizontal:.0f}px")
                                    print(f"     - Elbow angle: {elbow_angle:.0f}° (open)")
                                    print(f"     - Distance from head: {horizontal_dist_to_head:.0f}px")
                                    print(f"     → NOT HEAD HIDING!")
                    
                    if not is_reaching_motion:
                        if horizontal_dist_to_head < 60:
                            elbow_is_bent = elbow_angle and 30 < elbow_angle < 120
                            
                            if elbow_is_bent and is_extended:
                                is_moving_toward_head = False
                                
                                track = self.person_tracks.get(grabbed_hand, {})
                                if len(track.get('wrist_positions', [])) >= 3:
                                    prev_positions = [p for p in track['wrist_positions'] 
                                                    if p[0] == grabbed_hand and 
                                                    self.frame_count - p[2] <= 5]
                                    
                                    if prev_positions:
                                        prev_wrist = prev_positions[-1][1]
                                        prev_dist_to_head = math.sqrt(
                                            (prev_wrist[0] - head_x)**2 + 
                                            (prev_wrist[1] - head_y)**2
                                        )
                                        current_dist_to_head = math.sqrt(
                                            (wrist[0] - head_x)**2 + 
                                            (wrist[1] - head_y)**2
                                        )
                                        
                                        if current_dist_to_head < prev_dist_to_head - 10:
                                            is_moving_toward_head = True
                                            
                                            if self.debug_mode:
                                                print(f"  ⚠️ MOVING TOWARD HEAD:")
                                                print(f"     - Prev dist: {prev_dist_to_head:.0f}px")
                                                print(f"     - Current dist: {current_dist_to_head:.0f}px")
                                
                                if is_moving_toward_head or horizontal_dist_to_head < 40:
                                    vertical_conf = max(0.4, 1.0 - (abs(vertical_distance) / 30))
                                    horizontal_conf = max(0.5, 1.0 - (horizontal_dist_to_head / 60))
                                    elbow_conf = 0.9 if elbow_is_bent else 0.6
                                    
                                    movement_bonus = 0.2 if is_moving_toward_head else 0.0
                                    
                                    confidence = ((vertical_conf + horizontal_conf + elbow_conf) / 3 + movement_bonus) * 0.95
                                    confidence = min(0.95, confidence)
                                    
                                    if vertical_distance < -40:
                                        pose_type = SuspiciousPose.HIDING_IN_HAT
                                        description = f"🚨 Memasukkan barang ke TOPI ({grabbed_hand.upper()})"
                                    elif vertical_distance < 0:
                                        pose_type = SuspiciousPose.HAND_ON_HEAD
                                        description = f"⚠️ Tangan DI ATAS KEPALA ({grabbed_hand.upper()})"
                                    else:
                                        pose_type = SuspiciousPose.HAND_ON_HEAD
                                        description = f"⚠️ Tangan di AREA KEPALA ({grabbed_hand.upper()})"
                                    
                                    suspicious_poses.append((pose_type, confidence, description))
                                    
                                    if self.debug_mode:
                                        print(f"    🎩 HEAD POSE DETECTED:")
                                        print(f"       - Hand: {grabbed_hand.upper()}")
                                        print(f"       - Type: {pose_type.value}")
                                        print(f"       - Wrist Y: {wrist[1]:.0f}, Head Y: {head_y:.0f}")
                                        print(f"       - Vertical offset: {vertical_distance:.0f}px")
                                        print(f"       - Horizontal dist: {horizontal_dist_to_head:.0f}px")
                                        print(f"       - Moving toward: {is_moving_toward_head}")
                                        print(f"       - Confidence: {confidence:.2f}")
                                else:
                                    if self.debug_mode:
                                        print(f"  ✅ HEAD CHECK PASSED: Close but moving AWAY")
                                        print(f"     - Horizontal dist: {horizontal_dist_to_head:.0f}px")
                                        print(f"     - Vertical offset: {vertical_distance:.0f}px")
                            else:
                                if self.debug_mode and elbow_angle:
                                    print(f"  ✅ HEAD CHECK SKIPPED: Elbow too straight ({elbow_angle:.0f}°)")
                        else:
                            if self.debug_mode:
                                print(f"  ✅ HEAD CHECK SKIPPED: Too far horizontally ({horizontal_dist_to_head:.0f}px)")
        
        # 5.4: HIDING IN SHOES 
        left_ankle = self.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint(keypoints, 'right_ankle')
        left_knee_kp = self.get_keypoint(keypoints, 'left_knee')
        right_knee_kp = self.get_keypoint(keypoints, 'right_knee')
        
        if all([wrist, elbow, shoulder, hip]):
            # Check LEFT SHOE
            if left_ankle and left_knee_kp:
                wrist_to_left_ankle = self.distance(wrist, left_ankle)
                wrist_to_left_knee = self.distance(wrist, left_knee_kp)
                
                if wrist_to_left_ankle and wrist_to_left_ankle < 180:
                    is_below_knee = wrist[1] > left_knee_kp[1]
                    
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    is_elbow_bent = elbow_angle and 40 < elbow_angle < 140
                    
                    is_bending = False
                    if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                        hip_y = (left_hip[1] + right_hip[1]) / 2
                        torso_vertical = abs(hip_y - shoulder_y)
                        
                        if torso_vertical < 120:
                            is_bending = True
                    
                    is_targeting_ankle = wrist_to_left_ankle < wrist_to_left_knee * 0.7
                    
                    criteria_met = sum([is_below_knee, is_elbow_bent, is_bending, is_targeting_ankle])
                    
                    if criteria_met >= 1:
                        distance_conf = 1.0 - (wrist_to_left_ankle / 120)
                        angle_conf = 0.8 if is_elbow_bent else 0.5
                        bending_conf = 1.0 if is_bending else 0.6
                        
                        confidence = (distance_conf + angle_conf + bending_conf) / 3 * 0.92
                        
                        suspicious_poses.append((
                            SuspiciousPose.HIDING_IN_SHOE_LEFT,
                            confidence,
                            f"🚨 Memasukkan barang ke SEPATU KIRI ({grabbed_hand.upper()})"
                        ))
                        
                        if self.debug_mode:
                            print(f"    👟 SHOE LEFT DETECTED:")
                            print(f"       - Hand: {grabbed_hand.upper()}")
                            print(f"       - Distance to ankle: {wrist_to_left_ankle:.0f}px")
                            print(f"       - Criteria met: {criteria_met}/4")
            
            # Check RIGHT SHOE 
            if right_ankle and right_knee_kp:
                wrist_to_right_ankle = self.distance(wrist, right_ankle)
                wrist_to_right_knee = self.distance(wrist, right_knee_kp)
                
                if wrist_to_right_ankle and wrist_to_right_ankle < 120:
                    is_below_knee = wrist[1] > right_knee_kp[1]
                    
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    is_elbow_bent = elbow_angle and 40 < elbow_angle < 140
                    
                    is_bending = False
                    if all([left_shoulder, right_shoulder, left_hip, right_hip]):
                        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                        hip_y = (left_hip[1] + right_hip[1]) / 2
                        torso_vertical = abs(hip_y - shoulder_y)
                        
                        if torso_vertical < 150:
                            is_bending = True
                    
                    is_targeting_ankle = wrist_to_right_ankle < wrist_to_right_knee * 0.7
                    
                    criteria_met = sum([is_below_knee, is_elbow_bent, is_bending, is_targeting_ankle])
                    
                    if criteria_met >= 3:
                        distance_conf = 1.0 - (wrist_to_right_ankle / 120)
                        angle_conf = 0.8 if is_elbow_bent else 0.5
                        bending_conf = 1.0 if is_bending else 0.6
                        
                        confidence = (distance_conf + angle_conf + bending_conf) / 3 * 0.92
                        
                        suspicious_poses.append((
                            SuspiciousPose.HIDING_IN_SHOE_RIGHT,
                            confidence,
                            f"🚨 Memasukkan barang ke SEPATU KANAN ({grabbed_hand.upper()})"
                        ))
        
        # 5.5: WALKING WITH ITEM
        if is_walking and grabbed_hand:
            confidence = 0.85
            suspicious_poses.append((
                SuspiciousPose.HANDS_NEAR_BODY,
                confidence,
                f"🚶 Berjalan membawa barang setelah grab ({grabbed_hand.upper()})"
            ))
            
            if self.debug_mode:
                print(f"  🎯 {len(suspicious_poses)} POSE(S) DETECTED for {grabbed_hand.upper()} hand:")
        
        # ═══════════════════════════════════════════════════════════
        # FINAL DEBUG OUTPUT
        # ═══════════════════════════════════════════════════════════
        if self.debug_mode and len(suspicious_poses) > 0:
            print(f"  🎯 {len(suspicious_poses)} POSE(S) DETECTED for {grabbed_hand.upper()} hand:")
            for pose in suspicious_poses:
                print(f"      - {pose[2]} (conf: {pose[1]:.2f})")
        
        return suspicious_poses

    def detect_returning_item(self, keypoints, track_id):
        """
        Deteksi apakah orang mengembalikan barang ke shelf
        Returns: (is_returning, confidence)
        """
        track = self.person_tracks[track_id]
        
        # Hanya cek jika sudah grab
        if not track['grab_detected']:
            return False, 0.0
        
        grabbed_hand = track['grabbed_hand']
        if not grabbed_hand or grabbed_hand not in ['left', 'right']:
            if self.debug_mode:
                print(f"⚠️ Track {track_id}: Invalid grabbed_hand in detect_returning_item")
            return False, 0.0
        
        # Get keypoints
        wrist_key = f'{grabbed_hand}_wrist'
        shoulder_key = f'{grabbed_hand}_shoulder'
        elbow_key = f'{grabbed_hand}_elbow'
        
        wrist = self.get_keypoint(keypoints, wrist_key)
        shoulder = self.get_keypoint(keypoints, shoulder_key)
        elbow = self.get_keypoint(keypoints, elbow_key)
        
        if not all([wrist, shoulder, elbow]):
            return False, 0.0
        
        # KRITERIA: Tangan NAIK kembali ke atas shoulder level
        is_hand_raised = wrist[1] < shoulder[1]  

        # KRITERIA 2: Arm extended (reaching back to shelf)
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        is_extended = elbow_angle and elbow_angle > 120
        
        # KRITERIA 3: Check movement history (naik dari bawah)
        is_moving_up = False
        if len(track['wrist_positions']) >= 3:
            # ✅ FIX APPLIED HERE - Convert deque to list first
            recent_wrists = [p for p in list(track['wrist_positions'])[-5:] 
                            if p[0] == grabbed_hand]
            
            if len(recent_wrists) >= 2:
                prev_y = recent_wrists[-2][1][1]
                curr_y = wrist[1]
                is_moving_up = curr_y < prev_y - 15  
        
        # DECISION
        is_returning = is_hand_raised and is_extended and is_moving_up
        confidence = 0.85 if is_returning else 0.0
        
        return is_returning, confidence

    def update_phase(self, track_id, keypoints, current_frame, bbox):
        """Update fase deteksi dengan zone detection - FIXED LOGIC"""
        track = self.person_tracks[track_id]
        current_phase = track['phase']
        
        # FASE 1: IDLE -> REACHING_SHELF
        if current_phase == DetectionPhase.IDLE:
            is_reaching, hand_side, confidence, reach_type = self.detect_hand_reaching(keypoints, track_id)
            
            if is_reaching:
                if not hand_side or hand_side not in ['left', 'right']:
                    if self.debug_mode:
                        print(f"⚠️ Track {track_id}: Invalid hand_side='{hand_side}' from detect_hand_reaching")
                    
                    # Reset counters and skip
                    track['hand_extended_frames'] = 0
                    track['hand_extended'] = False
                    return False, [], []

                is_sufficient, kp_conf = self.check_phase_keypoint_requirements(
                    keypoints, DetectionPhase.REACHING_SHELF, hand_side
                )
            
                if not is_sufficient:
                    if self.debug_mode:
                        print(f"⚠️ Track {track_id}: Insufficient keypoints for REACHING")
                        print(f"   Keypoint confidence: {kp_conf:.1%}")

                    track['grabbed_hand'] = None
                    track['hand_extended_frames'] = 0
                    track['hand_extended'] = False
                    
                    return False, [], []

                track['hand_extended'] = True
                track['hand_extended_frames'] += 1
                track['grabbed_hand'] = hand_side
                track['reach_type'] = reach_type 
                
                if track['hand_extended_frames'] >= self.GRAB_THRESHOLDS['min_extension_frames']:
                    track['phase'] = DetectionPhase.REACHING_SHELF
                    track['phase_start_frame'] = current_frame
                    
                    if self.debug_mode:
                        print(f"🟡 Track {track_id}: IDLE -> REACHING ({hand_side} hand)")
            else:
                track['hand_extended_frames'] = 0
                track['hand_extended'] = False
                track['grabbed_hand'] = None
        
        # FASE 2: REACHING_SHELF -> GRABBING
        elif current_phase == DetectionPhase.REACHING_SHELF:
            grabbed_hand = track['grabbed_hand']

            if not grabbed_hand or grabbed_hand not in ['left', 'right']:
                # Invalid state - timeout faster
                if self.debug_mode:
                    print(f"⚠️ Track {track_id}: Invalid grabbed_hand='{grabbed_hand}' in REACHING phase")
                
                frames_in_reaching = current_frame - track['phase_start_frame']
                if frames_in_reaching > 10: 
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: Invalid hand → IDLE")
                
                return False, [], []
    
            # ✅ STEP 3.1: CHECK keypoint sufficiency DULU
            is_sufficient, kp_conf = self.check_phase_keypoint_requirements(
                keypoints, DetectionPhase.GRABBING, grabbed_hand
            )
            
            if not is_sufficient:
                # ❌ Keypoints tidak cukup - FASTER timeout
                frames_in_reaching = current_frame - track['phase_start_frame']
                
                if self.debug_mode:
                    print(f"⚠️ Track {track_id}: Insufficient keypoints for GRABBING detection")
                    print(f"   Keypoint confidence: {kp_conf:.1%}")
                    print(f"   Frames in REACHING: {frames_in_reaching}")
                
                # Timeout lebih cepat (15 frames instead of 30)
                if frames_in_reaching > 15:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: Lost keypoints → IDLE")
                
                return False, [], []  # STOP processing
            
            is_grabbing, confidence = self.detect_grabbing_motion(keypoints, track_id)
            
            if is_grabbing:
                track['phase'] = DetectionPhase.GRABBING
                track['grab_detected'] = True
                track['grab_frame'] = current_frame
                track['phase_start_frame'] = current_frame
                
                if self.debug_mode:
                    print(f"🟠 Track {track_id}: REACHING -> GRABBING (confidence: {confidence:.2f})")
                    print(f"   Keypoint confidence: {kp_conf:.1%}")
                return False, [], []
            else:
                is_still_reaching, _, _, _ = self.detect_hand_reaching(keypoints, track_id)
                
                if not is_still_reaching:
                    grabbed_hand = track['grabbed_hand']
                    wrist_key = f'{grabbed_hand}_wrist'
                    shoulder_key = f'{grabbed_hand}_shoulder'
                    hip_key = f'{grabbed_hand}_hip'
                    
                    wrist = self.get_keypoint(keypoints, wrist_key)
                    shoulder = self.get_keypoint(keypoints, shoulder_key)
                    hip = self.get_keypoint(keypoints, hip_key)
                    
                    if all([wrist, shoulder, hip]):
                        wrist_to_shoulder = self.distance(wrist, shoulder)
                        wrist_to_hip = self.distance(wrist, hip)
                        
                        if (wrist_to_shoulder and wrist_to_shoulder < 120) or \
                           (wrist_to_hip and wrist_to_hip < 100):
                            track['phase'] = DetectionPhase.GRABBING
                            track['grab_detected'] = True
                            track['grab_frame'] = current_frame
                            track['phase_start_frame'] = current_frame
                            
                            if self.debug_mode:
                                print(f"🟠 Track {track_id}: REACHING -> GRABBING (implicit)")
                            return False, [], []
                        else:
                            if current_frame - track['phase_start_frame'] > 30:
                                track['phase'] = DetectionPhase.IDLE
                                track['hand_extended'] = False
                                track['hand_extended_frames'] = 0
                                if self.debug_mode:
                                    print(f"⚪ Track {track_id}: REACHING timeout -> IDLE")
                else:
                    if current_frame - track['phase_start_frame'] > 60:
                        track['phase'] = DetectionPhase.IDLE
                        track['hand_extended'] = False
                        track['hand_extended_frames'] = 0
                        if self.debug_mode:
                            print(f"⚪ Track {track_id}: REACHING timeout -> IDLE")
        
        # FASE 3: GRABBING 
        elif current_phase == DetectionPhase.GRABBING:
            grabbed_hand = track['grabbed_hand']

            if not grabbed_hand or grabbed_hand not in ['left', 'right']:
                if self.debug_mode:
                    print(f"⚠️ Track {track_id}: Invalid grabbed_hand='{grabbed_hand}' in GRABBING phase")
                    print(f"   Resetting to IDLE")
                
                track['phase'] = DetectionPhase.IDLE
                self.reset_track(track_id)
                return False, [], []

            # STEP 3A: CHECK RETURNING ITEM
            is_returning, return_conf = self.detect_returning_item(keypoints, track_id)
            
            if is_returning:
                # RESET ke IDLE - barang dikembalikan!
                if self.debug_mode:
                    print(f"↩️ Track {track_id}: Item RETURNED to shelf → IDLE")
                
                track['phase'] = DetectionPhase.IDLE
                self.reset_track(track_id)
                return False, [], []
            
            # STEP 3B: CHECK NATURAL POSITION + WALKING
            is_walking, speed, direction = self.detect_walking_state(keypoints, track_id, bbox)
            
            # Get natural position check
            shoulder_key = f'{grabbed_hand}_shoulder'
            elbow_key = f'{grabbed_hand}_elbow'
            wrist_key = f'{grabbed_hand}_wrist'
            hip_key = f'{grabbed_hand}_hip'
            
            shoulder = self.get_keypoint(keypoints, shoulder_key)
            elbow = self.get_keypoint(keypoints, elbow_key)
            wrist = self.get_keypoint(keypoints, wrist_key)
            hip = self.get_keypoint(keypoints, hip_key)
            
            is_natural = False
            if all([shoulder, elbow, wrist, hip]):
                if wrist[1] > hip[1]: 
                    horizontal_dist = abs(wrist[0] - hip[0])
                    if horizontal_dist < 70:
                        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                        is_straight = elbow_angle and elbow_angle >= 160
                        
                        wrist_to_hip = self.distance(wrist, hip)
                        is_relax = wrist_to_hip and wrist_to_hip < 80
                        
                        is_natural = is_straight or is_relax
            
            # AUTO-RESET: Natural position + Walking
            if is_natural and is_walking:
                frames_since_grab = current_frame - track['grab_frame']
                
                if frames_since_grab >= 6:
                    if self.debug_mode:
                        print(f"✅ Track {track_id}: Natural + Walking → AUTO-RESET to IDLE")
                        print(f"   Frames since grab: {frames_since_grab}")
                        print(f"   Walking speed: {speed:.1f}px/f")
                    
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    return False, [], []

            if not grabbed_hand or grabbed_hand not in ['left', 'right']:
                # Invalid state - reset to IDLE
                if self.debug_mode:
                    print(f"⚠️ Track {track_id}: Invalid grabbed_hand='{grabbed_hand}' in GRABBING phase")
                    print(f"   Resetting to IDLE")
                
                track['phase'] = DetectionPhase.IDLE
                self.reset_track(track_id)
                return False, [], []
            
            # STEP 4.1: CHECK keypoint sufficiency untuk SUSPICIOUS detection
            is_sufficient, kp_conf = self.check_phase_keypoint_requirements(
                keypoints, DetectionPhase.SUSPICIOUS_MOVEMENT, grabbed_hand
            )
            
            if not is_sufficient:
                # ❌ Keypoints tidak cukup - CAN'T detect suspicious poses reliably
                if self.debug_mode:
                    print(f"⚠️ Track {track_id}: Insufficient keypoints for SUSPICIOUS detection")
                    print(f"   Keypoint confidence: {kp_conf:.1%}")
                
                # DECAY suspicious counters (jangan increment)
                track['suspicious_frame_count'] = max(0, track['suspicious_frame_count'] - 1)
                track['last_normal_frame'] = current_frame
                
                # RESET zone penetration counters
                for zone_name in track['wrist_in_zone_frames'].keys():
                    if track['wrist_in_zone_frames'][zone_name] > 0:
                        track['wrist_in_zone_frames'][zone_name] = max(
                            0, 
                            track['wrist_in_zone_frames'][zone_name] - 1
                        )
                
                # Timeout check
                frames_since_grab = current_frame - track['grab_frame']
                if frames_since_grab >= self.SUSPICIOUS_VALIDATION['timeout_normal_behavior']:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: GRABBING -> IDLE (Poor keypoints + timeout)")
                
                return False, [], []  

            if not track['pocket_zones']:
                self.initialize_pocket_zones(track_id, keypoints)
            
            # ✅ STEP 1: Deteksi rotasi DULU sebelum apapun
            is_rotating = False 
            rotation_conf = 0.0
            
            # ✅ STEP 3: CEK POSISI NATURAL (TANGAN LURUS KEBAWAH)
            grabbed_hand = track['grabbed_hand']
            shoulder_key = f'{grabbed_hand}_shoulder'
            elbow_key = f'{grabbed_hand}_elbow'
            wrist_key = f'{grabbed_hand}_wrist'
            hip_key = f'{grabbed_hand}_hip'
            knee_key = f'{grabbed_hand}_knee'
            
            shoulder = self.get_keypoint(keypoints, shoulder_key)
            elbow = self.get_keypoint(keypoints, elbow_key)
            wrist = self.get_keypoint(keypoints, wrist_key)
            hip = self.get_keypoint(keypoints, hip_key)
            knee = self.get_keypoint(keypoints, knee_key)
            
            # CEK: Apakah tangan dalam posisi NATURAL (straight down)?
            is_natural_position = False
            natural_check_details = {}
            
            if all([shoulder, elbow, wrist, hip]):
                # HELPER FUNCTION: Cek Natural Position
                def is_hand_in_natural_position(shoulder, elbow, wrist, hip, knee=None):
                    """
                    Cek apakah tangan dalam posisi natural (lurus kebawah di samping badan)
                    Returns: (is_natural, details_dict)
                    """
                    details = {
                        'wrist_below_hip': False,
                        'horizontal_distance': 0,
                        'elbow_angle': 0,
                        'wrist_to_hip_distance': 0,
                        'vertical_alignment': 0,
                        'is_straight_arm': False,
                        'is_relax_position': False,
                        'is_natural': False
                    }
                    
                    # CHECK 1: Wrist harus di BAWAH hip (Y-coordinate lebih besar)
                    if wrist[1] <= hip[1]:
                        details['wrist_below_hip'] = False
                        return False, details
                    
                    details['wrist_below_hip'] = True
                    
                    # CHECK 2: Horizontal distance dari hip harus MINIMAL (di samping badan)
                    horizontal_dist = abs(wrist[0] - hip[0])
                    details['horizontal_distance'] = horizontal_dist
                    
                    # Threshold: 70px 
                    if horizontal_dist > 70:  
                        return False, details
                    
                    # CHECK 3: Elbow angle harus STRAIGHT 
                    elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                    details['elbow_angle'] = elbow_angle if elbow_angle else 0
                    
                    if elbow_angle and elbow_angle >= 160:
                        details['is_straight_arm'] = True
                        details['is_natural'] = True
                        return True, details
                    
                    # CHECK 4: ALTERNATIF - Wrist sangat dekat dengan hip 
                    wrist_to_hip = self.distance(wrist, hip)
                    details['wrist_to_hip_distance'] = wrist_to_hip if wrist_to_hip else 0
                    
                    if wrist_to_hip and wrist_to_hip < 80:
                        if horizontal_dist < 50:
                            details['is_relax_position'] = True
                            details['is_natural'] = True
                            return True, details
                    
                    # CHECK 5: Vertical alignment (tangan lurus kebawah secara geometris)
                    elbow_wrist_horizontal = abs(wrist[0] - elbow[0])
                    elbow_wrist_vertical = abs(wrist[1] - elbow[1])
                    
                    if elbow_wrist_vertical > 0:
                        vertical_ratio = elbow_wrist_horizontal / elbow_wrist_vertical
                        details['vertical_alignment'] = vertical_ratio
                        
                        # Jika ratio < 0.25 = sangat vertikal 
                        if vertical_ratio < 0.25:
                           
                            if horizontal_dist < 60:
                                details['is_natural'] = True
                                return True, details
                    
                    # CHECK 6: Wrist dekat dengan knee (posisi standing relax)
                    if knee:
                        wrist_to_knee = self.distance(wrist, knee)
                       
                        if wrist_to_knee and wrist_to_knee < 100 and horizontal_dist < 50:
                            details['is_natural'] = True
                            return True, details
                    
                    return False, details
                
                # Panggil helper function
                is_natural_position, natural_check_details = is_hand_in_natural_position(
                    shoulder, elbow, wrist, hip, knee
                )
            
            # STEP 4: JIKA POSISI NATURAL → RESET counter & SKIP deteksi
            if is_natural_position:
                # Decay suspicious frame count dengan cepat
                track['suspicious_frame_count'] = max(0, track['suspicious_frame_count'] - 2)
                track['last_normal_frame'] = current_frame
                
                # Reset zone penetration counters (jika ada)
                for zone_name in track['wrist_in_zone_frames'].keys():
                    if track['wrist_in_zone_frames'][zone_name] > 0:
                        track['wrist_in_zone_frames'][zone_name] = max(
                            0, 
                            track['wrist_in_zone_frames'][zone_name] - 1
                        )
                
                # DEBUG output
                if self.debug_mode:
                    print(f"✅ Track {track_id}: Hand in NATURAL position - No suspicious detection")
                    print(f"   Details:")
                    print(f"     - Wrist below hip: {natural_check_details['wrist_below_hip']}")
                    print(f"     - Horizontal distance: {natural_check_details['horizontal_distance']:.0f}px")
                    print(f"     - Elbow angle: {natural_check_details['elbow_angle']:.0f}°")
                    print(f"     - Wrist to hip: {natural_check_details['wrist_to_hip_distance']:.0f}px")
                    print(f"     - Straight arm: {natural_check_details['is_straight_arm']}")
                    print(f"     - Relax position: {natural_check_details['is_relax_position']}")
                
                # Timeout check 
                frames_since_grab = current_frame - track['grab_frame']
                if frames_since_grab >= self.SUSPICIOUS_VALIDATION['timeout_normal_behavior']:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: GRABBING -> IDLE (Natural position timeout)")
                
                # RETURN FALSE - tidak ada alert, tidak ada suspicious
                return False, [], []
            
            # STEP 5: Lanjutkan deteksi HANYA jika TIDAK natural position
            zone_penetration, zone_details, zone_conf, zone_poses = self.detect_zone_penetration(
                track_id, keypoints
            )
            
            suspicious_poses = self.detect_suspicious_poses(
                keypoints, 
                grabbed_hand=grabbed_hand, 
                is_rotating=False,
                is_walking=track['walking_with_item']
            )
            
            # Update buffer 
            track['suspicious_buffer'].append({
                'frame': current_frame,
                'has_suspicious': len(suspicious_poses) > 0 or zone_penetration,
                'poses': suspicious_poses + zone_poses,
                'zone_details': zone_details,
                'high_severity': any(p[0] in self.SUSPICIOUS_VALIDATION['high_severity_poses'] 
                                    for p in suspicious_poses) or zone_penetration
            })
            
            # STEP 6: ZONE PENETRATION 
            transition_to_suspicious = False
            transition_reason = ""
            
            if zone_penetration and zone_details:
                very_deep_zones = [z for z in zone_details 
                                if z['depth'] >= 0.60  
                                and z['frames_in_zone'] >= 8]  
                
                if very_deep_zones:
                    track['phase'] = DetectionPhase.ALERT
                    track['phase_start_frame'] = current_frame
                    
                    track['suspicion_score'] += 60
                    track['zone_penetration_detected'] = True
                    track['zone_penetration_frames'] = current_frame
                    track['zone_penetration_zones'] = [z['zone'] for z in very_deep_zones]
                    
                    # Update pose counts
                    for zone_pose in zone_poses:
                        track['pose_counts'][zone_pose[0]] += 10
                    
                    # Generate alert reasons
                    alert_reasons = []
                    for z in very_deep_zones:
                        zone_name = z['zone'].replace('_', ' ').upper()
                        alert_reasons.append(f"🚨 DEEP {zone_name} ({z['depth']:.0%})")
                    
                    if self.debug_mode:
                        print(f"🚨🚨 Track {track_id}: INSTANT ALERT - VERY DEEP ZONE PENETRATION!")
                        for z in very_deep_zones:
                            print(f"    - {z['zone']}: depth {z['depth']:.1%}, {z['frames_in_zone']}f")
                    
                    return True, suspicious_poses + zone_poses, alert_reasons
            
            # STEP 7: CEK IMMEDIATE HIGH SEVERITY ZONE 
            if zone_penetration:
                high_severity_zones = [z for z in zone_details 
                                    if z['depth'] >= self.ZONE_THRESHOLDS['immediate_suspicious_depth']
                                    and z['frames_in_zone'] >= self.ZONE_THRESHOLDS['immediate_suspicious_frames']]
                
                if high_severity_zones:
                    transition_to_suspicious = True
                    transition_reason = f"ZONE_PENETRATION: {[z['zone'] for z in high_severity_zones]}"
                    
                    track['suspicion_score'] += 35
                    track['consecutive_suspicious'] = 10
                    
                    for zone_pose in zone_poses:
                        track['pose_counts'][zone_pose[0]] += 1
                    
                    track['zone_penetration_detected'] = True
                    track['zone_penetration_frames'] = current_frame
                    track['zone_penetration_zones'] = [z['zone'] for z in high_severity_zones]
                
                else:
                    if 'zone_consecutive_frames' not in track:
                        track['zone_consecutive_frames'] = 0
                    
                    track['zone_consecutive_frames'] += 1
                    
                    if track['zone_consecutive_frames'] >= 5:
                        transition_to_suspicious = True
                        transition_reason = f"ACCUMULATED_ZONE: {track['zone_consecutive_frames']}f"
                        
                        track['zone_penetration_detected'] = True
                        track['zone_penetration_frames'] = current_frame
                        track['zone_penetration_zones'] = [z['zone'] for z in zone_details]
                        
                        for zone_pose in zone_poses:
                            track['pose_counts'][zone_pose[0]] += 1
            else:
                track['zone_consecutive_frames'] = 0
            
            # STEP 8: CEK HIGH SEVERITY POSE 
            if not transition_to_suspicious and suspicious_poses:
                
                # FILTER OUT false positive HEAD POSES during REACHING phase
                if track['phase'] == DetectionPhase.REACHING_SHELF:
                    # Remove head poses if still in reaching phase
                    suspicious_poses = [p for p in suspicious_poses 
                                    if p[0] not in [SuspiciousPose.HIDING_IN_HAT, 
                                                    SuspiciousPose.HAND_ON_HEAD]]
                    
                    if self.debug_mode and len(suspicious_poses) == 0:
                        print(f"  ⚠️ Filtered out head poses - Still REACHING!")
                
                # FILTER 1: HEAD POSES (threshold lebih rendah, lebih sensitif)
                head_poses = [p for p in suspicious_poses 
                            if p[0] in [SuspiciousPose.HIDING_IN_HAT, SuspiciousPose.HAND_ON_HEAD]
                            and p[1] >= self.SUSPICIOUS_VALIDATION.get('head_pose_threshold', 0.60)]
    
                
                # FILTER 2: NON-HEAD HIGH SEVERITY POSES (threshold normal)
                non_head_severity = [p for p in suspicious_poses 
                                    if p[0] in self.SUSPICIOUS_VALIDATION['high_severity_poses']
                                    and p[0] not in [SuspiciousPose.HIDING_IN_HAT, SuspiciousPose.HAND_ON_HEAD]
                                    and p[1] >= self.SUSPICIOUS_VALIDATION['suspicious_confidence_threshold']]
                
                # DEBUG OUTPUT 
                if self.debug_mode and suspicious_poses:
                    print(f"  🔍 SEVERITY CHECK:")
                    print(f"     - Head poses: {len(head_poses)} (threshold: {self.SUSPICIOUS_VALIDATION.get('head_pose_threshold', 0.60):.2f})")
                    print(f"     - Non-head severity: {len(non_head_severity)} (threshold: {self.SUSPICIOUS_VALIDATION['suspicious_confidence_threshold']:.2f})")
                    print(f"     - Suspicious frame count: {track['suspicious_frame_count']}")
                
                # TRIGGER LOGIC: HEAD POSES 
                if head_poses:
                    track['suspicious_frame_count'] += 1
                    
                    if track['suspicious_frame_count'] >= 2:  
                        transition_to_suspicious = True
                        transition_reason = f"HEAD_POSE: {[p[2] for p in head_poses]}"
                        
                        track['suspicion_score'] += 20
                        if self.debug_mode:
                            print(f"  🎩 HEAD POSE BONUS: +20 → score={track['suspicion_score']:.0f}")
                
                # TRIGGER LOGIC: NON-HEAD POSES 
                elif non_head_severity:
                    track['suspicious_frame_count'] += 1
                    
                    if track['suspicious_frame_count'] >= 3:  
                        transition_to_suspicious = True
                        transition_reason = f"HIGH_SEVERITY_POSE: {[p[2] for p in non_head_severity]}"
                
                else:
                    # RESET counter jika tidak ada pose yang memenuhi syarat
                    if not zone_penetration:
                        track['suspicious_frame_count'] = max(0, track['suspicious_frame_count'] - 1)
                        track['last_normal_frame'] = current_frame
            else:
                if not zone_penetration:
                    track['suspicious_frame_count'] = max(0, track['suspicious_frame_count'] - 1)
                    track['last_normal_frame'] = current_frame
            
            # STEP 9: CEK CONSISTENT SUSPICIOUS BEHAVIOR
            if not transition_to_suspicious and len(track['suspicious_buffer']) >= self.SUSPICIOUS_VALIDATION['pose_consistency_window']:
                recent_suspicious = [b for b in track['suspicious_buffer'] if b['has_suspicious']]
                suspicious_ratio = len(recent_suspicious) / len(track['suspicious_buffer'])
                
                all_poses = []
                for b in recent_suspicious:
                    all_poses.extend([p[0] for p in b['poses']])
                unique_poses = len(set(all_poses))
                
                if suspicious_ratio >= 0.50 and unique_poses >= self.SUSPICIOUS_VALIDATION['min_unique_poses']:
                    avg_confidence = np.mean([p[1] for b in recent_suspicious for p in b['poses']]) if recent_suspicious else 0
                    
                    if avg_confidence >= self.SUSPICIOUS_VALIDATION['suspicious_confidence_threshold']:
                        transition_to_suspicious = True
                        transition_reason = f"CONSISTENT: ratio={suspicious_ratio:.1%}, poses={unique_poses}"
            
            # STEP 10: TRANSISI KE SUSPICIOUS 
            if transition_to_suspicious:
                track['phase'] = DetectionPhase.SUSPICIOUS_MOVEMENT
                track['phase_start_frame'] = current_frame
                
                if self.debug_mode:
                    print(f"🔴 Track {track_id}: GRABBING -> SUSPICIOUS")
                    print(f"    Reason: {transition_reason}")
                    if zone_details:
                        for z in zone_details:
                            print(f"    - {z['zone']}: {z['hand']} (depth: {z['depth']:.1%}, frames: {z['frames_in_zone']})")
            
            # STEP 11: TIMEOUT CHECK - Kembali ke IDLE jika normal behavior
            frames_since_grab = current_frame - track['grab_frame']
            
            if frames_since_grab >= self.SUSPICIOUS_VALIDATION['timeout_normal_behavior']:
                normal_frames = len([b for b in track['suspicious_buffer'] if not b['has_suspicious']])
                
                if normal_frames >= len(track['suspicious_buffer']) * 0.7:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
                    
                    if self.debug_mode:
                        print(f"⚪ Track {track_id}: GRABBING -> IDLE (Normal behavior)")
                    return False, [], []
        
        # FASE 4: SUSPICIOUS_MOVEMENT
        elif current_phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
            grabbed_hand = track['grabbed_hand']

            # Deteksi rotasi
            is_rotating, _, rotation_conf = self.detect_body_rotation(keypoints, track_id)
            
            # Jika sedang rotate, SKIP deteksi dan kembali ke GRABBING
            if is_rotating and track['is_rotating']:
                track['phase'] = DetectionPhase.GRABBING
                track['phase_start_frame'] = current_frame
                track['suspicion_score'] = max(0, track['suspicion_score'] - 20)  # Kurangi score
                track['consecutive_suspicious'] = 0
                
                if self.debug_mode:
                    print(f"🔄 Track {track_id}: SUSPICIOUS -> GRABBING (Body rotation detected)")
                
                return False, [], []
            
            # Lanjutkan deteksi normal jika TIDAK rotate
            suspicious_poses = self.detect_suspicious_poses(keypoints, grabbed_hand=grabbed_hand, is_rotating=False)
            zone_penetration, zone_details, _, zone_poses = self.detect_zone_penetration(track_id, keypoints)

            all_poses = suspicious_poses + zone_poses
            
            if all_poses or zone_penetration:
                self.update_suspicion_score(track_id, all_poses, zone_details, current_frame)
                
                should_alert, reasons = self.should_alert(track_id, all_poses, current_frame)
                
                if should_alert:
                    track['phase'] = DetectionPhase.ALERT
                    return True, all_poses, reasons
            else:
                if current_frame - track['phase_start_frame'] > 10:
                    track['phase'] = DetectionPhase.IDLE
                    self.reset_track(track_id)
        
        # FASE 5: ALERT
        elif current_phase == DetectionPhase.ALERT:
            if current_frame - track['last_alert_frame'] > self.SUSPICIOUS_THRESHOLDS['alert_cooldown'] * 3:
                track['phase'] = DetectionPhase.IDLE
                self.reset_track(track_id)
        
        return False, [], []

    def log_keypoint_quality(self, track_id, phase, is_sufficient, confidence, keypoints):
        """
        Helper method untuk log keypoint quality dengan detail
        """
        if not self.debug_mode:
            return
        
        track = self.person_tracks[track_id]
        
        if not is_sufficient:
            print(f"\n⚠️ KEYPOINT QUALITY CHECK FAILED")
            print(f"   Track ID: {track_id}")
            print(f"   Phase: {phase.value if hasattr(phase, 'value') else phase}")
            print(f"   Confidence: {confidence:.1%}")
            
            # Show which keypoints are missing
            required_keypoints = self.REQUIRED_KEYPOINTS.get(
                str(phase).split('.')[-1].lower(), []
            )
            
            missing = []
            for kp_name in required_keypoints:
                kp = self.get_keypoint(keypoints, kp_name)
                if not kp:
                    missing.append(kp_name)
            
            if missing:
                print(f"   Missing keypoints: {', '.join(missing)}")
            
            print(f"   Action: Detection SKIPPED for this frame")
        else:
            print(f"✅ Keypoint quality OK for {phase.value if hasattr(phase, 'value') else phase}")
            print(f"   Track ID: {track_id}")
            print(f"   Confidence: {confidence:.1%}")
    
    def reset_track(self, track_id):
        """Reset tracking untuk person tertentu"""
        track = self.person_tracks[track_id]
        track['hand_extended'] = False
        track['hand_extended_frames'] = 0
        track['grab_detected'] = False
        track['grabbed_hand'] = None
        track['suspicion_score'] = 0
        track['consecutive_suspicious'] = 0
        track['suspicious_poses'].clear()
        track['wrist_positions'].clear()
        track['wrist_in_zone_frames'].clear()
        track['max_zone_depth'].clear()
        track['zone_penetration_detected'] = False
        track['zone_penetration_frames'] = 0
        track['zone_penetration_zones'].clear()
        track['zone_consecutive_frames'] = 0  
        track['suspicious_frame_count'] = 0
        track['position_history'].clear()
        track['is_walking'] = False
        track['walking_frames'] = 0
        track['stationary_frames'] = 0
        track['last_movement_speed'] = 0.0
        track['walking_direction'] = None
    
    def update_suspicion_score(self, track_id, suspicious_poses, zone_details, current_frame):
        """Update suspicion score dengan zone detection"""
        track = self.person_tracks[track_id]
        
        track['suspicion_score'] = max(0, track['suspicion_score'] - self.SUSPICIOUS_THRESHOLDS['score_decay'])
        track['total_frames_tracked'] += 1
        
        # Zone detection scoring
        if zone_details:
            for zone_detail in zone_details:
                zone_name = zone_detail['zone']
                depth = zone_detail['depth']
                confidence = zone_detail['confidence']
                
                if 'pants' in zone_name:
                    base_score = 30  
                    depth_multiplier = 1.5
                else:
                    base_score = 26  
                    depth_multiplier = 1.3
                
                zone_score = base_score * confidence * (1 + depth * depth_multiplier)
                track['suspicion_score'] += zone_score
                
                if self.debug_mode:
                    print(f"  Zone {zone_name}: +{zone_score:.1f} (depth: {depth:.1%}, conf: {confidence:.2f})")
        
        if suspicious_poses:
            track['suspicious_poses'].append({
                'frame': current_frame,
                'poses': suspicious_poses
            })
            
            track['consecutive_suspicious'] += 1
            
            for pose_type, confidence, _ in suspicious_poses:
                track['pose_counts'][pose_type] += 1

                if pose_type == SuspiciousPose.HIDING_IN_HAT:
                    track['suspicion_score'] += 35 * confidence 
                    track['head_pose_frames'] += 1
                    track['head_pose_detected'] = True
                    if not track['head_pose_first_frame']:
                        track['head_pose_first_frame'] = current_frame
                    track['head_pose_max_confidence'] = max(track['head_pose_max_confidence'], confidence)
                    
                    if self.debug_mode:
                        print(f"  🎩 HEAD POSE BONUS: +{35 * confidence:.1f} → score={track['suspicion_score']:.0f}")
                
                elif pose_type == SuspiciousPose.HAND_ON_HEAD:
                    track['suspicion_score'] += 28 * confidence 
                    track['head_pose_frames'] += 1
                    track['head_pose_detected'] = True
                    if not track['head_pose_first_frame']:
                        track['head_pose_first_frame'] = current_frame
                    track['head_pose_max_confidence'] = max(track['head_pose_max_confidence'], confidence)
                    
                    if self.debug_mode:
                        print(f"  🎩 HEAD POSE BONUS: +{28 * confidence:.1f} → score={track['suspicion_score']:.0f}")
                
                if pose_type == SuspiciousPose.HIDING_UNDER_CLOTHING:
                    track['suspicion_score'] += 25 * confidence
                elif pose_type == SuspiciousPose.PUTTING_IN_PANTS_POCKET:
                    track['suspicion_score'] += 22 * confidence
                elif pose_type == SuspiciousPose.SQUATTING_LOW:
                    track['suspicion_score'] += 18 * confidence
                elif pose_type == SuspiciousPose.BENDING_DOWN:
                    track['suspicion_score'] += 15 * confidence
                elif pose_type == SuspiciousPose.HIDING_IN_SHOE_LEFT:
                    track['suspicion_score'] += 35 * confidence 
                elif pose_type == SuspiciousPose.HIDING_IN_SHOE_RIGHT:
                    track['suspicion_score'] += 35 * confidence 
                elif pose_type == SuspiciousPose.REACHING_SHOE_AREA:
                    track['suspicion_score'] += 25 * confidence 
                elif pose_type == SuspiciousPose.CROUCHING:
                    track['suspicion_score'] += 12 * confidence
                elif pose_type in [SuspiciousPose.ZONE_PANTS_POCKET_LEFT, SuspiciousPose.ZONE_PANTS_POCKET_RIGHT]:
                    track['suspicion_score'] += 28 * confidence
                elif pose_type in [SuspiciousPose.ZONE_JACKET_POCKET_LEFT, SuspiciousPose.ZONE_JACKET_POCKET_RIGHT]:
                    track['suspicion_score'] += 24 * confidence
                elif pose_type == SuspiciousPose.HIDING_IN_HAT:
                    track['suspicion_score'] += 30 * confidence  
                elif pose_type == SuspiciousPose.HAND_ON_HEAD:
                    track['suspicion_score'] += 22 * confidence  
                elif pose_type == SuspiciousPose.SQUATTING_LOW:
                    track['suspicion_score'] += 18 * confidence
                
            if track['consecutive_suspicious'] >= self.SUSPICIOUS_THRESHOLDS['min_consecutive_for_bonus']:
                track['suspicion_score'] += self.SUSPICIOUS_THRESHOLDS['continuous_pose_bonus']
            
            unique_poses = len(set(p[0] for p in suspicious_poses))
            if unique_poses >= 2:
                track['suspicion_score'] += 20
        else:
            track['consecutive_suspicious'] = 0
        
        track['suspicion_score'] = min(track['suspicion_score'], 100)
        
        suspicious_frame_count = sum(1 for p in track['suspicious_poses'])
        track['suspicious_ratio'] = suspicious_frame_count / max(track['total_frames_tracked'], 1)
        
        return track['suspicion_score']
    
    def should_alert(self, track_id, suspicious_poses, current_frame):
        """Tentukan apakah harus trigger alert - FASTER VERSION"""
        track = self.person_tracks[track_id]
        
        if not track['grab_detected']:
            return False, []
        
        frames_since_alert = current_frame - track['last_alert_frame']
        if frames_since_alert < self.SUSPICIOUS_THRESHOLDS['alert_cooldown']:
            return False, []
        
        reasons = []
        
        # HEAD POSES
        head_poses = [p for p in suspicious_poses 
                    if p[0] in [SuspiciousPose.HIDING_IN_HAT, SuspiciousPose.HAND_ON_HEAD]
                    and p[1] >= 0.60]

        if head_poses and track['consecutive_suspicious'] >= 3:  
            reasons = [f"GRAB+{p[2]}" for p in head_poses[:2]]
            return True, reasons

        if head_poses:
            max_conf = max(p[1] for p in head_poses)
            head_frames = track['head_pose_frames']
            
            instant_high_conf = max_conf >= 0.80 and head_frames >= 2
            instant_medium_conf = max_conf >= 0.60 and head_frames >= 4
            
            if instant_high_conf or instant_medium_conf:
                reasons = [f"GRAB + {p[2]}" for p in head_poses[:2]]
                
                if self.debug_mode:
                    print(f"  🚨 HEAD POSE ALERT TRIGGERED:")
                    print(f"     - Max confidence: {max_conf:.2f}")
                    print(f"     - Head pose frames: {head_frames}")
                    print(f"     - Reasons: {reasons}")
                
                return True, reasons

        # HIGH CONFIDENCE 
        high_conf_poses = [p for p in suspicious_poses 
                        if p[1] >= self.SUSPICIOUS_THRESHOLDS['high_confidence_threshold']]
        if high_conf_poses and track['consecutive_suspicious'] >= 5:  
            reasons = [f"GRAB+{p[2]}" for p in high_conf_poses[:2]]
            return True, reasons

        # ZONE PENETRATION
        if track['zone_penetration_detected']:
            zone_pose_counts = sum(1 for pose_type in track['pose_counts'] 
                                if pose_type in [SuspiciousPose.ZONE_PANTS_POCKET_LEFT,
                                                SuspiciousPose.ZONE_PANTS_POCKET_RIGHT,
                                                SuspiciousPose.ZONE_JACKET_POCKET_LEFT,
                                                SuspiciousPose.ZONE_JACKET_POCKET_RIGHT])
            
            if zone_pose_counts >= 6:  
                for zone_name in track['zone_penetration_zones']:
                    reasons.append(f"GRAB + {zone_name.replace('_', ' ').upper()}")
                return True, reasons
        
        # SUSPICION SCORE 
        if (track['suspicion_score'] >= 80 and  
            track['suspicious_ratio'] >= 0.80): 
            
            top_poses = sorted(track['pose_counts'].items(), 
                            key=lambda x: x[1], reverse=True)[:2]
            for pose_type, count in top_poses:
                if count > 0:
                    reasons.append(f"{pose_type.value}: {count}x")
            
            recent_count = sum(1 for p in track['suspicious_poses'] 
                            if p['frame'] > current_frame - 30)
            if recent_count >= 6: 
                reasons.append(f"GRABBED + {recent_count}f suspicious")
                return True, reasons
        
        # MULTIPLE POSES
        if (len(suspicious_poses) >= 2 and 
            track['suspicion_score'] > 45 and  
            track['consecutive_suspicious'] >= 4): 
            reasons = [f"GRAB+{p[2][:25]}" for p in suspicious_poses[:2]]
            return True, reasons
        
        return False, []
    
    def save_alert_clip(self, track_id, alert_info, current_frame, bbox=None):
        """Save video clip: 5s sebelum + 5s setelah DENGAN VISUAL"""
        try:
            clips_dir = "alert_clips"
            os.makedirs(clips_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"shoplifting_track{track_id}_{timestamp}"
            
            frames_before = int(10 * self.fps)
            frames_after = int(10 * self.fps)
            
            frames_pre_alert = []
            if len(self.frame_buffer) >= frames_before:
                frames_pre_alert = list(self.frame_buffer)[-frames_before:]
            else:
                frames_pre_alert = list(self.frame_buffer)
            
            print(f"📹 Recording alert clip for Track {track_id}...")
            print(f"   - Pre-alert frames: {len(frames_pre_alert)}")
            print(f"   - Waiting for {frames_after} post-alert frames...")
            
            self.recording_alerts[track_id] = {
                'start_frame': current_frame,
                'frames_before': frames_pre_alert,
                'frames_after': [],
                'frames_needed': frames_after,
                'alert_info': alert_info,
                'base_filename': base_filename,
                'alert_frame_index': len(frames_pre_alert),
                'bbox': bbox  
            }
            
            return base_filename
            
        except Exception as e:
            print(f"❌ Error preparing alert clip: {e}")
            return None

    def crop_person_bbox(self, frame, bbox, padding=20):
            """
            Crop person PERSEGI (1:1) - Fokus pada SETENGAH BADAN ATAS + WAJAH
            Args:
                frame: input frame
                bbox: (x1, y1, x2, y2) bounding box
                padding: pixel padding di sekitar bbox
            Returns:
                cropped image (square), crop coordinates
            """
            try:
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center_x = (x1 + x2) // 2
                
                y_top = y1
                y_bottom = y1 + int(bbox_height * 0.55)  
                
                # Hitung tinggi crop area
                crop_height = y_bottom - y_top + (2 * padding)
                
                # BUAT PERSEGI: lebar = tinggi
                square_size = crop_height
                
                # Tentukan koordinat persegi crop
                x1_crop = center_x - (square_size // 2)
                x2_crop = center_x + (square_size // 2)
                y1_crop = y_top - padding
                y2_crop = y_top + square_size - padding
                
                # Pastikan tidak keluar dari frame
                x1_crop = max(0, x1_crop)
                y1_crop = max(0, y1_crop)
                x2_crop = min(w, x2_crop)
                y2_crop = min(h, y2_crop)
                
                # Crop
                cropped = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                
                # Validasi bentuk persegi 
                crop_h, crop_w = cropped.shape[:2]
                if crop_h != crop_w:
                    # Buat canvas persegi dengan ukuran max dimension
                    target_size = max(crop_h, crop_w)
                    square_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    
                    # Paste cropped image di tengah canvas
                    y_offset = (target_size - crop_h) // 2
                    x_offset = (target_size - crop_w) // 2
                    square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped
                    
                    cropped = square_canvas
                
                return cropped, (x1_crop, y1_crop, x2_crop, y2_crop)
            
            except Exception as e:
                print(f"❌ Error cropping person: {e}")
                return None, None
    
    def _clean_alert_info_for_json(self, alert_info):
        """
        Clean alert_info untuk JSON serialization
        Convert Enum keys dan numpy types menjadi native Python types
        """
        import numpy as np
        
        def convert_value(val):
            """Recursively convert numpy types to native Python types"""
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, dict):
                return {convert_value(k): convert_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert_value(item) for item in val]
            elif isinstance(val, tuple):
                return tuple(convert_value(item) for item in val)
            elif hasattr(val, 'value'):  # Enum
                return val.value
            else:
                return val
        
        cleaned = {}
        for key, value in alert_info.items():
            cleaned[convert_value(key)] = convert_value(value)
        
        return cleaned

    def finalize_alert_clip(self, track_id):
        """Finalize dan save clip DENGAN VISUAL MARKER + CROPPED IMAGES"""
        if track_id not in self.recording_alerts:
            return None
        
        try:
            recording = self.recording_alerts[track_id]
            base_filename = recording['base_filename']
            alert_frame_idx = recording.get('alert_frame_index', 0)
            bbox = recording.get('bbox', None)
            
            all_frames = recording['frames_before'] + recording['frames_after']
            
            if len(all_frames) < 10: 
                print(f"⚠️ Only {len(all_frames)} frames for Track {track_id}, still saving...")
            
            clips_dir = "alert_clips"
            video_filename = os.path.join(clips_dir, f"{base_filename}.mp4")
            
            crop_dir = os.path.join(clips_dir, f"{base_filename}_crops")
            os.makedirs(crop_dir, exist_ok=True)
            
            first_frame = all_frames[0]
            height, width = first_frame.shape[:2]
            
            codecs_to_try = [
                ('avc1', '.mp4'),
                ('mp4v', '.mp4'), 
                ('XVID', '.avi'),  
            ]
            
            out = None
            actual_filename = None
            
            for codec_str, ext in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)
                    test_filename = video_filename.replace('.mp4', ext)
                    out = cv2.VideoWriter(test_filename, fourcc, self.fps, (width, height))
                    
                    if out.isOpened():
                        actual_filename = test_filename
                        print(f"✅ Using codec: {codec_str} ({ext})")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    print(f"⚠️ Codec {codec_str} failed: {e}")
                    continue
            
            if out is None or not out.isOpened():
                print(f"❌ Cannot open video writer for Track {track_id} - All codecs failed")
                del self.recording_alerts[track_id]
                return None

            video_filename = actual_filename
            
            crop_count = 0
            # crop_interval = 5
            
            if bbox is not None:
                print(f"📍 Bbox for Track {track_id}: {bbox}")
            else:
                print(f"⚠️ WARNING: Bbox is None for Track {track_id}!")
            
            crop_count = 0

            for idx, frame in enumerate(all_frames):
                frame_copy = frame.copy()
                
                # SAVE CROPPED IMAGE 
                if bbox is not None and len(bbox) == 4:
                    if idx == alert_frame_idx: 
                        try:
                            cropped, crop_coords = self.crop_person_bbox(frame, bbox, padding=30)
                            
                            if cropped is not None and cropped.size > 0:
                                alert_crop_filename = os.path.join(crop_dir, f"ALERT_crop.jpg")
                                success = cv2.imwrite(alert_crop_filename, cropped)
                                
                                if success:
                                    crop_count = 1
                                    print(f"  ✅ Saved ALERT crop: {alert_crop_filename}")
                                else:
                                    print(f"  ❌ Failed to save crop: {alert_crop_filename}")
                        except Exception as e:
                            print(f"  ❌ Error cropping alert frame: {e}")
                else:
                    if idx == alert_frame_idx:
                        print(f"  ⚠️ Cannot crop - bbox is invalid: {bbox}")
                
                out.write(frame_copy)
            
            # Release video writer PROPERLY
            out.release()
            cv2.waitKey(1)  
            
            # Update JSON dengan info lengkap
            json_filename = os.path.join(clips_dir, f"{base_filename}.json")

            # Clean alert_info untuk JSON
            alert_info = recording['alert_info']
            cleaned_alert_info = self._clean_alert_info_for_json(alert_info)
            pose_descriptions = self._generate_pose_descriptions(alert_info)
            
            clip_info = {
                'alert_info': cleaned_alert_info,
                'clip_info': {
                    'video_file': os.path.basename(video_filename),
                    'total_frames': len(all_frames),
                    'pre_alert_frames': len(recording['frames_before']),
                    'post_alert_frames': len(recording['frames_after']),
                    'alert_frame_index': alert_frame_idx,
                    'duration_seconds': len(all_frames) / self.fps,
                    'fps': self.fps,
                    'resolution': f"{width}x{height}",
                    'created_at': datetime.now().isoformat(),
                    'visual_markers': True,
                    'cropped_images': {
                        'folder': f"{base_filename}_crops",
                        'total_crops': crop_count,
                        'alert_crop': 'ALERT_crop.jpg' if crop_count == 1 else 'N/A',  
                        'alert_frame_only': True,  
                        'bbox': [float(x) for x in bbox] if bbox is not None else 'N/A'
                    }
                },
                'detection_summary': {
                    'track_id': int(track_id),  
                    'phase_sequence': str(cleaned_alert_info.get('phase', 'unknown')),
                    'grab_frame': int(cleaned_alert_info.get('grab_frame', 0)),
                    'alert_frame': int(cleaned_alert_info.get('frame', 0)),
                    'grabbed_hand': str(cleaned_alert_info.get('grabbed_hand', 'unknown')),
                    'suspicion_score': float(cleaned_alert_info.get('suspicion_score', 0)),
                    'suspicious_ratio': float(cleaned_alert_info.get('suspicious_ratio', 0)),
                    'reasons': cleaned_alert_info.get('reasons', []),
                    'pose_counts': cleaned_alert_info.get('pose_counts', {}),
                    'zone_penetration': cleaned_alert_info.get('zone_penetration_detected', False),
                    'zone_penetration_zones': cleaned_alert_info.get('zone_penetration_zones', [])
                },
                'behavior_analysis': {
                    'description': pose_descriptions['full_description'],
                    'action_sequence': pose_descriptions['action_sequence'],
                    'suspicious_actions': pose_descriptions['suspicious_actions'],
                    'dominant_pose': pose_descriptions['dominant_pose'],
                    'severity_level': pose_descriptions['severity_level'],
                    'detailed_breakdown': pose_descriptions['detailed_breakdown']
                }
            }
            
            with open(json_filename, 'w') as f:
                json.dump(clip_info, f, indent=2)
            
            video_exists = os.path.exists(video_filename) and os.path.getsize(video_filename) > 0
            json_exists = os.path.exists(json_filename)
            
            if video_exists and json_exists:
                self.alert_clips_saved.append(base_filename)
                print(f"✅ Alert clip saved successfully:")
                print(f"   📹 Video: {video_filename} ({os.path.getsize(video_filename)/1024:.1f} KB)")
                print(f"   📄 JSON:  {json_filename}")
                print(f"   🖼️  Crops: {crop_count} images in {crop_dir}")
                print(f"   ⏱️  Duration: {len(all_frames) / self.fps:.1f}s ({len(all_frames)} frames)")
                print(f"   🎯 Alert at frame: {alert_frame_idx + 1}/{len(all_frames)}")
                print(f"   📝 Behavior: {pose_descriptions['full_description']}")
                
                base_dir = Path(__file__).parent.parent.parent
                video_path = base_dir / video_filename
                
                def upload_in_background():
                    """Background thread for upload"""
                    print(f"🚀 Starting background upload for Track {track_id}: {video_path}")
                    
                    # Create NEW event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        upload_results = loop.run_until_complete(
                            upload_files_with_retry(video_path, max_retries=10)
                        )
                        
                        if upload_results["video"]:
                            print(f"✅ Video uploaded: {upload_results['video']['url']}")
                        else:
                            print(f"❌ Video upload failed")
                        
                        if upload_results["photo"]:
                            print(f"✅ Photo uploaded: {upload_results['photo']['url']}")
                        else:
                            print(f"❌ Photo upload failed")
                    except Exception as e:
                        print(f"❌ Upload error: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        loop.close()
                
                # Start background thread
                upload_thread = threading.Thread(target=upload_in_background, daemon=True)
                upload_thread.start()
                print(f"🔄 Upload started in background thread")

                del self.recording_alerts[track_id]
                return base_filename
            else:
                print(f"❌ Failed to save files for Track {track_id}")
                print(f"   Video exists: {video_exists} ({video_filename})")
                print(f"   JSON exists: {json_exists} ({json_filename})")
                del self.recording_alerts[track_id]
                return None
                
        except Exception as e:
            print(f"❌ Error finalizing clip for Track {track_id}: {e}")
            import traceback
            traceback.print_exc()
            if track_id in self.recording_alerts:
                del self.recording_alerts[track_id]
            return None
    
    def _generate_pose_descriptions(self, alert_info):
        """Generate deskripsi lengkap tentang pose/behavior"""
        pose_counts = alert_info.get('pose_counts', {})
        reasons = alert_info.get('reasons', [])
        grabbed_hand = alert_info.get('grabbed_hand', 'unknown')
        suspicion_score = alert_info.get('suspicion_score', 0)
        zone_penetration = alert_info.get('zone_penetration_detected', False)
        zone_names = alert_info.get('zone_penetration_zones', [])
        
        pose_descriptions = {
            'bending_down': 'membungkuk ke bawah',
            'crouching': 'berjongkok',
            'hiding_under_clothing': 'memasukkan sesuatu ke dalam pakaian',
            'concealing_at_waist': 'menyembunyikan sesuatu di area pinggang',
            'reaching_pocket': 'meraih kantong',
            'hands_near_body': 'tangan dekat dengan tubuh',
            'putting_in_pants_pocket': 'memasukkan sesuatu ke kantong celana',
            'hands_behind_back': 'meletakkan tangan di belakang punggung',
            'squatting_low': 'jongkok rendah',
            'reaching_waist_back': 'meraih area pinggang belakang',
            'zone_pants_pocket_left': 'memasukkan tangan ke kantong celana kiri',
            'zone_pants_pocket_right': 'memasukkan tangan ke kantong celana kanan',
            'zone_jacket_pocket_left': 'memasukkan tangan ke kantong jaket kiri',
            'zone_jacket_pocket_right': 'memasukkan tangan ke kantong jaket kanan',
            'hiding_in_hat': 'memasukkan barang ke topi atau area kepala',
            'hand_on_head': 'tangan berada di area kepala',
            'zone_pants_pocket_left': 'memasukkan tangan ke kantong celana kiri',
            'hiding_in_shoe_left': 'memasukkan barang ke sepatu kiri',
            'hiding_in_shoe_right': 'memasukkan barang ke sepatu kanan',
            'reaching_shoe_area': 'tangan mendekati area sepatu',
        }
        
        sorted_poses = sorted(pose_counts.items(), key=lambda x: x[1], reverse=True)
        
        action_sequence = []
        action_sequence.append(f"1. Mengangkat tangan {grabbed_hand} untuk mengambil barang")

        if zone_penetration and zone_names:
            for zone in zone_names:
                zone_desc = zone.replace('_', ' ').title()
                action_sequence.append(f"2. 🚨 ZONE DETECTION: Tangan masuk ke {zone_desc}")

        if zone_penetration:
            zone_desc_list = [z.replace('_', ' ') for z in zone_names]
            full_description = (
                f"🚨 ZONE PENETRATION DETECTED! Orang terdeteksi mengambil barang dengan tangan {grabbed_hand}, "
                f"kemudian langsung memasukkan tangan ke zona: {', '.join(zone_desc_list)}. "
            )
            
        for i, (pose_key, count) in enumerate(sorted_poses[:3], start=2):
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name.replace('_', ' '))
            action_sequence.append(f"{i}. Terdeteksi {desc} sebanyak {count} kali")
        
        suspicious_actions = []
        for pose_key, count in sorted_poses:
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name.replace('_', ' '))
            suspicious_actions.append({
                'action': desc,
                'count': count,
                'pose_type': pose_name
            })
        
        if sorted_poses:
            dominant_pose_key = sorted_poses[0][0]
            dominant_pose_key_str = str(dominant_pose_key).replace('SuspiciousPose.', '').lower() if hasattr(dominant_pose_key, '__class__') else str(dominant_pose_key).lower()
            dominant_pose = pose_descriptions.get(dominant_pose_key_str, dominant_pose_key_str.replace('_', ' '))
            dominant_count = sorted_poses[0][1]
        else:
            dominant_pose = "tidak teridentifikasi"
            dominant_count = 0
        
        if sorted_poses:
            top_3_poses = []
            for pose_key, _ in sorted_poses[:3]:
                pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
                top_3_poses.append(pose_descriptions.get(pose_name, pose_name.replace('_', ' ')))
            
            full_description = (
                f"Orang terdeteksi mengambil barang dengan tangan {grabbed_hand}, "
                f"kemudian melakukan gerakan mencurigakan: {', '.join(top_3_poses)}. "
                f"Pose dominan adalah '{dominant_pose}' yang terdeteksi {dominant_count} kali."
            )
        else:
            full_description = (
                f"Orang terdeteksi mengambil barang dengan tangan {grabbed_hand} "
                f"dan melakukan gerakan mencurigakan setelahnya."
            )
        
        if suspicion_score >= 85:
            severity = "SANGAT TINGGI - Kemungkinan besar shoplifting"
        elif suspicion_score >= 75:
            severity = "TINGGI - Perilaku sangat mencurigakan"
        elif suspicion_score >= 65:
            severity = "SEDANG - Perilaku cukup mencurigakan"
        else:
            severity = "RENDAH - Perilaku agak mencurigakan"
        
        detailed_breakdown = {
            'initial_action': f"Mengangkat tangan {grabbed_hand} untuk mengambil barang dari rak/shelf",
            'grabbing_confirmed': True,
            'suspicious_movements': [],
            'concealment_method': None,
            'body_position': [],
            'zone_penetration': zone_penetration,  #
            'zone_details': zone_names if zone_penetration else []
        }
        
        for pose_key, count in sorted_poses:
            pose_name = str(pose_key).replace('SuspiciousPose.', '').lower() if hasattr(pose_key, '__class__') else str(pose_key).lower()
            desc = pose_descriptions.get(pose_name, pose_name)
            
            if 'hiding' in pose_name or 'pocket' in pose_name or 'concealing' in pose_name:
                if not detailed_breakdown['concealment_method']:
                    detailed_breakdown['concealment_method'] = desc
            
            if 'bending' in pose_name or 'crouch' in pose_name or 'squat' in pose_name:
                detailed_breakdown['body_position'].append(desc)
            
            detailed_breakdown['suspicious_movements'].append({
                'movement': desc,
                'frequency': count,
                'severity': 'high' if count > 10 else ('medium' if count > 5 else 'low')
            })
        
        if not detailed_breakdown['concealment_method']:
            detailed_breakdown['concealment_method'] = "Metode penyembunyian tidak teridentifikasi dengan jelas"
        
        if not detailed_breakdown['body_position']:
            detailed_breakdown['body_position'] = ["Posisi tubuh normal/berdiri"]
        
        return {
            'full_description': full_description,
            'action_sequence': action_sequence,
            'suspicious_actions': suspicious_actions,
            'dominant_pose': {
                'pose': dominant_pose,
                'count': dominant_count
            },
            'severity_level': severity,
            'detailed_breakdown': detailed_breakdown
        }
    
    def update_recording_alerts(self, original_frame):  
        """
        Update semua recording alerts dengan ORIGINAL frame (clean, no visual)
        """
        to_finalize = []
        
        for track_id, recording in list(self.recording_alerts.items()):
            frames_after = recording['frames_after']
            frames_needed = recording['frames_needed']
            
            if len(frames_after) < frames_needed:
                recording['frames_after'].append(original_frame.copy()) 
                
                if len(frames_after) >= frames_needed:
                    to_finalize.append(track_id)
        
        for track_id in to_finalize:
            self.finalize_alert_clip(track_id)

    def draw_suspicious_poses_overlay(self, frame, suspicious_poses, x1, y1, x2, y2):
            """
            Draw suspicious poses dengan visual yang jelas
            Returns: modified frame, y_offset untuk info tambahan
            """
            if not suspicious_poses or len(suspicious_poses) == 0:
                return frame, y1
            
            y_offset_poses = y1 - 15
            
            # Sort by confidence 
            sorted_poses = sorted(suspicious_poses, key=lambda p: p[1], reverse=True)[:3]
            
            for pose_type, conf, desc in sorted_poses:
                # Shorten description
                short_desc = desc[:40] + "..." if len(desc) > 40 else desc
                pose_text = f"{short_desc} ({conf:.0%})"
                
                # Color based on severity
                if conf >= 0.85:
                    pose_color = (0, 0, 255) 
                    bg_color = (0, 0, 100)
                elif conf >= 0.70:
                    pose_color = (0, 140, 255) 
                    bg_color = (0, 70, 100)
                else:
                    pose_color = (0, 200, 255)
                    bg_color = (0, 100, 100)
                
                # Get text size
                text_size = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background box
                bg_x1 = max(0, x1 - 5)
                bg_y1 = max(0, y_offset_poses - text_size[1] - 8)
                bg_x2 = min(frame.shape[1], x1 + text_size[0] + 15)
                bg_y2 = y_offset_poses + 5
                
                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Draw border
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), pose_color, 2)
                
                # Draw text
                cv2.putText(frame, pose_text, (x1 + 5, y_offset_poses),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                y_offset_poses -= (text_size[1] + 15)
            
            return frame, y_offset_poses

    def detect_walking_state(self, keypoints, track_id, bbox):
        """
        Deteksi apakah orang sedang JALAN atau STATIONARY
        
        Returns:
            (is_walking, movement_speed, direction)
        """
        track = self.person_tracks[track_id]
        
        # GET BBOX CENTER as position proxy
        x1, y1, x2, y2 = bbox
        current_position = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # SAVE POSITION HISTORY
        track['position_history'].append({
            'position': current_position,
            'frame': self.frame_count
        })
        
        # NEED at least 5 frames untuk calculate movement
        if len(track['position_history']) < 5:
            return False, 0.0, None
        
        # CALCULATE MOVEMENT SPEED (average displacement)
        positions = [p['position'] for p in track['position_history']]
        total_distance = 0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            distance = math.sqrt(dx**2 + dy**2)
            total_distance += distance
        
        # Average speed (pixels per frame)
        avg_speed = total_distance / (len(positions) - 1)
        
        # WALKING THRESHOLD
        WALKING_SPEED_MIN = 3.0  
        WALKING_SPEED_MAX = 50.0  
        
        is_walking = WALKING_SPEED_MIN <= avg_speed <= WALKING_SPEED_MAX
        
        # DETECT DIRECTION (optional, for debugging)
        direction = None
        if is_walking and len(positions) >= 3:
            # Compare first vs last position
            dx = positions[-1][0] - positions[0][0]
            dy = positions[-1][1] - positions[0][1]
            
            if abs(dx) > abs(dy):
                direction = 'right' if dx > 0 else 'left'
            else:
                direction = 'backward' if dy > 0 else 'forward'  
        
        # UPDATE TRACKING STATE
        if is_walking:
            track['walking_frames'] += 1
            track['stationary_frames'] = 0
        else:
            track['stationary_frames'] += 1
            if track['stationary_frames'] > 3:  
                track['walking_frames'] = 0  
        
        track['last_movement_speed'] = avg_speed
        track['walking_direction'] = direction
        
        track['is_walking'] = track['walking_frames'] >= 2

        if avg_speed > 8.0: 
            track['walking_frames'] = 2  
            track['is_walking'] = True
            if self.debug_mode:
                print(f"  ⚡ INSTANT WALKING detected (speed: {avg_speed:.1f}px/f)")
        
        return track['is_walking'], avg_speed, direction

    def detect_squatting_posture(self, keypoints):
        """
        Deteksi postur jongkok/squatting berdasarkan knee angles dan body position
        
        Returns:
            (is_squatting, confidence, available_keypoints)
        """
        left_hip = self.get_keypoint(keypoints, 'left_hip')
        right_hip = self.get_keypoint(keypoints, 'right_hip')
        left_knee = self.get_keypoint(keypoints, 'left_knee')
        right_knee = self.get_keypoint(keypoints, 'right_knee')
        left_ankle = self.get_keypoint(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint(keypoints, 'right_ankle')
        
        available_keypoints = []
        
        # Check if we have minimum required keypoints for squatting detection
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return False, 0.0, []
        
        # Calculate knee angles
        left_angle = None
        right_angle = None
        
        if left_ankle:
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            available_keypoints.extend(['left_hip', 'left_knee', 'left_ankle'])
        
        if right_ankle:
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            available_keypoints.extend(['right_hip', 'right_knee', 'right_ankle'])
        
        # Need at least one valid angle
        if not left_angle and not right_angle:
            return False, 0.0, available_keypoints
        
        # Calculate average angle (use only valid angles)
        angles = [a for a in [left_angle, right_angle] if a is not None]
        avg_knee_angle = sum(angles) / len(angles)
                
        is_squatting = False
        confidence = 0.0
        
        if avg_knee_angle < 130:
            is_squatting = True
            
            # Calculate confidence based on angle
            if avg_knee_angle < 90:
                # Deep squat - very high confidence
                confidence = 0.95
            elif avg_knee_angle < 115:
                # Medium squat - high confidence
                confidence = 0.80 + (115 - avg_knee_angle) / 115 * 0.15
            else:
                # Light squat - medium confidence
                confidence = 0.60 + (130 - avg_knee_angle) / 130 * 0.20

            if left_angle and right_angle:
                confidence = min(0.98, confidence * 1.1)
            
            left_hip_knee_dist = abs(left_hip[1] - left_knee[1])
            right_hip_knee_dist = abs(right_hip[1] - right_knee[1])
            avg_hip_knee_dist = (left_hip_knee_dist + right_hip_knee_dist) / 2
            
            if avg_hip_knee_dist < 100: 
                confidence = min(0.98, confidence * 1.15)
        
        return is_squatting, confidence, available_keypoints

    def process_frame(self, frame):
        # TODO: comment Testing purpose: after 15 seconds sleep, send fake alert
        # start_time = time.time()
        # while time.time() - start_time < 15:
        #     asyncio.sleep(0.1)

        # alert_payload = {
        #     "id": str(uuid.uuid4()),
        #     "title": "Shoplifting Alert",
        #     "incident_start": datetime.utcnow().isoformat(),
        #     "is_valid": None,
        #     "video_url": "shoplifting_track13_20251113_141745.mp4",
        #     "notes": "   Reasons: Testing purpose"
        # }

        # return "processed", [alert_payload]
        """Process frame dengan zone visualization + natural position indicator"""
        self.frame_count += 1
        
        results = self.pose_model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=self.inference_size,
            device='cpu'
        )
        
        alert_persons = []
        
        processed = frame.copy()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_data = results[0].keypoints.data.cpu().numpy()
            
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                keypoints = keypoints_data[i]

                is_squatting, squat_conf, _ = self.detect_squatting_posture(keypoints)
                    
                if is_squatting:
                        # VISUAL: Draw ORANGE overlay untuk squatting
                        overlay = processed.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), -1)
                        cv2.addWeighted(overlay, 0.20, processed, 0.80, 0, processed)
                        
                        # Label SQUATTING
                        squat_label = f"SQUATTING ({squat_conf:.0%})"
                        cv2.putText(processed, squat_label, (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                        # SHOW AVAILABLE KEYPOINTS 
                        available_upper = []
                        for kp_name in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                                        'left_wrist', 'right_wrist']:
                            if self.get_keypoint(keypoints, kp_name):
                                available_upper.append(kp_name.split('_')[0][0].upper())  # L/R
                        
                        if available_upper:
                            kp_text = f"Upper: {','.join(available_upper)}"
                        else:
                            kp_text = "Upper: NONE!"  
                        
                        cv2.putText(processed, kp_text, (x1, y2 + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                        track = self.person_tracks[track_id]
                        if track['phase'] in [DetectionPhase.REACHING_SHELF, DetectionPhase.GRABBING]:
                            reach_type = track.get('reach_type', '')
                            if reach_type == 'squat':
                                # HIGHLIGHT: Yellow border for squat + reaching
                                cv2.rectangle(processed, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 5)
                                cv2.putText(processed, "SQUAT + REACHING!", (x1, y2 + 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Check keypoint quality (now relaxed for squatting)
                is_quality_ok, visibility, missing = self.check_keypoint_quality(keypoints)
                    
                if not is_quality_ok and not is_squatting:
                        
                        cv2.rectangle(processed, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        cv2.putText(processed, "OCCLUDED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                        continue

                is_quality_ok, visibility, missing = self.check_keypoint_quality(keypoints)
                
                if not is_quality_ok:
                    if self.debug_mode:
                        print(f"⚠️ Track {track_id}: POOR KEYPOINT QUALITY")
                        print(f"   Visibility: {visibility:.1%}")
                        print(f"   Missing critical: {missing}")
                    
                    # Draw WARNING overlay
                    cv2.rectangle(processed, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(processed, "OCCLUDED - NO DETECTION", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                    
                    continue  

                # STEP 0.5: DETECT WALKING STATE
                is_walking, speed, direction = self.detect_walking_state(keypoints, track_id, box)
                
                track = self.person_tracks[track_id]
                
                # UPDATE PHASE TO WALKING if walking detected
                if is_walking:
                    if track['grab_detected']:
                        
                        if not track['walking_with_item']:
                            track['pre_walking_phase'] = track['phase']
                            track['walking_with_item'] = True
                            
                            if self.debug_mode:
                                print(f"🚶🎒 Track {track_id}: WALKING WITH GRABBED ITEM!")
                                print(f"   Current phase: {track['phase'].value}")
                                print(f"   Speed: {speed:.1f}px/f, Direction: {direction}")
                        
                    elif track['phase'] == DetectionPhase.IDLE:
                       
                        track['phase'] = DetectionPhase.WALKING
                        track['phase_start_frame'] = self.frame_count
                        
                        if self.debug_mode:
                            print(f"🚶 Track {track_id}: IDLE -> WALKING (speed: {speed:.1f}px/f, dir: {direction})")
                
                # SKIP DETECTION if in WALKING state
                if track['phase'] == DetectionPhase.WALKING:
                  
                    if not is_walking and track['stationary_frames'] >= 8: 
                        track['phase'] = DetectionPhase.IDLE
                        track['phase_start_frame'] = self.frame_count
                        
                        if self.debug_mode:
                            print(f"🛑 Track {track_id}: WALKING -> IDLE (stopped)")
                    
                    # DRAW WALKING INDICATOR
                    cv2.rectangle(processed, (x1, y1), (x2, y2), (255, 200, 0), 2)  
                    label = f"ID:{track_id} [WALKING {speed:.1f}px/f]"
                    if direction:
                        label += f" {direction.upper()}"
                    
                    cv2.putText(processed, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    
                    continue

                # HANDLE WALKING WITH GRABBED ITEM 
                if track['walking_with_item'] and is_walking:
                    # Visual indicator: YELLOW border 
                    cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow
                    label = f"ID:{track_id} [WALKING + GRABBED!]"
                    cv2.putText(processed, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show speed
                    cv2.putText(processed, f"Speed: {speed:.1f}px/f", (x1, y1 - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                if track['phase'] in [DetectionPhase.REACHING_SHELF, DetectionPhase.GRABBING]:
                    track['is_walking'] = False
                    track['walking_frames'] = 0

                # HANDLE STOP AFTER WALKING WITH ITEM
                if track['walking_with_item'] and not is_walking and track['stationary_frames'] >= 8:
                    # Orang berhenti setelah jalan bawa barang
                    if self.debug_mode:
                        print(f"🛑 Track {track_id}: STOPPED after walking with item")
                        print(f"   Pre-walking phase: {track['pre_walking_phase'].value if track['pre_walking_phase'] else 'None'}")
                    
                    # Kembali ke phase sebelumnya 
                    if track['pre_walking_phase'] and track['pre_walking_phase'] != DetectionPhase.IDLE:
                        track['phase'] = track['pre_walking_phase']
                        if self.debug_mode:
                            print(f"   → Restored to {track['phase'].value}")
                    else:
                        
                        track['phase'] = DetectionPhase.SUSPICIOUS_MOVEMENT
                        track['phase_start_frame'] = self.frame_count
                        if self.debug_mode:
                            print(f"   → Forced to SUSPICIOUS")
                    
                    # Add bonus suspicious score untuk walking with item
                    track['suspicion_score'] += 25
                    if self.debug_mode:
                        print(f"   → Bonus +25 score (walking with item)")
                    
                    # Clear walking flags
                    track['walking_with_item'] = False
                    track['pre_walking_phase'] = None
                
                # STEP 3.2: CHECK KEYPOINT STABILITY
                is_stable, stability_score, jittery = self.calculate_keypoint_stability(
                    track_id, keypoints
                )
                
                if not is_stable:
                    
                    track = self.person_tracks[track_id]
                    track['unstable_frames'] += 1
                    
                    if self.debug_mode:
                        print(f"⚠️ Track {track_id}: UNSTABLE KEYPOINTS")
                        print(f"   Stability: {stability_score:.1%}")
                        print(f"   Jittery: {jittery}")
                    
                    # Allow beberapa unstable frames 
                    if track['unstable_frames'] > 5:
                        
                        self.reset_track(track_id)
                        
                        cv2.rectangle(processed, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        cv2.putText(processed, "UNSTABLE - NO DETECTION", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                    continue  
                else:   
                    self.person_tracks[track_id]['unstable_frames'] = 0
                
                track = self.person_tracks[track_id]
                track['current_keypoints'] = keypoints
                
                if not track['pocket_zones']:
                    self.initialize_pocket_zones(track_id, keypoints)
                
                is_alert, suspicious_poses, reasons = self.update_phase(
                    track_id, keypoints, self.frame_count, box
                )

                if suspicious_poses and not is_alert:
                    high_conf_poses = [p for p in suspicious_poses if p[1] >= 0.75]
                    if high_conf_poses:
                        overlay = processed.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                        cv2.addWeighted(overlay, 0.15, processed, 0.85, 0, processed)
                
                # DRAW KEYPOINTS
                for kp in keypoints:
                    if kp[2] > 0.6:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(processed, (x, y), 4, (0, 255, 0), -1)
                
                # DRAW SKELETON
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                for start_idx, end_idx in connections:
                    if keypoints[start_idx][2] > 0.6 and keypoints[end_idx][2] > 0.6:
                        start = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(processed, start, end, (0, 255, 0), 2)
                
                # VISUALISASI NATURAL POSITION CHECK
                if track['grab_detected'] and track['grabbed_hand']:
                    grabbed_hand = track['grabbed_hand']
                    shoulder_key = f'{grabbed_hand}_shoulder'
                    elbow_key = f'{grabbed_hand}_elbow'
                    wrist_key = f'{grabbed_hand}_wrist'
                    hip_key = f'{grabbed_hand}_hip'
                    knee_key = f'{grabbed_hand}_knee'
                    
                    shoulder = self.get_keypoint(keypoints, shoulder_key)
                    elbow = self.get_keypoint(keypoints, elbow_key)
                    wrist = self.get_keypoint(keypoints, wrist_key)
                    hip = self.get_keypoint(keypoints, hip_key)
                    knee = self.get_keypoint(keypoints, knee_key)
                    
                    if all([shoulder, elbow, wrist, hip]):
                        # Check natural position 
                        if wrist[1] > hip[1]:  
                            horizontal_dist = abs(wrist[0] - hip[0])
                            
                            if horizontal_dist < 70:  
                                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                                wrist_to_hip = self.distance(wrist, hip)
                                
                                is_straight = elbow_angle and elbow_angle >= 160
                                is_relax = wrist_to_hip and wrist_to_hip < 80 and horizontal_dist < 50
                                
                                # VISUAL INDICATOR: GREEN = NATURAL POSITION
                                if is_straight or is_relax:
                                    # Draw GREEN circle di wrist
                                    cv2.circle(processed, (int(wrist[0]), int(wrist[1])), 15, (0, 255, 0), 3)
                                    
                                    # Label "NATURAL"
                                    cv2.putText(processed, "NATURAL", 
                                            (int(wrist[0]) - 35, int(wrist[1]) - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    # Draw line dari shoulder
                                    cv2.line(processed, 
                                        (int(shoulder[0]), int(shoulder[1])), 
                                        (int(elbow[0]), int(elbow[1])), 
                                        (0, 255, 0), 3)
                                    cv2.line(processed, 
                                        (int(elbow[0]), int(elbow[1])), 
                                        (int(wrist[0]), int(wrist[1])), 
                                        (0, 255, 0), 3)
                                    
                                    if elbow_angle:
                                        angle_text = f"{elbow_angle:.0f}deg"
                                        cv2.putText(processed, angle_text,
                                                (int(elbow[0]) + 10, int(elbow[1]) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                                    if wrist_to_hip:
                                        dist_text = f"{wrist_to_hip:.0f}px"
                                        cv2.putText(processed, dist_text,
                                                (int(wrist[0]) + 10, int(wrist[1]) + 35),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                
                                # VISUAL INDICATOR: YELLOW = BORDERLINE
                                else:
                                    if 50 < horizontal_dist < 70 or (elbow_angle and 150 < elbow_angle < 160):
                                        cv2.circle(processed, (int(wrist[0]), int(wrist[1])), 12, (0, 255, 255), 2)
                                        cv2.putText(processed, "BORDERLINE", 
                                                (int(wrist[0]) - 45, int(wrist[1]) - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                        
                                        # Info debug
                                        if elbow_angle:
                                            cv2.putText(processed, f"{elbow_angle:.0f}deg",
                                                    (int(elbow[0]) + 10, int(elbow[1]) - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # DRAW POCKET ZONES
                if track['grab_detected'] and track['pocket_zones']:
                    self.update_pocket_zones(track_id, keypoints)
                    
                    for zone_name, zone in track['pocket_zones'].items():
                        # CRITICAL FIX: Check if zone exists 
                        if zone is None:
                            continue  
                        
                        if zone.zone_box:  
                            x1z, y1z, x2z, y2z = zone.zone_box
                            
                            frames_in_zone = track['wrist_in_zone_frames'][zone_name]
                            
                            if frames_in_zone > 0:
                                depth = track['max_zone_depth'].get(zone_name, 0)
                                
                                if depth >= 0.5:
                                    color = (0, 0, 255)  
                                    thickness = 4
                                    alpha = 0.5
                                elif depth >= 0.3:
                                    color = (0, 100, 255)  
                                    thickness = 3
                                    alpha = 0.4
                                else:
                                    color = (0, 165, 255) 
                                    thickness = 3
                                    alpha = 0.3
                            else:
                                color = (255, 200, 0)  
                                thickness = 2
                                alpha = 0.15
                            
                            # Draw filled zone
                            overlay = processed.copy()
                            cv2.rectangle(overlay, (x1z, y1z), (x2z, y2z), color, -1)
                            cv2.addWeighted(overlay, alpha, processed, 1 - alpha, 0, processed)
                            
                            # Draw border
                            cv2.rectangle(processed, (x1z, y1z), (x2z, y2z), color, thickness)
                            
                            # Label
                            label = zone_name.replace('_', ' ').upper()
                            if frames_in_zone > 0:
                                depth_pct = track['max_zone_depth'].get(zone_name, 0) * 100
                                label += f" ⚠️ {frames_in_zone}f ({depth_pct:.0f}%)"
                            
                            # Background for label
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                            cv2.rectangle(processed, (x1z, y1z - label_size[1] - 8),
                                        (x1z + label_size[0] + 5, y1z - 2),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(processed, label, (x1z + 2, y1z - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # VISUALISASI BODY ROTATION
                if track['is_rotating'] and track['rotation_frames'] > 3:
                    # Draw overlay transparan ORANGE
                    overlay = processed.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 165, 0), -1)
                    cv2.addWeighted(overlay, 0.2, processed, 0.8, 0, processed)
                    
                    # Text ROTATING
                    rotation_text = f"ROTATING - DETECTION PAUSED"
                    cv2.putText(processed, rotation_text, (x1, y2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
                # ALERT TRIGGERED
                if is_alert and not track['alert_triggered']:
                    alert_info = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': self.frame_count,
                        'track_id': track_id,
                        'phase': track['phase'].value,
                        'grab_frame': track['grab_frame'],
                        'grabbed_hand': track['grabbed_hand'],
                        'suspicion_score': track['suspicion_score'],
                        'suspicious_ratio': track['suspicious_ratio'],
                        'total_frames': track['total_frames_tracked'],
                        'reasons': reasons,
                        'zone_penetration_detected': track['zone_penetration_detected'],
                        'zone_penetration_zones': track['zone_penetration_zones'],
                        'pose_counts': dict(track['pose_counts'])
                    }
                    
                    self.alert_log.append(alert_info)
                    track['alert_triggered'] = True
                    track['last_alert_frame'] = self.frame_count
                    
                    alert_persons.append(track_id)
                    
                    clip_name = self.save_alert_clip(track_id, alert_info, self.frame_count, bbox=box)
                    
                    if self.debug_mode:
                        print(f"\n🚨 SHOPLIFTING ALERT: Track {track_id}")
                        print(f"   Grabbed: Frame {track['grab_frame']} ({track['grabbed_hand']} hand)")
                        print(f"   Score: {track['suspicion_score']:.1f}")
                        print(f"   Reasons: {reasons}")
                        if clip_name:
                            print(f"   Clip: {clip_name}")

                    # TODO: use real data
                    alert_payload = {
                        "id": str(uuid.uuid4()),
                        "title": "Shoplifting Alert",
                        "incident_start": datetime.utcnow().isoformat(),
                        "is_valid": None,
                        "video_url": clip_name,
                        "notes": f"   Reasons: {reasons}"
                    }

                    return "processed", [alert_payload]
                    
                    # Draw RED bounding box untuk alert
                    cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    label = f"SHOPLIFTING! ID:{track_id}"
                    cv2.putText(processed, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    
                    # Draw alert reasons
                    y_offset = y2 + 25
                    for reason in reasons[:2]:
                        cv2.putText(processed, reason, (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        y_offset += 20

                    alert_payload = {
                        "id": str(uuid.uuid4()),
                        "title": "Shoplifting Alert",
                        "incident_start": datetime.utcnow().isoformat(),
                        "is_valid": None,
                        "video_url": clip_name,
                        "notes": f"Reasons: {reasons}"
                    }

                    return "processed", [alert_payload]
                
                # NON-ALERT: Draw phase status
                else:
                    phase = track['phase']
                    score = track['suspicion_score']
                    
                    # Tentukan warna dan label berdasarkan fase
                    if phase == DetectionPhase.IDLE:
                        color = (0, 255, 0)  
                        label = f"ID:{track_id} [IDLE]"
                        thickness = 2
                    elif phase == DetectionPhase.REACHING_SHELF:
                        color = (0, 255, 255)  
                        label = f"ID:{track_id} [REACHING {track['grabbed_hand']}]"
                        thickness = 3
                    elif phase == DetectionPhase.GRABBING:
                        color = (0, 165, 255)  
                        label = f"ID:{track_id} [GRABBED!]"
                        thickness = 3
                    elif phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
                        color = (0, 0, 255)  
                        label = f"ID:{track_id} [SUSPICIOUS {score:.0f}]"
                        thickness = 4
                    elif phase == DetectionPhase.ALERT:
                        color = (0, 0, 255) 
                        label = f"ID:{track_id} [ALERTED]"
                        thickness = 4
                    else:
                        color = (0, 255, 0)  
                        label = f"ID:{track_id}"
                        thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(processed, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(processed, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw suspicious poses overlay (jika ada)
                    if phase in [DetectionPhase.SUSPICIOUS_MOVEMENT, DetectionPhase.GRABBING]:
                        if suspicious_poses and len(suspicious_poses) > 0:
                            processed, pose_y_offset = self.draw_suspicious_poses_overlay(
                                processed, suspicious_poses, x1, y1, x2, y2
                            )
                    
                    # Draw score info untuk SUSPICIOUS phase
                    if phase == DetectionPhase.SUSPICIOUS_MOVEMENT:
                        info_text = f"Score:{score:.0f} Ratio:{track['suspicious_ratio']:.1%}"
                        cv2.putText(processed, info_text, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw grab info (jika sudah grab)
                    if track['grab_detected'] and phase != DetectionPhase.IDLE:
                        frames_since_grab = self.frame_count - track['grab_frame']
                        reach_info = track.get('reach_type', 'unknown')
                        grab_info = f"Grabbed {frames_since_grab}f ago ({track['grabbed_hand']}) [{reach_info}]"
                        cv2.putText(processed, grab_info, (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # GLOBAL ALERT BANNER (jika ada alert)
        if alert_persons:
            text = f"SHOPLIFTING DETECTED - {len(alert_persons)} PERSON(S)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (processed.shape[1] - text_size[0]) // 2
            
            # Background RED
            cv2.rectangle(processed, (text_x - 20, 10), 
                        (text_x + text_size[0] + 20, 60), (0, 0, 255), -1)
            # Text WHITE
            cv2.putText(processed, text, (text_x, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # SUSPICIOUS ACTIVITY STATS (sidebar kanan atas)
        active_suspicious = [(tid, t) for tid, t in self.person_tracks.items() 
                            if t['phase'] in [DetectionPhase.SUSPICIOUS_MOVEMENT, DetectionPhase.GRABBING]
                            and len(t['suspicious_poses']) > 0]
        
        if active_suspicious:
            y_stat = 30
            x_stat = processed.shape[1] - 350
            
            # Background hitam transparan
            cv2.rectangle(processed, (x_stat - 10, 10),
                        (processed.shape[1] - 10, y_stat + len(active_suspicious) * 25 + 10),
                        (0, 0, 0), -1)
            
            # Header
            cv2.putText(processed, "SUSPICIOUS ACTIVITY", (x_stat, y_stat),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_stat += 25
            
            # List top 3 suspicious persons
            for track_id, track in active_suspicious[:3]:
                pose_count = len(track['suspicious_poses'])
                score = track['suspicion_score']
                stat_text = f"ID{track_id}: {pose_count} poses (score: {score:.0f})"
                cv2.putText(processed, stat_text, (x_stat, y_stat),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                y_stat += 20
        
        # RECORDING STATUS 
        if self.recording_alerts:
            recording_text = f"RECORDING: {len(self.recording_alerts)} clip(s)"
            cv2.putText(processed, recording_text, (10, processed.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # SAVE TO BUFFER & UPDATE RECORDINGS
        self.frame_buffer.append(frame.copy())
        self.update_recording_alerts(frame)
        
        return processed, alert_persons
    
    def save_session_log(self):
        """Save detection log"""
        if self.alert_log:
            cleaned_alerts = []
            for alert in self.alert_log:
                cleaned_alert = self._clean_alert_info_for_json(alert)  
                cleaned_alerts.append(cleaned_alert)
            
            log_data = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_frames': self.frame_count,
                    'total_alerts': len(self.alert_log),
                    'method': 'POSE_WITH_GRABBING_DETECTION_v6_PLUS_ZONES',
                    'grab_thresholds': self.GRAB_THRESHOLDS,
                    'suspicious_thresholds': self.SUSPICIOUS_THRESHOLDS,
                    'zone_thresholds': self.ZONE_THRESHOLDS
                },
                'alerts': cleaned_alerts  
            }
            
            filename = f"shoplifting_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"\n✅ Log saved: {filename}")
            return filename
        return None
    



def main():
    
    print("=" * 80)
    print("SHOPLIFTING DETECTION WITH GRABBING PHASE + POCKET ZONES v6+")
    print("5-PHASE DETECTION: REACHING -> GRABBING -> ZONE PENETRATION -> SUSPICIOUS -> ALERT")
    print("=" * 80)
    print("\n📋 DETECTION PHASES:")
    print("  1. IDLE           - Normal behavior")
    print("  2. REACHING SHELF - Tangan meraih")
    print("  3. GRABBING       - Hand closing/menggenggam")
    print("  4. SUSPICIOUS     - Gerakan mencurigakan setelah grab:")
    print("                      • Zone penetration (kantong celana/jaket)")
    print("                      • Memasukkan ke baju")
    print("                      • Membungkuk/jongkok")
    print("                      • Menyembunyikan di topi/kepala")
    print("                      • Pose mencurigakan lain")
    print("  5. ALERT          - Alert triggered!")
    print("\n NEW FEATURES:")
    print("  • POCKET ZONE DETECTION - Dynamic zones for pants & jacket pockets")
    print("  • Multi-level penetration tracking - Depth measurement (0-1)")
    print("  • Automatic zone initialization - Based on body keypoints")
    print("  • Zone-based alert - Instant suspicious on high depth penetration")
    print("  • Visual zone overlay - Real-time zone visualization")
    print("\n ZONE CONFIGURATION:")
    print("  • Pants pockets (left/right) - Bottom area, 30% body width")
    print("  • Jacket pockets (left/right) - Mid-body, 25% body width")
    print("  • Dynamic sizing - Adapts to person height/width")
    print("  • Penetration depth - 0.3+ triggers medium, 0.6+ triggers high severity")
    print("=" * 80)

    print("\n🚀 FPS OPTIMIZATION MODE:")
    print("1. MAX SPEED    (70-90 FPS)  - nano model, 640x480, skip 2 frames")
    print("2. BALANCED     (45-60 FPS)  - nano model, 960x540, skip 1 frame")
    print("3. QUALITY      (30-40 FPS)  - small model, 960x540, no skip")
    print("4. MAX QUALITY  (20-30 FPS)  - medium model, 1280x720, no skip")
    
    fps_mode = input("Choose mode (1-4) [2]: ").strip() or "4"
    
    if fps_mode == "1":
        model_name = "yolo11n-pose.pt"
        frame_resolution = (640, 480)
        inference_size = 320
        skip_frames = 2
        debug_default = False
    elif fps_mode == "3":
        model_name = "yolo11s-pose.pt"
        frame_resolution = (960, 540)
        inference_size = 416
        skip_frames = 1
        debug_default = False
    elif fps_mode == "4":
        model_name = "yolo11m-pose.pt"
        frame_resolution = (1280, 720)
        inference_size = 640
        skip_frames = 1
        debug_default = True
    else:  
        model_name = "yolo11m-pose.pt"
        frame_resolution = (1280, 720)
        inference_size = 640
        skip_frames = 1
        debug_default = True
    
    print(f"\n✅ Selected: {model_name}, {frame_resolution}, skip={skip_frames}")
    print("=" * 80)
    
    debug = input("\nDebug mode? (y/n) [n]: ").lower() == 'y'
    
    print("\n📹 CCTV VIEWING ANGLE:")
    print("1. Normal (horizontal/eye-level)")
    print("2. High-angle (top-down/ceiling CCTV)")
    angle_mode = input("Choose angle (1-2) [1]: ").strip() or "1"
    high_angle_mode = (angle_mode == "2")
    
    if high_angle_mode:
        print("✅ HIGH-ANGLE MODE: Relaxed reaching detection for ceiling CCTV")
    
    threaded_capture = None
    cap = None
    
    try:
        detector = ShopliftingPoseDetectorWithGrab(
            pose_model=model_name,
            debug_mode=debug,
            high_angle_mode=high_angle_mode
        )
        detector.process_every_n_frames = skip_frames
        detector.inference_size = inference_size
    except Exception as e:
        print(f"\nInitialization failed: {e}")
        return
    
    print("\nVideo source:")
    print("1. Webcam")
    print("2. Video file")
    print("3. RTSP/IP Camera")
    choice = input("Choose (1-3) [1]: ").strip() or "1"
    
    if choice == "2":
        path = input("Video path: ").strip()
        if not os.path.exists(path):
            print("File not found")
            return
        cap = cv2.VideoCapture(path)
        source = f"Video: {os.path.basename(path)}"
    elif choice == "3":
        rtsp_url = input("RTSP URL: ").strip()
        if not rtsp_url:
            print("RTSP URL required")
            return
        
        threaded_capture = ThreadedRTSPCapture(
            rtsp_url=rtsp_url,
            buffer_size=1,
            name="ShopliftingCam"
        )
        
        try:
            threaded_capture.start()
            time.sleep(2)  
            
            stats = threaded_capture.get_stats()
            if not stats['is_alive']:
                print(f"❌ Cannot start RTSP capture")
                return
            
            source = f"RTSP (Threaded): {rtsp_url}"
            
        except Exception as e:
            print(f"❌ Failed to start capture: {e}")
            return
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source = "Webcam"

    if threaded_capture is None:
        if not cap.isOpened():
            print(f"❌ Cannot open {source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            detector.fps = fps
        print(f"✅ {source} ready | FPS: {detector.fps}")
    else:
        detector.fps = 25
        print(f"✅ {source} ready | FPS: {detector.fps} (RTSP default)")
    
    print(f"✅ {source} ready | FPS: {detector.fps}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save log")
    print("  'd' - Toggle debug overlay")
    print("  'r' - Reset all tracks")
    print("=" * 80)
    
    total_alerts = 0
    start_time = time.time()
    frame_times = deque(maxlen=30)
    show_debug_info = True
  
    try:
        while True:
            t_start = time.time()
            

            if threaded_capture:
                ret, frame = threaded_capture.read()
                
                if not ret or frame is None:
                    stats = threaded_capture.get_stats()
                    if stats['time_since_last_frame'] > 5.0:
                        print("⚠️  No frames received for 5s, stream may be down")
                    time.sleep(0.01)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame = cv2.resize(frame, frame_resolution)
            
            processed, alerts = detector.process_frame(frame)
            
            if alerts:
                total_alerts += len(alerts)
            
            frame_times.append(time.time() - t_start)
            current_fps = 1.0 / np.mean(frame_times) if frame_times else 0
            
            cv2.putText(processed, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if total_alerts > 0:
                cv2.putText(processed, f"TOTAL ALERTS: {total_alerts}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if threaded_capture and show_debug_info:
                stats = threaded_capture.get_stats()
                stats_text = f"RTSP: Dropped: {stats['frames_dropped']} ({stats['drop_rate']*100:.1f}%)"
                cv2.putText(processed, stats_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if show_debug_info:
                y_pos = 100
                overlay = processed.copy()
                
                cv2.rectangle(overlay, (5, y_pos - 20), (450, y_pos + 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, processed, 0.4, 0, processed)
                
                cv2.putText(processed, "=== PHASE STATUS ===", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                active_tracks = [(tid, t) for tid, t in detector.person_tracks.items() 
                               if t['total_frames_tracked'] > 0 or t['phase'] != DetectionPhase.IDLE]
                
                if not active_tracks:
                    cv2.putText(processed, "No active tracks", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    for track_id, track_data in active_tracks[:5]:
                        phase_color = {
                            DetectionPhase.IDLE: (0, 255, 0),
                            DetectionPhase.REACHING_SHELF: (0, 255, 255),
                            DetectionPhase.GRABBING: (0, 165, 255),
                            DetectionPhase.SUSPICIOUS_MOVEMENT: (0, 0, 255),
                            DetectionPhase.ALERT: (0, 0, 255)
                        }.get(track_data['phase'], (255, 255, 255))
                        
                        phase_text = f"ID{track_id}: {track_data['phase'].value.upper()}"
                        cv2.putText(processed, phase_text, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
                        y_pos += 20
                        
                        if track_data['phase'] == DetectionPhase.REACHING_SHELF:
                            info = f"  Hand: {track_data['grabbed_hand']} ({track_data['hand_extended_frames']}f)"
                            cv2.putText(processed, info, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                            y_pos += 18
                        elif track_data['grab_detected']:
                            frames_since_grab = detector.frame_count - track_data['grab_frame']
                            info = f"  Grabbed: {frames_since_grab}f ago | Score: {track_data['suspicion_score']:.0f}"
                            cv2.putText(processed, info, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, phase_color, 1)
                            y_pos += 18
            
            cv2.imshow("Shoplifting Detection", processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detector.save_session_log()
            elif key == ord('d'):
                show_debug_info = not show_debug_info
                print(f"Debug overlay: {'ON' if show_debug_info else 'OFF'}")
            elif key == ord('r'):
                detector.person_tracks.clear()
                print("All tracks reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if detector.recording_alerts:
            print(f"\nFinalizing {len(detector.recording_alerts)} pending clip(s)...")
            for track_id in list(detector.recording_alerts.keys()):
                detector.finalize_alert_clip(track_id)

        if threaded_capture is not None:
            threaded_capture.stop()
        elif cap is not None:
            cap.release()
        
        cv2.destroyAllWindows()
        
        runtime = time.time() - start_time
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Runtime: {runtime:.1f}s")
        print(f"Frames Processed: {detector.frame_count}")
        print(f"Total Alerts: {total_alerts}")
        print(f"Clips Saved: {len(detector.alert_clips_saved)}")
        print(f"Average FPS: {detector.frame_count / runtime:.1f}")
        if runtime > 0:
            print(f"Alert Rate: {(total_alerts / (runtime / 60)):.2f} alerts/minute")
        
        if detector.alert_clips_saved:
            print("\nSaved Alert Clips:")
            for i, clip in enumerate(detector.alert_clips_saved, 1):
                print(f"   {i}. {clip}.mp4 + {clip}.json")
        
        print("=" * 80)
        
        detector.save_session_log()
        print("\nSession completed successfully")


if __name__ == "__main__":
    main()

async def wait_for_file_ready(file_path: Path, timeout: int = 60, check_interval: float = 10):
    """
    Wait for a file to exist and be fully written (stable size).
    
    Args:
        file_path: Path to the file
        timeout: Maximum seconds to wait
        check_interval: Seconds between checks
    
    Returns:
        True if file is ready, False if timeout
    """
    start_time = time.time()
    last_size = -1
    
    while time.time() - start_time < timeout:
        if file_path.exists():
            current_size = file_path.stat().st_size
            
            # Check if file size is stable (file finished writing)
            if current_size > 0 and current_size == last_size:
                print(f"✅ File ready: {file_path.name} ({current_size} bytes)")
                return True
            
            last_size = current_size
            print(f"⏳ Waiting for file to stabilize: {file_path.name} ({current_size} bytes)")
        else:
            print(f"⏳ Waiting for file to exist: {file_path.name}")
        
        await asyncio.sleep(check_interval)
    
    print(f"❌ Timeout waiting for file: {file_path.name}")
    return False

async def upload_files_with_retry(video_path: Path, max_retries: int = 10):
    """
    Upload video and photo with retry logic and file readiness check.
    
    Args:
        video_path: Path to the video file
        max_retries: Maximum upload attempts per file
    
    Returns:
        dict with video and photo upload results
    """
    results = {"video": None, "photo": None}
    
    # Wait for video file to be ready
    print(f"🔍 Checking video file: {video_path}")
    if await wait_for_file_ready(video_path, timeout=60):
        # Try uploading video with retries
        for attempt in range(max_retries):
            try:
                print(f"📤 Uploading video (attempt {attempt + 1}/{max_retries})...")
                upload_res = upload_video_to_supabase(str(video_path))
                
                if upload_res:
                    print(f"✅ Video upload success: {upload_res}")
                    results["video"] = upload_res
                    break
                else:
                    print(f"⚠️ Video upload returned None")
                    
            except Exception as e:
                print(f"❌ Video upload error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(60)  # Wait before retry
    else:
        print(f"❌ Video file not ready: {video_path}")
    
    # Wait for photo file to be ready
    photo_path = video_path.with_name(video_path.stem + "_crops") / "ALERT_crop.jpg"
    print(f"🔍 Checking photo file: {photo_path}")
    
    if await wait_for_file_ready(photo_path, timeout=60):
        # Try uploading photo with retries
        for attempt in range(max_retries):
            try:
                print(f"📤 Uploading photo (attempt {attempt + 1}/{max_retries})...")
                upload_res = upload_photo_to_supabase(str(photo_path))
                
                if upload_res:
                    print(f"✅ Photo upload success: {upload_res}")
                    results["photo"] = upload_res
                    break
                else:
                    print(f"⚠️ Photo upload returned None")
                    
            except Exception as e:
                print(f"❌ Photo upload error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(60)  # Wait before retry
    else:
        print(f"❌ Photo file not ready: {photo_path}")
    
    return results

