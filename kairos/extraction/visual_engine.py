"""
Visual Engine (Module B)
Extracts 8 visual features measuring WHAT their face is communicating
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class VisualEngine:
    """
    Module B:   Visual Feature Extraction (8 features)
    Indices [8-15] in the biomarker vector
    """
    
    LANDMARKS = {
        'left_eye_top': 159, 'left_eye_bottom':  145,
        'left_eye_inner':  133, 'left_eye_outer': 33,
        'right_eye_top': 386, 'right_eye_bottom': 374,
        'right_eye_inner': 362, 'right_eye_outer': 263,
        'left_brow_inner': 107, 'left_brow_outer': 70,
        'right_brow_inner': 336, 'right_brow_outer': 300,
        'mouth_left': 61, 'mouth_right': 291,
        'mouth_top': 13, 'mouth_bottom': 14,
        'nose_tip':  4, 'chin': 152, 'forehead':  10,
        'left_cheek': 234, 'right_cheek':  454,
        'left_iris': 468, 'right_iris': 473,
    }
    
    def __init__(self):
        self.face_mesh = None
        self.mp_face_mesh = None
        self._initialized = False
        self._init_error = None
        self._face_cascade = None
    
    def _initialize_mediapipe(self):
        """Try to initialize MediaPipe FaceMesh"""
        if self._initialized:
            return self. face_mesh is not None
        
        self._initialized = True
        
        try:
            print("     [VisualEngine] Attempting MediaPipe initialization...")
            import mediapipe as mp
            
            # Try standard solutions API
            if hasattr(mp, 'solutions'):
                solutions = mp.solutions
                if hasattr(solutions, 'face_mesh'):
                    self. mp_face_mesh = solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    print("     [VisualEngine] ✅ MediaPipe initialized successfully")
                    return True
            
            print("     [VisualEngine] MediaPipe solutions. face_mesh not available")
            return False
            
        except Exception as e:
            print(f"     [VisualEngine] MediaPipe initialization failed: {e}")
            return False
    
    def _initialize_opencv_cascade(self):
        """Initialize OpenCV Haar Cascade for face detection"""
        if self._face_cascade is not None:
            return True
        
        try:
            import cv2
            
            # Try multiple paths for the cascade file
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv/haarcascades/haarcascade_frontalface_default. xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    self._face_cascade = cv2.CascadeClassifier(path)
                    if not self._face_cascade. empty():
                        print(f"     [VisualEngine] ✅ OpenCV cascade loaded from:  {path}")
                        return True
            
            # Try downloading if not found
            print("     [VisualEngine] Cascade file not found locally, trying to download...")
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            cascade_local = "/tmp/haarcascade_frontalface_default.xml"
            
            import urllib.request
            urllib.request.urlretrieve(cascade_url, cascade_local)
            
            self._face_cascade = cv2.CascadeClassifier(cascade_local)
            if not self._face_cascade.empty():
                print("     [VisualEngine] ✅ OpenCV cascade downloaded and loaded")
                return True
            
            print("     [VisualEngine] ❌ Could not load OpenCV cascade")
            return False
            
        except Exception as e:
            print(f"     [VisualEngine] OpenCV cascade initialization failed: {e}")
            return False
    
    def extract(self, video_array: np.ndarray) -> np.ndarray:
        """Extract all 8 visual features from video frames."""
        features = np.zeros(8, dtype=np.float32)
        
        if video_array is None or len(video_array) == 0:
            print("     [VisualEngine] No video frames provided")
            return features
        
        print(f"     [VisualEngine] Processing {len(video_array)} frames, shape: {video_array.shape}")
        
        # Try MediaPipe first
        if self._initialize_mediapipe():
            features = self._extract_with_mediapipe(video_array)
            if np.any(features != 0):
                return features
            print("     [VisualEngine] MediaPipe extraction returned zeros, trying fallback...")
        
        # Fallback to OpenCV-based analysis
        features = self._extract_with_opencv(video_array)
        
        return features
    
    def _extract_with_mediapipe(self, video_array: np.ndarray) -> np.ndarray:
        """Extract features using MediaPipe FaceMesh."""
        features = np.zeros(8, dtype=np.float32)
        
        try:
            all_landmarks = []
            
            for i, frame in enumerate(video_array):
                landmarks, confidence = self._extract_frame_landmarks(frame)
                if landmarks is not None:
                    all_landmarks.append(landmarks)
            
            print(f"     [VisualEngine] MediaPipe detected faces in {len(all_landmarks)}/{len(video_array)} frames")
            
            if len(all_landmarks) < 3:
                return features
            
            landmarks_array = np.array(all_landmarks)
            
            features[0] = self._compute_masking_score(landmarks_array)
            features[1] = self._compute_brow_tension(landmarks_array)
            features[2] = self._compute_gaze_aversion(landmarks_array)
            features[3] = self._compute_facial_dynamism(landmarks_array)
            features[4] = self._compute_stare_duration(landmarks_array)
            features[5] = self._compute_blink_rate(landmarks_array, len(video_array))
            features[6] = self._compute_head_nodding(landmarks_array)
            features[7] = self._compute_head_tilt(landmarks_array)
            
            print(f"     [VisualEngine] MediaPipe features:  {features}")
            
        except Exception as e:
            print(f"     [VisualEngine] MediaPipe extraction error: {e}")
        
        return features
    
    def _extract_with_opencv(self, video_array: np.ndarray) -> np.ndarray:
        """Extract features using OpenCV (fallback method)."""
        features = np. zeros(8, dtype=np. float32)
        
        try:
            import cv2
            
            print("     [VisualEngine] Running OpenCV-based analysis...")
            
            # Initialize cascade
            cascade_available = self._initialize_opencv_cascade()
            
            face_positions = []
            face_sizes = []
            brightness_values = []
            motion_values = []
            edge_densities = []
            
            prev_gray = None
            
            for i, frame in enumerate(video_array):
                # Ensure correct format
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame. astype(np.uint8)
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else: 
                    gray = frame
                
                # Try face detection with cascade
                face_detected = False
                if cascade_available and self._face_cascade is not None: 
                    try:
                        faces = self._face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.1, 
                            minNeighbors=3,
                            minSize=(30, 30)
                        )
                        
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_center = (x + w/2, y + h/2)
                            face_positions.append(face_center)
                            face_sizes.append(w * h)
                            
                            # Get face region for analysis
                            face_region = gray[y:y+h, x:x+w]
                            brightness_values.append(np.mean(face_region))
                            
                            # Edge density in face region (activity indicator)
                            edges = cv2.Canny(face_region, 50, 150)
                            edge_densities.append(np.mean(edges) / 255.0)
                            
                            face_detected = True
                    except Exception as e:
                        pass
                
                # If no face detected, use center of frame as estimate
                if not face_detected: 
                    h, w = gray.shape
                    face_positions.append((w/2, h/2))
                    brightness_values.append(np.mean(gray))
                    edges = cv2.Canny(gray, 50, 150)
                    edge_densities. append(np.mean(edges) / 255.0)
                
                # Calculate motion between frames
                if prev_gray is not None:
                    # Optical flow or simple frame difference
                    diff = cv2.absdiff(prev_gray, gray)
                    motion_values.append(np.mean(diff) / 255.0)
                
                prev_gray = gray. copy()
            
            # Compute features from collected data
            if len(face_positions) > 1:
                positions = np.array(face_positions)
                
                # Feature 3: Facial Dynamism - movement variance
                movement_std = np.std(positions, axis=0)
                features[3] = float(np.clip(np.mean(movement_std) / 30.0, 0, 1))
                print(f"     [VisualEngine] Facial Dynamism:  {features[3]:.3f}")
                
                # Feature 6: Head Nodding - vertical oscillations
                y_positions = positions[:, 1]
                y_centered = y_positions - np.mean(y_positions)
                sign_changes = np.sum(np.abs(np.diff(np.sign(y_centered))) > 0)
                features[6] = float(np.clip(sign_changes / len(positions) * 2, 0, 1))
                print(f"     [VisualEngine] Head Nodding:  {features[6]:.3f}")
                
                # Feature 7: Head Tilt - horizontal variance
                x_variance = np.var(positions[:, 0])
                features[7] = float(np.clip(x_variance / 500.0, 0, 1))
                print(f"     [VisualEngine] Head Tilt: {features[7]:.3f}")
            
            # Feature based on motion
            if motion_values:
                avg_motion = np.mean(motion_values)
                motion_var = np.var(motion_values)
                
                # Feature 4: Stare Duration - inverse of motion (low motion = staring)
                features[4] = float(np. clip(1.0 - avg_motion * 10, 0, 1))
                print(f"     [VisualEngine] Stare Duration:  {features[4]:.3f}")
            
            # Feature based on brightness changes (proxy for blinks)
            if brightness_values and len(brightness_values) > 3:
                brightness = np.array(brightness_values)
                brightness_diffs = np.abs(np.diff(brightness))
                
                # Count significant dips (potential blinks)
                threshold = np.std(brightness_diffs) * 1.5
                potential_blinks = np.sum(brightness_diffs > threshold)
                
                # Feature 5: Blink Rate
                fps = 5  # Assumed FPS
                duration_minutes = len(video_array) / fps / 60.0
                if duration_minutes > 0:
                    blinks_per_minute = potential_blinks / duration_minutes
                    features[5] = float(np.clip(blinks_per_minute / 30.0, 0, 1))
                    print(f"     [VisualEngine] Blink Rate: {features[5]:.3f} ({blinks_per_minute:.1f}/min)")
            
            # Feature based on edge density (facial expression complexity)
            if edge_densities: 
                edge_var = np.var(edge_densities)
                avg_edges = np.mean(edge_densities)
                
                # Feature 0: Masking Score - low edge variance might indicate fake/held expression
                features[0] = float(np.clip(1.0 - edge_var * 50, 0, 1)) * 0.5
                print(f"     [VisualEngine] Masking Score (est): {features[0]:.3f}")
                
                # Feature 1: Brow Tension - higher edges in upper face
                features[1] = float(np. clip(avg_edges * 2, 0, 1))
                print(f"     [VisualEngine] Brow Tension (est): {features[1]:.3f}")
            
            # Feature 2: Gaze Aversion - estimate from face position variance
            if len(face_positions) > 1:
                positions = np.array(face_positions)
                position_var = np.var(positions, axis=0)
                # High variance in face position might indicate looking away
                features[2] = float(np.clip(np. mean(position_var) / 200.0, 0, 1))
                print(f"     [VisualEngine] Gaze Aversion (est): {features[2]:.3f}")
            
            print(f"     [VisualEngine] OpenCV features: {features}")
            
        except Exception as e: 
            print(f"     [VisualEngine] OpenCV extraction error: {e}")
            import traceback
            traceback.print_exc()
        
        return features
    
    def _extract_frame_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]: 
        """Extract facial landmarks from a single frame using MediaPipe."""
        try:
            import cv2
            
            if frame. dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                frame_rgb = frame
            
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                in_bounds = np.all((landmarks[: , : 2] >= 0) & (landmarks[:, :2] <= 1))
                confidence = 0.9 if in_bounds else 0.5
                return landmarks, confidence
            
            return None, 0.0
            
        except Exception: 
            return None, 0.0
    
    # MediaPipe-based feature computation methods
    def _compute_masking_score(self, landmarks:  np.ndarray) -> float:
        try:
            mouth_widths = []
            eye_heights = []
            
            for frame_lm in landmarks: 
                mouth_width = np.linalg.norm(
                    frame_lm[self. LANDMARKS['mouth_left']][: 2] - 
                    frame_lm[self.LANDMARKS['mouth_right']][:2]
                )
                
                left_eye_h = abs(frame_lm[self. LANDMARKS['left_eye_top']][1] - frame_lm[self.LANDMARKS['left_eye_bottom']][1])
                right_eye_h = abs(frame_lm[self.LANDMARKS['right_eye_top']][1] - frame_lm[self. LANDMARKS['right_eye_bottom']][1])
                eye_height = (left_eye_h + right_eye_h) / 2
                
                mouth_widths. append(mouth_width)
                eye_heights.append(eye_height)
            
            if np.mean(eye_heights) < 0.001:
                return 0.5
            
            ratio = np.mean(mouth_widths) / (np.mean(eye_heights) + 1e-8)
            return float(np.clip(ratio / 10, 0, 1))
        except: 
            return 0.0
    
    def _compute_brow_tension(self, landmarks: np.ndarray) -> float:
        try:
            distances = []
            for frame_lm in landmarks:
                dist = np.linalg.norm(
                    frame_lm[self.LANDMARKS['left_brow_inner']][:2] - 
                    frame_lm[self.LANDMARKS['right_brow_inner']][:2]
                )
                distances.append(dist)
            tension = 1.0 - np.clip(np.mean(distances) / 0.15, 0, 1)
            return float(tension)
        except:
            return 0.0
    
    def _compute_gaze_aversion(self, landmarks: np. ndarray) -> float:
        try:
            aversion_count = 0
            for frame_lm in landmarks:
                if len(frame_lm) > 473: 
                    left_eye_center = (frame_lm[self. LANDMARKS['left_eye_inner']][:2] + frame_lm[self.LANDMARKS['left_eye_outer']][:2]) / 2
                    left_iris = frame_lm[468][: 2]
                    left_eye_width = np.linalg.norm(frame_lm[self.LANDMARKS['left_eye_inner']][:2] - frame_lm[self.LANDMARKS['left_eye_outer']][:2])
                    if left_eye_width > 0:
                        deviation = np.linalg.norm(left_iris - left_eye_center) / left_eye_width
                        if deviation > 0.15:
                            aversion_count += 1
            return float(aversion_count / len(landmarks)) if len(landmarks) > 0 else 0.0
        except:
            return 0.0
    
    def _compute_facial_dynamism(self, landmarks:  np.ndarray) -> float:
        try:
            nose_positions = np.array([frame_lm[self.LANDMARKS['nose_tip']][:2] for frame_lm in landmarks])
            if len(nose_positions) < 2:
                return 0.0
            dynamism = np.sqrt(np.var(nose_positions[: , 0]) + np.var(nose_positions[:, 1]))
            return float(np.clip(dynamism / 0.05, 0, 1))
        except:
            return 0.0
    
    def _compute_stare_duration(self, landmarks: np.ndarray) -> float:
        try: 
            ear_values = [(self._eye_aspect_ratio(lm, 'left') + self._eye_aspect_ratio(lm, 'right')) / 2 for lm in landmarks]
            if len(ear_values) < 2:
                return 0.0
            ear_changes = np.abs(np.diff(ear_values))
            max_stare = current_stare = 0
            for change in ear_changes:
                if change < 0.01:
                    current_stare += 1
                    max_stare = max(max_stare, current_stare)
                else:
                    current_stare = 0
            return float(np.clip(max_stare / len(landmarks), 0, 1))
        except:
            return 0.0
    
    def _compute_blink_rate(self, landmarks: np.ndarray, total_frames: int) -> float:
        try:
            ear_values = [(self._eye_aspect_ratio(lm, 'left') + self._eye_aspect_ratio(lm, 'right')) / 2 for lm in landmarks]
            blink_count = 0
            was_blink = False
            for ear in ear_values:
                if ear < 0.2:
                    if not was_blink:
                        blink_count += 1
                        was_blink = True
                else:
                    was_blink = False
            duration_minutes = total_frames / 5 / 60
            if duration_minutes <= 0:
                return 0.0
            return float(np.clip((blink_count / duration_minutes) / 40, 0, 1))
        except:
            return 0.0
    
    def _compute_head_nodding(self, landmarks: np.ndarray) -> float:
        try: 
            nose_y = np.array([lm[self.LANDMARKS['nose_tip']][1] for lm in landmarks])
            if len(nose_y) < 4:
                return 0.0
            nose_y = nose_y - np.mean(nose_y)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(nose_y))) > 0)
            return float(np.clip(zero_crossings / len(nose_y), 0, 1))
        except:
            return 0.0
    
    def _compute_head_tilt(self, landmarks: np.ndarray) -> float:
        try:
            roll_angles = []
            for lm in landmarks:
                left_eye = lm[self.LANDMARKS['left_eye_outer']][:2]
                right_eye = lm[self. LANDMARKS['right_eye_outer']][:2]
                roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                roll_angles.append(roll)
            if len(roll_angles) < 2:
                return 0.0
            return float(np.clip(np. var(roll_angles) / 0.05, 0, 1))
        except:
            return 0.0
    
    def _eye_aspect_ratio(self, landmarks: np.ndarray, eye:  str) -> float:
        try:
            if eye == 'left':
                top, bottom = self.LANDMARKS['left_eye_top'], self.LANDMARKS['left_eye_bottom']
                inner, outer = self.LANDMARKS['left_eye_inner'], self. LANDMARKS['left_eye_outer']
            else:
                top, bottom = self.LANDMARKS['right_eye_top'], self.LANDMARKS['right_eye_bottom']
                inner, outer = self.LANDMARKS['right_eye_inner'], self. LANDMARKS['right_eye_outer']
            
            vertical = np.linalg.norm(landmarks[top][:2] - landmarks[bottom][:2])
            horizontal = np.linalg.norm(landmarks[inner][:2] - landmarks[outer][:2])
            return vertical / (horizontal + 1e-8)
        except:
            return 0.3