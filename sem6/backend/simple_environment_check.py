import cv2
import numpy as np
from typing import Dict, Any

class SimpleEnvironmentChecker:
    """
    A lightweight environment checker that performs basic analysis of test conditions:
    - Lighting conditions (brightness analysis)
    - Face detection (using OpenCV's Haar cascades)
    - Basic device detection (contour analysis)
    """
    
    def __init__(self):
        # Lighting thresholds
        self.min_brightness = 40
        self.max_brightness = 220
        
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def check_environment(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze the test environment from an image"""
        try:
            # Convert image bytes to numpy array
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Failed to decode image"}
            
            # Run all checks
            lighting = self._check_lighting(img)
            faces = self._check_faces(img)
            devices = self._check_devices(img)
            
            # Determine overall status
            passed = all([
                lighting["passed"],
                faces["passed"],
                devices["passed"]
            ])
            
            return {
                "passed": passed,
                "checks": {
                    "lighting": lighting,
                    "face": faces,
                    "devices": devices
                }
            }
            
        except Exception as e:
            return {"error": f"Environment check failed: {str(e)}"}
    
    def _check_lighting(self, img: np.ndarray) -> Dict[str, Any]:
        """Check if lighting conditions are adequate"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        passed = self.min_brightness <= brightness <= self.max_brightness
        status = "good"
        
        if brightness < self.min_brightness:
            status = "too dark"
        elif brightness > self.max_brightness:
            status = "too bright"
            
        return {
            "passed": passed,
            "brightness": float(brightness),
            "status": status,
            "message": f"Lighting is {status} ({brightness:.1f})"
        }
    
    def _check_faces(self, img: np.ndarray) -> Dict[str, Any]:
        """Check for single face in frame"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        face_count = len(faces)
        passed = face_count == 1
        
        return {
            "passed": passed,
            "face_count": face_count,
            "message": "Single face detected" if passed else 
                      f"Found {face_count} faces (expected 1)"
        }
    
    def _check_devices(self, img: np.ndarray) -> Dict[str, Any]:
        """Basic check for potential electronic devices"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count potential devices (rectangular contours of certain size)
        device_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:  # Ignore small contours
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            
            # Rectangles have 4 corners
            if len(approx) == 4:
                device_count += 1
                
        passed = device_count == 0
        
        return {
            "passed": passed,
            "device_count": device_count,
            "message": "No devices detected" if passed else
                      f"Found {device_count} potential devices"
        }