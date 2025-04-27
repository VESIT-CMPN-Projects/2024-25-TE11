import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import io
from PIL import Image

class FaceVerifier:
    def __init__(self):
        # Initialize the face analysis app with buffalo_l model
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Set similarity threshold - adjust based on testing
        self.threshold = 0.50  # Lower is stricter, higher is more lenient
        
    def get_embedding(self, img_data):
        """Extract face embedding from image data"""
        # Convert bytes to image if needed
        if isinstance(img_data, bytes):
            img = self._bytes_to_image(img_data)
        else:
            img = img_data
            
        # Detect faces
        faces = self.face_app.get(img)
        
        if not faces:
            raise ValueError("No face detected in the image")
        
        # Use the largest face if multiple detected
        if len(faces) > 1:
            areas = [face.bbox[2] * face.bbox[3] for face in faces]
            largest_face = faces[np.argmax(areas)]
        else:
            largest_face = faces[0]
        
        # Get embedding
        embedding = largest_face.embedding
        
        return embedding
    
    def verify_faces(self, image1_data, image2_data):
        """Compare two faces and return if they match"""
        try:
            # Get embeddings
            embedding1 = self.get_embedding(image1_data)
            embedding2 = self.get_embedding(image2_data)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            
            # Check if similarity exceeds threshold
            match = similarity >= self.threshold
            
            return {
                "match": bool(match),
                "similarity_score": float(similarity),
                "threshold": float(self.threshold)
            }
        except Exception as e:
            return {
                "match": False,
                "error": str(e),
                "similarity_score": 0.0
            }
    
    def _cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def _bytes_to_image(self, img_bytes):
        """Convert image bytes to CV2 image"""
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def preprocess_image(self, img_bytes):
        """Preprocess image for better face detection"""
        # Convert bytes to image
        img = self._bytes_to_image(img_bytes)
        
        # Perform basic preprocessing
        # Resize if too large
        h, w = img.shape[:2]
        max_dimension = 1200
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        return img