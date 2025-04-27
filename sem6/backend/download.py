from insightface.app import FaceAnalysis

# This will download the buffalo_l model if not already present
print("Downloading and preparing the InsightFace buffalo_l model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Model downloaded and prepared successfully!")