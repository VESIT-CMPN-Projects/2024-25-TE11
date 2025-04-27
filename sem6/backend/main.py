from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from face_verification import FaceVerifier
import io
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
from io import BytesIO
from insightface.app import FaceAnalysis  # Common import for InsightFace
from simple_environment_check import SimpleEnvironmentChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Face Verification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face verifier and environment checker
face_verifier = FaceVerifier()
environment_checker = SimpleEnvironmentChecker()

@app.get("/")
async def root():
    return {"message": "Face Verification API is running"}

@app.post("/compare_faces/")
async def compare_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    Compare two face images and determine if they are the same person.
    Returns match status and similarity score.
    """
    logger.info(f"Received face comparison request: {image1.filename} vs {image2.filename}")
    
    try:
        # Read image data
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        # Log image sizes
        logger.info(f"Image 1 size: {len(image1_data)} bytes")
        logger.info(f"Image 2 size: {len(image2_data)} bytes")
        
        # Preprocess images
        img1 = face_verifier.preprocess_image(image1_data)
        img2 = face_verifier.preprocess_image(image2_data)
        
        # Verify faces
        result = face_verifier.verify_faces(img1, img2)
        
        # Log the result
        logger.info(f"Face comparison result: {result}")
        
        if "error" in result and not result.get("match", False):
            # Return error but with 200 status code so frontend can handle it
            return {
                "match": False,
                "detail": result["error"],
                "similarity_score": result.get("similarity_score", 0)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in face comparison: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing images: {str(e)}"
        )
    
@app.post("/check_environment/")
async def check_environment(image: UploadFile = File(...)):
    """
    Analyze an image to verify if the testing environment meets requirements.
    
    Checks for:
    - Adequate lighting
    - Single person present
    - No additional devices
    
    Returns detailed analysis of the environment with pass/fail status.
    """
    try:
        # Read the uploaded image file
        image_data = await image.read()
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        # Process the environment check
        result = environment_checker.check_environment(image_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in environment check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process environment check: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)