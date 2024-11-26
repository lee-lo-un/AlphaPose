from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.image_analysis import analyze_image
from typing import Optional, Dict, List, Union

router = APIRouter()

class ImageData(BaseModel):
    image: str
    text: Optional[str] = None
    top5_predictions: Optional[List[Dict[str, Union[str, float]]]] = None

@router.post("/")
async def analyze_image_endpoint(data: ImageData):
    try:
        print("Received request data:", {
            "text": data.text,
            "top5_predictions": data.top5_predictions
        })
        return analyze_image(data.image, data.text, data.top5_predictions)
    except Exception as e:
        print(f"Error in analyze_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
