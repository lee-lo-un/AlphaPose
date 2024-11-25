from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.image_analysis import analyze_image
from typing import Optional, Dict, List, Union

router = APIRouter()

class ImageData(BaseModel):
    image: str
    top5_predictions: Optional[List[Dict[str, Union[str, float]]]] = None

@router.post("/")
async def analyze_image_endpoint(data: ImageData):
    try:
        return analyze_image(data.image, data.top5_predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
