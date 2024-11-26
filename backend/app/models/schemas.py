from pydantic import BaseModel
from typing import List, Optional, Dict

class ImageData(BaseModel):
    image: str

class TextData(BaseModel):
    text: str

class DetectionResult(BaseModel):
    skeleton_data: Optional[Dict]
    object_data: Optional[List]
    action_result: Optional[str]
    similar_actions: Optional[List[str]]