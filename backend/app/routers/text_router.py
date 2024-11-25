from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class TextData(BaseModel):
    text: str

@router.post("/")
async def process_text(data: TextData):
    try:
        print(f"받은 텍스트: {data.text}")
        return {"status": "success", "message": "텍스트가 성공적으로 처리되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
