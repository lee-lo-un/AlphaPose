import os
from ultralytics import YOLO

# YOLOv8 모델 로드 (Nano 모델 사용: 경량화된 버전)
model = YOLO('inference/yolov8n.pt')  # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt' 등 선택 가능

# 이미지 폴더 경로
image_folder = 'img'

# 이미지 폴더 내 모든 파일에 대해 반복
for image_file in os.listdir(image_folder):
    # 이미지 파일 경로 생성
    image_path = os.path.join(image_folder, image_file)
    
    # 이미지 파일인지 확인
    if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 단일 이미지에서 객체 탐지 실행
        results = model(image_path)
        
        # 모든 결과에 대해 탐지 결과 표시 및 출력
        for result in results:
            result.show()
            print(f"Results for {image_file}:")
            print(result.xyxy)  # xyxy 포맷의 좌표와 클래스 정보