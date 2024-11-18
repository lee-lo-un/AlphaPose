from ultralytics import YOLO
import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    print("CUDA is available! 🚀")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Running on CPU.")

# YOLO11 포즈 추정 모델 로드
model = YOLO('inference/yolo11l-pose.pt')  # 대형 모델 사용

# 이미지 폴더 경로 설정
image_folder = 'img'
output_folder = 'outputs'

# 포즈 추정 실행 및 결과 저장
results = model(image_folder,
               conf=0.5,          # 신뢰도 임계값
               save=True,         # 결과 저장
               save_txt=True,     # 텍스트 결과 저장
               save_conf=True,    # 신뢰도 점수 저장
               project=output_folder,
               name='pose_results')

# 결과 출력
for result in results:
    boxes = result.boxes  # 바운딩 박스
    keypoints = result.keypoints  # 키포인트 (x, y, confidence)
    
    if keypoints is not None:
        print(f"\nDetected poses in {result.path}:")
        print(f"Number of detections: {len(keypoints)}")
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Keypoints data: {keypoints.data}")

# 모델 내보내기 (선택사항)
#model.export(format="onnx", dynamic=True, simplify=True)