import sys
import os
import torch
# GPU가 사용 가능한지 확인
if torch.cuda.is_available():
    print("CUDA is available! 🚀")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Running on CPU.")

# 현재 파일의 디렉토리에서 두 단계 위로 이동하여 project_root 경로를 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 경로가 제대로 추가되었는지 확인
print(sys.path)

# alphapose 패키지 가져오기
import AlphaPose