import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
from pathlib import Path
import json


def load_stgcnpp_model(config_file, checkpoint_file):
    """STGCN++ 모델 로드"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 디바이스: {device}")

    # Config 로드
    config = Config.fromfile(config_file)

    # 모델 초기화
    model = init_recognizer(
        config=config,
        checkpoint=checkpoint_file,
        device=device.type
    )
    model.eval()

    return model, device


def preprocess_keypoints(json_file):
    """JSON에서 키포인트를 STGCN++ 입력 형식으로 변환"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('people') or len(data['people']) == 0:
        raise ValueError(f"No people data found in {json_file}")

    # 첫 번째 사람 데이터 가져오기
    person = data['people'][0]
    keypoints = person.get('keypoints', [])

    if not keypoints:
        raise ValueError(f"No keypoints found for person in {json_file}")

    # 키포인트 수 확인
    num_keypoints = len(keypoints)

    # (N, C, T, V, M): 배치, 채널(x, y, conf), 시간, 관절, 사람
    input_data = np.zeros((1, 3, 1, num_keypoints, 1))  # 단일 프레임, 단일 사람

    for i, kp in enumerate(keypoints):
        coordinates = kp['coordinates']
        input_data[0, 0, 0, i, 0] = coordinates['x']  # x 좌표
        input_data[0, 1, 0, i, 0] = coordinates['y']  # y 좌표
        input_data[0, 2, 0, i, 0] = kp['confidence']  # 신뢰도

    return torch.tensor(input_data, dtype=torch.float32)


def recognize_action(model, input_tensor, device):
    """STGCN++ 동작 인식 수행"""
    input_tensor = input_tensor.to(device)

    # 추론 수행
    result = inference_recognizer(model, input_tensor)

    # 결과 처리
    pred_scores = result.pred_scores.cpu().numpy()
    pred_label = pred_scores.argmax()
    confidence = pred_scores[pred_label]

    # 클래스 이름 가져오기
    classes = model.dataset_meta.get('classes', [])
    action_class = classes[pred_label] if classes else str(pred_label)

    return action_class, confidence


def main():
    # 현재 파일의 위치를 기준으로 상대 경로 설정
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    config_file = current_dir / 'configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
    checkpoint_file = current_dir / 'checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'
    json_file = project_root / 'YoloV8/outputs/1_pose.json'

    # 파일 확인
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    try:
        # 모델 로드
        model, device = load_stgcnpp_model(str(config_file), str(checkpoint_file))
        print("STGCN++ 모델이 성공적으로 로드되었습니다.")

        # 키포인트 데이터 처리
        input_tensor = preprocess_keypoints(json_file)

        # 동작 인식 수행
        action, confidence = recognize_action(model, input_tensor, device)

        print(f"인식된 동작: {action}")
        print(f"신뢰도: {confidence:.2f}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
