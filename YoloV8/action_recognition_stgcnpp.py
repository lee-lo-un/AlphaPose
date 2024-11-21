import torch
import numpy as np
import json
from pathlib import Path
import os
from mmengine.runner import load_checkpoint
from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config


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
        device=device.type,
        cfg_options=None  # 필요시 추가 설정 가능
    )
    model.eval()

    return model, device


def preprocess_keypoints_for_stgcnpp(json_path):
    """JSON 파일의 키포인트를 STGCN++ 입력 형식으로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('people', []):
        return None

    # 첫 번째 사람의 키포인트 가져오기
    keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

    # (N, C, T, V, M) 형식으로 변환
    # N: 배치 크기, C: 채널(x, y, confidence), T: 시간축, V: 관절 수, M: 사람 수
    num_keypoints = keypoints.shape[0]
    input_data = np.zeros((1, 3, 1, num_keypoints, 1))

    # 좌표 및 신뢰도 채우기
    input_data[0, 0, 0, :, 0] = keypoints[:, 0]  # x 좌표
    input_data[0, 1, 0, :, 0] = keypoints[:, 1]  # y 좌표
    input_data[0, 2, 0, :, 0] = keypoints[:, 2]  # 신뢰도

    # 좌표 정규화
    input_data[..., :2] = normalize_coordinates(input_data[..., :2])

    return torch.tensor(input_data, dtype=torch.float32)


def normalize_coordinates(coords):
    """좌표 정규화"""
    # -1 ~ 1 범위로 정규화 (기본적으로 0 ~ 1로 스케일링된 데이터를 가정)
    coords = (coords - 0.5) * 2
    return coords


def recognize_action_stgcnpp(model, input_tensor, device):
    """STGCN++를 사용한 동작 인식"""
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
    # 경로 설정
    base_path = Path('d:/WORK/ICT_AI/AlphaPose')
    config_file = base_path / 'YoloV8/configs/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
    checkpoint_file = base_path / 'YoloV8/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-b00440d2.pth'

    # 경로 확인
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    try:
        # 모델 로드
        model, device = load_stgcnpp_model(str(config_file), str(checkpoint_file))
        print("STGCN++ 모델이 성공적으로 로드되었습니다.")

        # JSON 파일 처리
        json_dir = base_path / 'YoloV8/outputs'
        if not json_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {json_dir}")

        json_files = sorted(json_dir.glob('*_pose.json'))
        if not json_files:
            print(f"No JSON files found in {json_dir}")
            return

        for json_file in json_files:
            print(f"\n처리 중인 파일: {json_file.name}")

            input_tensor = preprocess_keypoints_for_stgcnpp(json_file)
            if input_tensor is None:
                print(f"키포인트 데이터를 찾을 수 없습니다: {json_file}")
                continue

            action, confidence = recognize_action_stgcnpp(model, input_tensor, device)
            print(f"인식된 동작: {action}")
            print(f"신뢰도: {confidence:.2f}")

            # 결과 저장
            with open(json_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data['action_recognition_stgcnpp'] = {
                    'action': action,
                    'confidence': confidence
                }
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
