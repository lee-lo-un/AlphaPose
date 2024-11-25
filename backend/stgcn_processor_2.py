import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
from pathlib import Path
import time


class STGCNProcessor:
    def __init__(self, config_path, checkpoint_path, device=None):
        self.model = None
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialized = False
        self.frame_buffer = []
        self.clip_len = 16  # 클립 길이
        self.frame_interval = 2  # 프레임 간격
        self.buffer_size = self.clip_len * self.frame_interval
        self.action_classes = []
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def initialize(self):
        try:
            # MMACTION2 모델 초기화
            config = Config.fromfile(self.config_path)
            self.model = init_recognizer(config, self.checkpoint_path, device=self.device.type)

            # 클래스 정보 로드
            if hasattr(config, 'label_map'):
                self.action_classes = config.label_map
            elif hasattr(self.model, 'dataset_meta'):
                self.action_classes = self.model.dataset_meta.get('classes', [])

            self.model.eval()
            self.initialized = True
            print("ST-GCN++ 모델 초기화 완료")
        except Exception as e:
            print(f"ST-GCN++ 모델 초기화 실패: {e}")
            raise

    def add_frame(self, pose_data):
        """버퍼에 새 프레임 추가"""
        if not pose_data or not pose_data.get('keypoints', []):
            return False

        # 버퍼 크기 제한
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)

        self.frame_buffer.append(pose_data['keypoints'])
        return len(self.frame_buffer) == self.buffer_size

    def prepare_input_data(self):
        """ST-GCN에 전달할 입력 데이터 준비"""
        try:
            keypoint_data = np.array(self.frame_buffer, dtype=np.float32).transpose(1, 0, 2)
            keypoint_score = np.ones(keypoint_data.shape[:-1])  # 기본 신뢰도

            input_dict = {
                'keypoint': keypoint_data,
                'keypoint_score': keypoint_score,
                'modality': 'Pose',
                'clip_len': self.clip_len,
                'frame_interval': self.frame_interval,
                'num_clips': 1,
            }
            return input_dict
        except Exception as e:
            print(f"입력 데이터 준비 중 오류 발생: {e}")
            return None

    def recognize_action(self):
        """ST-GCN 모델로 동작 인식 수행"""
        if not self.initialized:
            raise RuntimeError("ST-GCN++ 모델이 초기화되지 않았습니다.")
        if len(self.frame_buffer) < self.buffer_size:
            print("버퍼에 충분한 프레임이 없습니다.")
            return None

        try:
            input_dict = self.prepare_input_data()
            if input_dict is None:
                return None

            with torch.no_grad():
                result = inference_recognizer(self.model, input_dict)
                pred_scores = result.pred_score.cpu().numpy()
                pred_label = pred_scores.argmax()
                confidence = pred_scores[pred_label]

                if confidence < 0.6:
                    print(f"신뢰도가 너무 낮습니다: {confidence * 100:.1f}%")
                    return None

                return {
                    'action': self.action_classes[pred_label],
                    'confidence': confidence,
                }
        except Exception as e:
            print(f"동작 인식 중 오류 발생: {e}")
            return None
