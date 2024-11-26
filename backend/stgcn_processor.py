import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
from pathlib import Path
import time


class STGCNProcessor:
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.frame_buffer = []
        self.real_fps = 10                     # YOLO 처리 속도 기준 FPS
        self.frame_time = 1.0 / self.real_fps  # 프레임 간 예상 시간 간격
        self.buffer_size = 6                   # 필요한 프레임 수
        self.collection_time = self.buffer_size / self.real_fps  # 예상 수집 시간
        self.start_time = None
        self.last_process_time = None
        self.action_classes = [
            'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pickup',
            'throw', 'sit down', 'stand up', 'clapping', 'reading', 'writing',
            'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe',
            'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
            'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something',
            'reach into pocket', 'hopping', 'jump up', 'phone call', 'play with phone/tablet',
            'type on a keyboard', 'point to something', 'taking a selfie', 'check time (from watch)',
            'rub two hands', 'nod head/bow', 'shake head', 'wipe face', 'salute',
            'put palms together', 'cross hands in front', 'sneeze/cough', 'staggering',
            'falling down', 'headache', 'chest pain', 'back pain', 'neck pain',
            'nausea/vomiting', 'fan self', 'punch/slap', 'kicking', 'pushing',
            'pat on back', 'point finger', 'hugging', 'giving object', 'touch pocket',
            'shaking hands', 'walking towards', 'walking apart'
        ]

    def initialize(self):
        try:
            # 설정 파일 및 체크포인트 경로
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            config_file = project_root / 'mmaction2/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
            checkpoint_file = project_root / 'mmaction2/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'

            # 디바이스 설정
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 중인 디바이스: {self.device}")

            # 모델 초기화
            config = Config.fromfile(str(config_file))
            self.model = init_recognizer(config, str(checkpoint_file), device=self.device.type)

            # 클래스 정보 업데이트
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

    def preprocess_keypoints(self, pose_data):
        current_time = time.time()

        # 유효한 키포인트 확인
        if not pose_data or not pose_data.get('keypoints', []):
            return None

        # 슬라이딩 윈도우 업데이트
        frame_data = {
            'timestamp': current_time,
            'keypoints': pose_data['keypoints']
        }
        self.frame_buffer.append(frame_data)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)  # 오래된 프레임 제거

        # 버퍼가 충분히 채워지지 않았을 경우, 현재 상태 기반으로 처리
        if len(self.frame_buffer) < self.buffer_size:
            print(f"버퍼 부족: {len(self.frame_buffer)}/{self.buffer_size}. 현재 데이터 처리.")
            return self.prepare_input_data(partial=True)

        # 충분히 채워진 경우 데이터 준비
        return self.prepare_input_data(partial=False)

    def prepare_input_data(self, partial=False):
        """입력 데이터 준비 (슬라이딩 윈도우 기반)"""
        try:
            actual_buffer_size = len(self.frame_buffer)
            num_keypoints = len(self.frame_buffer[0]['keypoints'])

            # 데이터 초기화
            keypoint_data = np.zeros((1, self.buffer_size, num_keypoints, 2))
            keypoint_score = np.zeros((1, self.buffer_size, num_keypoints))

            for i, frame_data in enumerate(self.frame_buffer):
                keypoints = frame_data['keypoints']
                for j, kp in enumerate(keypoints):
                    keypoint_data[0, i, j, 0] = kp['coordinates']['x']
                    keypoint_data[0, i, j, 1] = kp['coordinates']['y']
                    keypoint_score[0, i, j] = kp['confidence']

            # 부족한 데이터 패딩
            if partial and actual_buffer_size < self.buffer_size:
                keypoint_data[:, actual_buffer_size:, :, :] = 0
                keypoint_score[:, actual_buffer_size:, :] = 0

            # 정규화 처리
            valid_mask = keypoint_score > 0.5
            valid_points = keypoint_data.reshape(-1, 2)[valid_mask.reshape(-1)]
            mean_xy = np.mean(valid_points, axis=0) if valid_points.size > 0 else [0, 0]
            std_xy = np.std(valid_points, axis=0) if valid_points.size > 0 else [1e-6, 1e-6]
            keypoint_data = (keypoint_data - mean_xy) / std_xy

            return {
                'keypoint': keypoint_data,
                'keypoint_score': keypoint_score,
                'img_shape': (1080, 1920),
                'modality': 'Pose',
                'label': -1,
                'start_index': 0,
                'total_frames': self.buffer_size,
                'clip_len': self.buffer_size,
                'num_clips': 1,
                'frame_interval': 1
            }
        except Exception as e:
            print(f"입력 데이터 준비 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def recognize_action(self, pose_data):
        if not self.initialized:
            raise RuntimeError("ST-GCN++ 모델이 초기화되지 않았습니다.")

        try:
            # `preprocess_keypoints`로 입력 데이터 준비
            input_dict = self.preprocess_keypoints(pose_data)
            if input_dict is None:
                return None

            # ST-GCN++ 모델 실행
            with torch.no_grad():
                result = inference_recognizer(self.model, input_dict)

            # 결과 처리 및 상위 5개의 예측 출력
            pred_scores = result.pred_score.cpu().numpy() if hasattr(result, 'pred_score') else result[0].pred_score.cpu().numpy()
            
            top_k = 5
            top_indices = pred_scores.argsort()[-top_k:][::-1]
            # 상위 5개의 동작과 신뢰도를 저장
            top5_predictions = []
            for idx in top_indices:
                action_class = self.action_classes[idx]
                confidence = float(pred_scores[idx])
                top5_predictions.append({'action': action_class, 'confidence': confidence})
                print(f"동작: {action_class:<30} 신뢰도: {confidence * 100:>6.2f}%")
            
            # 가장 높은 점수의 동작 선택
            pred_label = pred_scores.argmax()
            confidence = float(pred_scores[pred_label])

            # 신뢰도가 낮으면 None 반환
            if confidence < 0.3:
                print(f"신뢰도가 너무 낮습니다: {confidence * 100:.1f}%")
                return None

            # 결과 반환
            return {
                'action': self.action_classes[pred_label],
                'confidence': confidence,
                'label_index': int(pred_label),
                'top5': top5_predictions  # 추가된 부분
            }
        except Exception as e:
            print(f"동작 인식 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def send_predictions_to_frontend(self, predictions):
        print("Sending predictions to frontend:", {
            "type": "action",
            "data": {
                "top5": predictions
            }
        })