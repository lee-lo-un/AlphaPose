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
        self.real_fps = 35                     # YOLO 처리 속도 기준 FPS
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

        # 유효한 키포인트가 없으면 반환
        if not pose_data or not pose_data.get('keypoints', []):
            return None

        # 초기화된 프레임 수집 시작
        if not self.start_time:
            self.start_time = current_time
            self.last_process_time = current_time
            self.frame_buffer = []

        # 처리 간격 계산
        process_interval = current_time - self.last_process_time if self.last_process_time else 0
        self.last_process_time = current_time

        # 프레임 데이터 추가
        frame_data = {
            'timestamp': current_time,
            'process_interval': process_interval,
            'frame_number': len(self.frame_buffer) + 1,
            'keypoints': pose_data['keypoints']
        }
        self.frame_buffer.append(frame_data)

        # FPS 계산
        elapsed_time = current_time - self.start_time
        current_fps = len(self.frame_buffer) / elapsed_time if elapsed_time > 0 else 0

        print(f"\n=== 프레임 처리 상태 ===")
        print(f"프레임 간격: {process_interval * 1000:.1f}ms")
        print(f"목표 간격: {self.frame_time * 1000:.1f}ms")
        print(f"수집 시간: {elapsed_time:.2f}초")
        print(f"프레임 수: {len(self.frame_buffer)} / {self.buffer_size}")
        print(f"현재 FPS: {current_fps:.1f} (목표: {self.real_fps})")
        print("=" * 30)

        # 필요한 프레임 수에 도달하지 않으면 반환
        if len(self.frame_buffer) < self.buffer_size:
            remaining_frames = self.buffer_size - len(self.frame_buffer)
            print(f"추가 수집 필요: {remaining_frames}프레임")
            return None

        # 수집 완료 후 입력 데이터 준비
        return self.prepare_input_data()

    def prepare_input_data(self):
        try:
            num_keypoints = len(self.frame_buffer[0]['keypoints'])
            keypoint_data = np.zeros((1, self.buffer_size, num_keypoints, 2))
            keypoint_score = np.zeros((1, self.buffer_size, num_keypoints))

            recent_frames = self.frame_buffer[-self.buffer_size:]
            for i, frame_data in enumerate(recent_frames):
                keypoints = frame_data['keypoints']
                for j, kp in enumerate(keypoints):
                    keypoint_data[0, i, j, 0] = kp['coordinates']['x']
                    keypoint_data[0, i, j, 1] = kp['coordinates']['y']
                    keypoint_score[0, i, j] = kp['confidence']

            # 정규화
            valid_mask = keypoint_score > 0.5
            valid_points = keypoint_data.reshape(-1, 2)[valid_mask.reshape(-1)]
            mean_xy = np.mean(valid_points, axis=0)
            std_xy = np.std(valid_points, axis=0)
            std_xy = np.where(std_xy == 0, 1e-6, std_xy)
            keypoint_data = (keypoint_data - mean_xy) / std_xy

            input_dict = {
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

            self.frame_buffer = self.frame_buffer[-self.buffer_size:]
            return input_dict
        except Exception as e:
            print(f"입력 데이터 준비 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def recognize_action(self, pose_data):
        if not self.initialized:
            raise RuntimeError("ST-GCN++ 모델이 초기화되지 않았습니다.")

        try:
            input_dict = self.preprocess_keypoints(pose_data)
            if input_dict is None:
                return None

            with torch.no_grad():
                result = inference_recognizer(self.model, input_dict)

            pred_scores = result.pred_score.cpu().numpy() if hasattr(result, 'pred_score') else result[0].pred_score.cpu().numpy()

            top_k = 5
            top_indices = pred_scores.argsort()[-top_k:][::-1]
            for idx in top_indices:
                action_class = self.action_classes[idx]
                confidence = float(pred_scores[idx])
                print(f"동작: {action_class:<30} 신뢰도: {confidence * 100:>6.2f}%")

            pred_label = pred_scores.argmax()
            confidence = float(pred_scores[pred_label])

            if confidence < 0.6:
                print(f"신뢰도가 너무 낮습니다: {confidence * 100:.1f}%")
                return None

            return {
                'action': self.action_classes[pred_label],
                'confidence': confidence,
                'label_index': int(pred_label)
            }
        except Exception as e:
            print(f"동작 인식 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
