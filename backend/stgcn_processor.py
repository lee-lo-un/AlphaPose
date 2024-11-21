import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
from pathlib import Path

class STGCNProcessor:
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        # NTU RGB+D 데이터셋의 60개 동작 클래스 직접 정의
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
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            config_file = project_root / 'mmaction2/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
            checkpoint_file = project_root / 'mmaction2/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 중인 디바이스: {self.device}")

            config = Config.fromfile(str(config_file))
            self.model = init_recognizer(config, str(checkpoint_file), device=self.device.type)
            self.model.eval()
            self.initialized = True
            print("ST-GCN++ 모델 초기화 완료")
            print(f"동작 클래스 수: {len(self.action_classes)}")
        except Exception as e:
            print(f"ST-GCN++ 모델 초기화 실패: {e}")
            raise

    def preprocess_keypoints(self, pose_data):
        """키포인트 데이터를 ST-GCN++ 입력 형식으로 변환"""
        if not pose_data or not pose_data.get('keypoints', []):
            print("유효한 키포인트 데이터가 없습니다.")
            return None

        keypoints = pose_data['keypoints']
        if not keypoints:  # 빈 리스트 체크
            print("키포인트 리스트가 비어있습니다.")
            return None

        try:
            # 유효한 키포인트만 필터링 (confidence > 0)
            valid_keypoints = [kp for kp in keypoints if kp['confidence'] > 0]
            if not valid_keypoints:
                print("유효한 키포인트가 없습니다.")
                return None

            # 이미지 크기 계산 (유효한 키포인트만 사용)
            max_x = max(kp['coordinates']['x'] for kp in valid_keypoints)
            max_y = max(kp['coordinates']['y'] for kp in valid_keypoints)
            img_shape = (int(max_y) + 1, int(max_x) + 1)

            num_keypoints = len(keypoints)
            keypoint_data = np.zeros((1, 1, num_keypoints, 2))  # (M, T, V, C)
            keypoint_score = np.zeros((1, 1, num_keypoints))    # (M, T, V)

            # 키포인트 데이터 채우기
            for i, kp in enumerate(keypoints):
                keypoint_data[0, 0, i, 0] = kp['coordinates']['x']  # x 좌표
                keypoint_data[0, 0, i, 1] = kp['coordinates']['y']  # y 좌표
                keypoint_score[0, 0, i] = kp['confidence']          # 신뢰도

            # mmaction2 형식에 맞게 입력 데이터 구성
            input_dict = {
                'keypoint': keypoint_data,           # (M, T, V, C)
                'keypoint_score': keypoint_score,    # (M, T, V)
                'img_shape': img_shape,              # (height, width)
                'modality': 'Pose',
                'label': -1,
                'start_index': 0,
                'frame_dir': '',
                'total_frames': 1,
                'clip_len': 1,
                'num_clips': 1,
                'frame_interval': 1
            }

            return input_dict

        except Exception as e:
            print(f"키포인트 전처리 중 오류 발생: {e}")
            return None

    def recognize_action(self, pose_data):
        """동작 인식 수행"""
        if not self.initialized:
            raise RuntimeError("ST-GCN++ 모델이 초기화되지 않았습니다.")

        try:
            input_dict = self.preprocess_keypoints(pose_data)
            if input_dict is None:
                return None

            with torch.no_grad():
                result = inference_recognizer(self.model, input_dict)

            if hasattr(result, 'pred_score'):
                pred_scores = result.pred_score.cpu().numpy()
            else:
                pred_scores = result[0].pred_score.cpu().numpy()

            # 가장 높은 확률의 동작만 반환
            pred_label = pred_scores.argmax()
            confidence = float(pred_scores[pred_label])
            action_class = self.action_classes[pred_label] if pred_label < len(self.action_classes) else f"action_{pred_label}"

            return {
                'action': action_class,
                'confidence': confidence,
                'label_index': int(pred_label)
            }

        except Exception as e:
            print(f"동작 인식 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None 