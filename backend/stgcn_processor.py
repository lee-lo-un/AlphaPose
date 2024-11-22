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
        self.buffer_size = 6                 # 필요한 프레임 수
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
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent
            config_file = project_root / 'mmaction2/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
            checkpoint_file = project_root / 'mmaction2/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 중인 디바이스: {self.device}")

            config = Config.fromfile(str(config_file))
            self.model = init_recognizer(config, str(checkpoint_file), device=self.device.type)
            
            # config 파일에서 클래스 정보 가져오기
            if hasattr(config, 'label_map'):
                self.action_classes = config.label_map
            elif hasattr(self.model, 'dataset_meta'):
                self.action_classes = self.model.dataset_meta.get('classes', [])
            
            print(f"동작 클래스 수: {len(self.action_classes)}")
            print("사용 가능한 동작 클래스:")
            for i, action in enumerate(self.action_classes):
                print(f"{i}: {action}")
            
            self.model.eval()
            self.initialized = True
            print("ST-GCN++ 모델 초기화 완료")
            
        except Exception as e:
            print(f"ST-GCN++ 모델 초기화 실패: {e}")
            raise

    def preprocess_keypoints(self, pose_data):
        current_time = time.time()
        
        if not pose_data or not pose_data.get('keypoints', []):
            return None

        # 첫 프레임이면 시작
        if not self.start_time:
            self.start_time = current_time
            self.last_process_time = current_time
            self.frame_buffer = []

        # 이전 처리와의 시간 간격 계산
        if self.last_process_time:
            process_interval = current_time - self.last_process_time
        else:
            process_interval = 0
        self.last_process_time = current_time

        frame_data = {
            'timestamp': current_time,
            'process_interval': process_interval,
            'frame_number': len(self.frame_buffer) + 1,
            'keypoints': pose_data['keypoints']
        }
        
        self.frame_buffer.append(frame_data)

        elapsed_time = current_time - self.start_time
        current_fps = len(self.frame_buffer) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n=== 프레임 처리 상태 ===")
        print(f"프레임 간격: {process_interval*1000:.1f}ms")
        print(f"목표 간격: {self.frame_time*1000:.1f}ms")
        print(f"수집 시간: {elapsed_time:.2f}초")
        print(f"프레임 수: {len(self.frame_buffer)} / {self.buffer_size}")
        print(f"현재 FPS: {current_fps:.1f} (목표: {self.real_fps})")
        print("="*30)

        # 필요한 프레임 수에 도달하지 않았으면 계속 수집
        if len(self.frame_buffer) < self.buffer_size:
            remaining_frames = self.buffer_size - len(self.frame_buffer)
            remaining_time = remaining_frames * self.frame_time
            print(f"추가 수집 필요: {remaining_frames}프레임")
            return None

        # 100프레임이 모이면 처리 시작
        return self.prepare_input_data()

    def prepare_input_data(self):
        """수집된 프레임으로 입력 데이터 준비"""
        try:
            num_keypoints = len(self.frame_buffer[0]['keypoints'])
            keypoint_data = np.zeros((1, self.buffer_size, num_keypoints, 2))
            keypoint_score = np.zeros((1, self.buffer_size, num_keypoints))

            # 가장 최근 100프레임 사용
            recent_frames = self.frame_buffer[-self.buffer_size:]
            
            for i, frame_data in enumerate(recent_frames):
                keypoints = frame_data['keypoints']
                for j, kp in enumerate(keypoints):
                    keypoint_data[0, i, j, 0] = kp['coordinates']['x']
                    keypoint_data[0, i, j, 1] = kp['coordinates']['y']
                    keypoint_score[0, i, j] = kp['confidence']

            # 데이터 정규화
            valid_mask = keypoint_score > 0.5
            valid_mask = valid_mask.reshape(-1)
            
            if np.mean(valid_mask) < 0.6:
                print("유효한 키포인트 비율이 너무 낮습니다.")
                return None

            valid_points = keypoint_data.reshape(-1, 2)[valid_mask]
            mean_xy = np.mean(valid_points, axis=0)
            std_xy = np.std(valid_points, axis=0)
            std_xy = np.where(std_xy == 0, 1e-6, std_xy)
            keypoint_data = (keypoint_data - mean_xy) / std_xy

            print(f"\n=== 시퀀스 데이터 분석 ===")
            print(f"원본 프레임 수: {len(self.frame_buffer)}")
            print(f"보간된 프레임 수: {self.buffer_size}")
            print(f"시퀀스 길이: {self.collection_time:.2f}초")
            print(f"유효한 키포인트 비율: {np.mean(valid_mask)*100:.1f}%")
            print("="*30)

            # mmaction2 형식으로 구성
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

            # 버퍼 유지 (슬라이딩 윈도우처럼 작동)
            self.frame_buffer = self.frame_buffer[-self.buffer_size:]
            
            return input_dict

        except Exception as e:
            print(f"입력 데이터 준비 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_same_frame(self, frame1, frame2):
        """두 프레임이 동일한지 확인"""
        if not frame1 or not frame2:
            return False
        
        kp1 = frame1['keypoints']
        kp2 = frame2['keypoints']
        
        if len(kp1) != len(kp2):
            return False
        
        # 첫 번째 키포인트의 좌표만 비교
        if len(kp1) > 0 and len(kp2) > 0:
            x1 = kp1[0]['coordinates']['x']
            y1 = kp1[0]['coordinates']['y']
            x2 = kp2[0]['coordinates']['x']
            y2 = kp2[0]['coordinates']['y']
            
            # 완전히 동일한 좌표라면 같은 프레임으로 간주
            return x1 == x2 and y1 == y2
        
        return False

    def recognize_action(self, pose_data):
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

            # 상위 5개 예측 결과 출력 (디버깅용)
            top_k = 5
            top_indices = pred_scores.argsort()[-top_k:][::-1]
            
            print("\n=== 상위 5개 동작 예측 결과 ===")
            for idx in top_indices:
                action_class = self.action_classes[idx]
                confidence = float(pred_scores[idx])
                print(f"동작: {action_class:<30} 신뢰도: {confidence*100:>6.2f}%")
            print("="*50)

            # 키포인트 움직임 분석
            if len(self.frame_buffer) >= 2:
                latest_frame = np.array([[kp['coordinates']['x'], kp['coordinates']['y']] 
                                       for kp in self.frame_buffer[-1]['keypoints']])
                prev_frame = np.array([[kp['coordinates']['x'], kp['coordinates']['y']] 
                                     for kp in self.frame_buffer[-2]['keypoints']])
                
                # 전체적인 움직임 계산
                movement = np.mean(np.abs(latest_frame - prev_frame))
                print(f"프레임 간 평균 움직임: {movement:.2f} pixels")
                
                # 상체와 하체의 움직임 분석
                upper_body = slice(0, 8)  # 상체 키포인트 인덱스
                lower_body = slice(8, 16)  # 하체 키포인트 인덱스
                
                upper_movement = np.mean(np.abs(latest_frame[upper_body] - prev_frame[upper_body]))
                lower_movement = np.mean(np.abs(latest_frame[lower_body] - prev_frame[lower_body]))
                
                print(f"상체 움직임: {upper_movement:.2f} pixels")
                print(f"하체 움직임: {lower_movement:.2f} pixels")
            print("="*50)

            # 신뢰도 임계값 적용
            pred_label = pred_scores.argmax()
            confidence = float(pred_scores[pred_label])
            
            # 신뢰도가 낮으면 None 반환
            if confidence < 0.6:  # 60% 미만의 신뢰도는 무시
                print(f"신뢰도가 너무 낮습니다: {confidence*100:.1f}%")
                return None

            action_class = self.action_classes[pred_label]

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