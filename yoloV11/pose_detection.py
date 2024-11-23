from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# COCO 데이터셋 클래스 이름 정의
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def get_keypoint_name(idx):
    """키포인트 인덱스에 해당하는 이름을 반환"""
    keypoint_names = {
        0: "코", 1: "왼쪽 눈", 2: "오른쪽 눈", 3: "왼쪽 귀", 4: "오른쪽 귀",
        5: "왼쪽 어깨", 6: "오른쪽 어깨", 7: "왼쪽 팔꿈치", 8: "오른쪽 팔꿈치",
        9: "왼쪽 손목", 10: "오른쪽 손목", 11: "왼쪽 엉덩이", 12: "오른쪽 엉덩이",
        13: "왼쪽 무릎", 14: "오른쪽 무릎", 15: "왼쪽 발목", 16: "오른쪽 발목"
    }
    return keypoint_names.get(idx, f"키포인트_{idx}")

def process_image(image_path, detect_model, pose_model, output_folder):
    """단일 이미지 처리"""
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 파일 이름 추출 (확장자 제외)
    image_name = Path(image_path).stem

    # 물체 인식 ��� 포즈 추정 실행
    detect_results = detect_model(image)
    pose_results = pose_model(image)

    # 결과 출력
    print(f"\n=== {image_name} 분석 결과 ===")
    
    # 물체 인식 결과
    print("\n물체 인식 결과:")
    for result in detect_results:
        boxes = result.boxes
        if len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                cls_name = COCO_CLASSES.get(cls, f"class_{cls}")
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy()
                print(f"클래스: {cls_name}, 신뢰도: {conf:.2f}")
                print(f"박스 좌표: {coords}")

    # JSON으로 저장할 데이터 구조 생성
    pose_data = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "people": []
    }

    # 포즈 추정 결과
    print("\n포즈 추정 결과:")
    for result in pose_results:
        keypoints = result.keypoints
        if keypoints is not None:
            keypoints_data = keypoints.data.cpu().numpy()
            for person_idx, person_keypoints in enumerate(keypoints_data):
                person_data = {
                    "person_id": person_idx,
                    "keypoints": []
                }
                print(f"\n사람 {person_idx + 1}의 키포인트:")
                for kp_idx, kp in enumerate(person_keypoints):
                    x, y, conf = kp
                    kp_name = get_keypoint_name(kp_idx)
                    print(f"{kp_name}: x={x:.2f}, y={y:.2f}, confidence={conf:.2f}")
                    
                    # JSON 데이터에 키포인트 추가
                    person_data["keypoints"].append({
                        "name": kp_name,
                        "id": kp_idx,
                        "coordinates": {
                            "x": float(x),
                            "y": float(y)
                        },
                        "confidence": float(conf)
                    })
                pose_data["people"].append(person_data)

    # JSON 파일로 저장
    json_output_path = os.path.join(output_folder, f'{image_name}_pose.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, ensure_ascii=False, indent=2)
    print(f"포즈 데이터 JSON 저장됨: {json_output_path}")

    # 결과 시각화
    img = image.copy()
    
    # 물체 인식 결과 그리기
    for result in detect_results:
        boxes = result.boxes
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            cls_name = COCO_CLASSES.get(cls, f"class_{cls}")
            conf = float(box.conf[0])
            
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (coords[0], coords[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 포즈 추정 결과 그리기
    for result in pose_results:
        keypoints = result.keypoints
        if keypoints is not None:
            keypoints_data = keypoints.data.cpu().numpy()
            for person_keypoints in keypoints_data:
                # 키포인트 그리기
                for x, y, conf in person_keypoints:
                    if conf > 0.5:  # 신뢰도가 높은 키포인트만 표시
                        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    # 결과 저장
    output_path = os.path.join(output_folder, f'{image_name}_result.jpg')
    cv2.imwrite(output_path, img)
    print(f"결과 이미지 저장됨: {output_path}")

def initialize_models():
    """모델 초기화"""
    detect_model = YOLO('inference/yolo11l.pt')
    pose_model = YOLO('inference/yolo11l-pose.pt')
    return detect_model, pose_model

def process_realtime(frame, detect_model, pose_model):
    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            # YOLO 처리
            detect_results = detect_model(frame, conf=0.5)  # 신뢰도 임계값 설정
            pose_results = pose_model(frame, conf=0.5)
            
            # 결과 처리
            results = {
                'objects': [],
                'poses': []
            }
            
            # 객체 감지 결과 처리
            for result in detect_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    cls_name = COCO_CLASSES.get(cls, f"class_{cls}")
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].cpu().numpy()
                    
                    results['objects'].append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': coords.tolist()
                    })
            
            # 포즈 추정 결과 처리
            for result in pose_results:
                keypoints = result.keypoints
                if keypoints is not None:
                    keypoints_data = keypoints.data.cpu().numpy()
                    for person_idx, person_keypoints in enumerate(keypoints_data):
                        person_data = {
                            'person_id': person_idx,
                            'keypoints': []
                        }
                        
                        for kp_idx, kp in enumerate(person_keypoints):
                            x, y, conf = kp
                            kp_name = get_keypoint_name(kp_idx)
                            person_data['keypoints'].append({
                                'name': kp_name,
                                'coordinates': {
                                    'x': float(x),
                                    'y': float(y)
                                },
                                'confidence': float(conf)
                            })
                        results['poses'].append(person_data)
            
            return results

def process_single_person_with_objects(image, detect_model, pose_model):
    """
    이미지에서 가장 큰 사람의 스켈레톤 데이터를 요청하신 구조로 반환하고, 객체 데이터를 함께 반환.
    """
    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            # YOLO 처리
            detect_results = detect_model(image, conf=0.5)  # 신뢰도 임계값 설정
            pose_results = pose_model(image, conf=0.5)
            
            # 결과 저장을 위한 구조
            skeleton_data = {
                "keypoints": {},
                "confidence_scores": {}
            }
            object_data = []

            # 객체 감지 결과 처리
            for result in detect_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    cls_name = COCO_CLASSES.get(cls, f"class_{cls}")
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].cpu().numpy()

                    object_data.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': coords.tolist()
                    })

            # 포즈 추정 결과 처리
            largest_area = 0  # 가장 큰 사람을 찾기 위한 기준
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]

            for result in pose_results:
                keypoints = result.keypoints
                if keypoints is not None:
                    keypoints_data = keypoints.data.cpu().numpy()
                    for person_keypoints in keypoints_data:
                        # x, y 좌표만 추출
                        x_coords = person_keypoints[:, 0]
                        y_coords = person_keypoints[:, 1]

                        # 빈 배열 방지
                        if x_coords.size == 0 or y_coords.size == 0:
                            continue  # 다음 사람으로 넘어감

                        # 키포인트를 기준으로 크기 계산
                        x_min, x_max = x_coords.min(), x_coords.max()
                        y_min, y_max = y_coords.min(), y_coords.max()
                        area = (x_max - x_min) * (y_max - y_min)

                        # 가장 큰 사람 찾기
                        if area > largest_area:
                            largest_area = area
                            skeleton_data = {  # 구조 생성
                                "keypoints": {},
                                "confidence_scores": {}
                            }
                            for idx, (x, y, conf) in enumerate(person_keypoints):
                                if idx < len(keypoint_names):  # 키포인트 이름이 있는 경우만 처리
                                    keypoint_name = keypoint_names[idx]
                                    skeleton_data["keypoints"][keypoint_name] = {
                                        "x": float(x),
                                        "y": float(y),
                                        "z": 0  # 2D 이미지이므로 z는 0으로 설정
                                    }
                                    skeleton_data["confidence_scores"][keypoint_name] = float(conf)

    return skeleton_data, object_data



def main():
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print("CUDA is available! 🚀")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Running on CPU.")

    # 모델 로드
    detect_model = YOLO('inference/yolo11l.pt')
    pose_model = YOLO('inference/yolo11l-pose.pt')

    # 입력/출력 폴더 설정
    input_folder = 'img'
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 입력 폴더의 모든 이미지 처리
    image_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"Error: No image files found in {input_folder}")
        return

    print(f"\n총 {len(image_files)}개의 이미지를 처리합니다.")
    
    # 각 이미지 처리
    for image_path in image_files:
        try:
            process_image(image_path, detect_model, pose_model, output_folder)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    print("\n모든 이미지 처리가 완료되었습니다.")
# alex 수정 test1
# alex 수정 test2
# alex 수정 test3

if __name__ == "__main__":
    main()