from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os

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

def visualize_results(image, detect_boxes, pose_keypoints, output_path):
    """결과를 시각화하여 저장"""
    img = image.copy()
    
    # 물체 인식 결과 그리기
    if detect_boxes is not None:
        for box in detect_boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            cls_name = COCO_CLASSES.get(cls, f"class_{cls}")
            conf = float(box.conf[0])
            
            cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (coords[0], coords[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 포즈 추정 결과 그리기
    if pose_keypoints is not None:
        keypoints_data = pose_keypoints.data.cpu().numpy()
        for person_keypoints in keypoints_data:
            # 키포인트 그리기
            for x, y, conf in person_keypoints:
                if conf > 0.5:  # 신뢰도가 높은 키포인트만 표시
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 결과 저장
    cv2.imwrite(output_path, img)

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

    # 이미지 경로 설정
    image_path = 'img/women_apple.jpg'
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 물체 인식 실행
    detect_results = detect_model(image)
    
    # 포즈 추정 실행
    pose_results = pose_model(image)

    # 물체 인식 결과 출력
    print("\n=== 물체 인식 결과 ===")
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

    # 포즈 추정 결과 출력
    print("\n=== 포즈 추정 결과 ===")
    for result in pose_results:
        keypoints = result.keypoints
        if keypoints is not None:
            keypoints_data = keypoints.data.cpu().numpy()
            for person_idx, person_keypoints in enumerate(keypoints_data):
                print(f"\n사람 {person_idx + 1}의 키포인트:")
                for kp_idx, kp in enumerate(person_keypoints):
                    x, y, conf = kp
                    kp_name = get_keypoint_name(kp_idx)
                    print(f"{kp_name}: x={x:.2f}, y={y:.2f}, confidence={conf:.2f}")

    # 결과 시각화
    output_path = os.path.join(output_folder, 'combined_results.jpg')
    visualize_results(image, detect_results[0].boxes, pose_results[0].keypoints, output_path)
    print(f"\n결과 이미지가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main()