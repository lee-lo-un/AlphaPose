import torch
import numpy as np
import json
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from graph import Graph
from st_gcn import st_gcn

class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting=True):
        super().__init__()
        
        # ST-GCN 모델 구조 정의
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # 데이터 정규화를 위한 공간 구성
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # ST-GCN 블록들
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))

        # 분류를 위한 fully connected 층
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # 데이터 정규화
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # ST-GCN 포워드 패스
        for gcn in self.st_gcn_networks:
            x = gcn(x, self.A)

        # 글로벌 풀링
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        
        # 예측
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

def load_stgcn_model(weights_path, num_class=60):
    """사전 학습된 ST-GCN 모델 로드"""
    model = STGCN(
        in_channels=3,
        num_class=num_class,
        graph_args={'layout': 'openpose', 'strategy': 'spatial'}
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

def preprocess_keypoints(json_path):
    """JSON 파일에서 키포인트 데이터를 ST-GCN 입력 형식으로 변환"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 키포인트 데이터 추출 및 변환
    people = data['people']
    if not people:
        return None
    
    # 첫 번째 사람의 키포인트만 사용 (여러 명일 경우 처리 방법 수정 가능)
    person = people[0]
    keypoints = person['keypoints']
    
    # ST-GCN 입력 형식으로 변환 (N, C, T, V, M)
    # N: 배치 크기, C: 채널 수 (x, y, confidence), T: 시간 프레임, V: 키포인트 수, M: 사람 수
    num_keypoints = len(keypoints)
    input_data = np.zeros((1, 3, 1, num_keypoints, 1))
    
    for i, kp in enumerate(keypoints):
        input_data[0, 0, 0, i, 0] = kp['coordinates']['x']  # x 좌표
        input_data[0, 1, 0, i, 0] = kp['coordinates']['y']  # y 좌표
        input_data[0, 2, 0, i, 0] = kp['confidence']        # 신뢰도
    
    return torch.FloatTensor(input_data)

def recognize_action(model, input_tensor, action_labels):
    """동작 인식 수행"""
    with torch.no_grad():
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        output = model(input_tensor)
        probability = F.softmax(output, dim=1)
        
        # 가장 높은 확률의 동작 클래스 선택
        max_prob, predicted = torch.max(probability, 1)
        action = action_labels[predicted.item()]
        confidence = max_prob.item()
        
        return action, confidence

def load_action_labels(label_file):
    """동작 레이블 파일 로드"""
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        return labels
    except FileNotFoundError:
        print(f"경고: {label_file}를 찾을 수 없습니다. 기본 레이블을 사용합니다.")
        # 기본 레이블 반환
        return [
            'walking', 'running', 'jumping', 'sitting', 'standing',
            'falling', 'waving', 'clapping', 'pointing', 'dancing',
            'eating', 'drinking', 'reading', 'typing', 'phone_calling',
            'exercising'
        ]

def main():
    # 모델 설정
    weights_path = 'weights/stgcn_kinetics.pth'  # 사전 학습된 가중치 경로
    action_labels = load_action_labels('config/action_labels.txt')  # 동작 레이블 파일
    model = load_stgcn_model(weights_path, num_class=len(action_labels))
    
    # JSON 파일 처리
    json_dir = Path('YoloV8/outputs')
    json_files = list(json_dir.glob('*_pose.json'))
    
    for json_file in json_files:
        print(f"\n처리 중인 파일: {json_file.name}")
        
        # 키포인트 데이터 전처리
        input_tensor = preprocess_keypoints(json_file)
        if input_tensor is None:
            print("키포인트 데이터를 찾을 수 없습니다.")
            continue
        
        # 동작 인식
        action, confidence = recognize_action(model, input_tensor, action_labels)
        print(f"인식된 동작: {action}")
        print(f"신뢰도: {confidence:.2f}")
        
        # 결과를 JSON 파일에 추가
        with open(json_file, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data['action_recognition'] = {
                'action': action,
                'confidence': confidence
            }
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

if __name__ == "__main__":
    main() 