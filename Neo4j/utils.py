import numpy as np
from config import GAUSSIAN_CONFIG, ANGLE_CONFIG

def calculate_gaussian_weights(angles, mu, sigma):
    """
    주어진 각도 데이터에 대해 가우시안 가중치를 계산합니다.
    중심값(mu)에 가까울수록 높은 가중치를 가집니다.
    """
    if sigma == 0:
        return [1.0] * len(angles)
    weights = np.exp(-((np.array(angles) - mu) ** 2) / (2 * sigma ** 2))
    return weights.tolist()

def update_gaussian_parameters(existing_angles, new_angle, current_mu, current_sigma, learning_rate=None):
    """
    새로운 각도 데이터가 들어올 때 가우시안 파라미터(μ, σ)를 업데이트합니다.
    
    Parameters:
    - existing_angles: 기존 각도 데이터 리스트
    - new_angle: 새로 추가될 각도 값
    - current_mu: 현재 평균값
    - current_sigma: 현재 표준편차
    - learning_rate: 학습률 (새로운 데이터의 영향력 조절)
    
    Returns:
    - new_mu: 업데이트된 평균
    - new_sigma: 업데이트된 표준편차
    - score: 새로운 각도의 적합도 점수 (0~1)
    """
    # 기본 학습률 설정
    if learning_rate is None:
        learning_rate = GAUSSIAN_CONFIG["LEARNING_RATE"]
    
    # 기존 데이터가 없는 경우
    if not existing_angles:
        return new_angle, GAUSSIAN_CONFIG["MIN_SIGMA"], 1.0
    
    # 새로운 데이터의 가우시안 점수 계산
    score = np.exp(-((new_angle - current_mu) ** 2) / (2 * current_sigma ** 2))
    
    # 모든 데이터 포함
    all_angles = existing_angles + [new_angle]
    
    # 적응적 평균 업데이트 (점수가 높을수록 더 큰 영향)
    weighted_mu = score * new_angle + (1 - score) * current_mu
    new_mu = current_mu + learning_rate * (weighted_mu - current_mu)
    
    # 적응적 표준편차 업데이트
    # 현재 분포에서 벗어난 데이터는 제한된 영향만 줌
    distances = np.abs(np.array(all_angles) - new_mu)
    new_sigma = np.sqrt(np.mean(distances ** 2))
    
    # 갑작스러운 변화 방지
    new_sigma = current_sigma + learning_rate * (new_sigma - current_sigma)
    
    # 최소 표준편차 보장
    new_sigma = max(new_sigma, GAUSSIAN_CONFIG["MIN_SIGMA"])
    
    return new_mu, new_sigma, score

def calculate_angle_range(mu, sigma):
    """
    현재 가우시안 파라미터를 기반으로 유효 범위를 계산합니다.
    설정된 신뢰구간을 사용
    """
    confidence = GAUSSIAN_CONFIG["CONFIDENCE_SIGMA"]
    range_min = max(ANGLE_CONFIG["MIN_ANGLE"], mu - confidence * sigma)
    range_max = min(ANGLE_CONFIG["MAX_ANGLE"], mu + confidence * sigma)
    return f"[{range_min:.1f}, {range_max:.1f}]"
