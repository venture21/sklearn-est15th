# Human Activity Recognition Using Smartphones Dataset

## 데이터셋 개요

| 항목 | 내용 |
|------|------|
| **출처** | UCI Machine Learning Repository |
| **DOI** | 10.24432/C54S4K |
| **라이선스** | CC BY 4.0 (Creative Commons Attribution 4.0 International) |
| **데이터 유형** | Multivariate, Time-Series |
| **결측치** | 없음 |

---

## 데이터셋 설명

Human Activity Recognition (HAR) 데이터셋은 스마트폰에 내장된 관성 센서(가속도계, 자이로스코프)를 사용하여 인간의 일상 활동을 인식하기 위해 수집된 데이터입니다. 이 데이터셋은 웨어러블 컴퓨팅 및 모바일 헬스케어 분야에서 널리 사용되는 벤치마크 데이터셋입니다.

### 실험 설계

- **참가자**: 19-48세 사이의 30명의 자원봉사자
- **장치**: Samsung Galaxy S II 스마트폰 (허리에 착용)
- **센서**: 내장 가속도계(Accelerometer) 및 자이로스코프(Gyroscope)
- **샘플링 레이트**: 50Hz (초당 50회 측정)

---

## 데이터 규모

| 구분 | 값 |
|------|-----|
| **전체 샘플 수** | 10,299 |
| **특성(Feature) 수** | 561 |
| **클래스 수** | 6 |
| **훈련 데이터** | 7,352 (21명, 70%) |
| **테스트 데이터** | 2,947 (9명, 30%) |

---

## 타겟 클래스 (Activities)

| 클래스 ID | 활동 (영문) | 활동 (한글) | 설명 |
|-----------|-------------|-------------|------|
| 1 | WALKING | 걷기 | 평지에서 걷는 동작 |
| 2 | WALKING_UPSTAIRS | 계단 오르기 | 계단을 올라가는 동작 |
| 3 | WALKING_DOWNSTAIRS | 계단 내려가기 | 계단을 내려가는 동작 |
| 4 | SITTING | 앉기 | 의자에 앉아있는 상태 |
| 5 | STANDING | 서기 | 서있는 상태 |
| 6 | LAYING | 눕기 | 누워있는 상태 |

### 활동 유형 분류

- **동적 활동 (Dynamic)**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
- **정적 활동 (Static)**: SITTING, STANDING, LAYING

---

## 센서 데이터 및 신호 처리

### 원시 센서 데이터

| 센서 | 측정 값 | 축 |
|------|---------|-----|
| **가속도계 (Accelerometer)** | 선형 가속도 (Linear Acceleration) | X, Y, Z (3축) |
| **자이로스코프 (Gyroscope)** | 각속도 (Angular Velocity) | X, Y, Z (3축) |

### 신호 전처리

1. **노이즈 필터링**: 센서 노이즈 제거
2. **슬라이딩 윈도우**: 2.56초 (128 readings), 50% 오버랩
3. **버터워스 저역통과 필터**: 0.3Hz 차단 주파수
   - 중력 가속도(Gravitational Acceleration) 분리
   - 신체 가속도(Body Acceleration) 분리

### 가속도 신호 분해

```
Total Acceleration = Body Acceleration + Gravity Acceleration
```

- **Body Acceleration**: 신체 움직임에 의한 가속도
- **Gravity Acceleration**: 중력에 의한 가속도 (저주파 성분)

---

## 특성(Features) 상세 설명

### 특성 도메인

총 561개의 특성은 **시간 도메인(Time Domain)**과 **주파수 도메인(Frequency Domain)**으로 구분됩니다.

| 도메인 | 접두사 | 설명 |
|--------|--------|------|
| 시간 도메인 | `t` | 시간에 따른 신호 특성 |
| 주파수 도메인 | `f` | FFT 변환 후 주파수 특성 |

### 신호 유형

| 신호명 | 설명 |
|--------|------|
| `tBodyAcc-XYZ` | 시간 도메인 신체 가속도 (3축) |
| `tGravityAcc-XYZ` | 시간 도메인 중력 가속도 (3축) |
| `tBodyAccJerk-XYZ` | 시간 도메인 신체 가속도 Jerk (급격한 변화) |
| `tBodyGyro-XYZ` | 시간 도메인 각속도 (3축) |
| `tBodyGyroJerk-XYZ` | 시간 도메인 각속도 Jerk |
| `tBodyAccMag` | 신체 가속도 크기 (Magnitude) |
| `tGravityAccMag` | 중력 가속도 크기 |
| `tBodyAccJerkMag` | 신체 가속도 Jerk 크기 |
| `tBodyGyroMag` | 각속도 크기 |
| `tBodyGyroJerkMag` | 각속도 Jerk 크기 |
| `fBodyAcc-XYZ` | 주파수 도메인 신체 가속도 |
| `fBodyAccJerk-XYZ` | 주파수 도메인 신체 가속도 Jerk |
| `fBodyGyro-XYZ` | 주파수 도메인 각속도 |
| `fBodyAccMag` | 주파수 도메인 신체 가속도 크기 |
| `fBodyAccJerkMag` | 주파수 도메인 가속도 Jerk 크기 |
| `fBodyGyroMag` | 주파수 도메인 각속도 크기 |
| `fBodyGyroJerkMag` | 주파수 도메인 각속도 Jerk 크기 |

### 통계적 특성 (각 신호에 적용)

| 특성 | 설명 |
|------|------|
| `mean()` | 평균값 |
| `std()` | 표준편차 |
| `mad()` | 중앙절대편차 (Median Absolute Deviation) |
| `max()` | 최대값 |
| `min()` | 최소값 |
| `sma()` | 신호 크기 영역 (Signal Magnitude Area) |
| `energy()` | 에너지 (제곱합을 샘플 수로 나눈 값) |
| `iqr()` | 사분위수 범위 (Interquartile Range) |
| `entropy()` | 신호 엔트로피 |
| `arCoeff()` | 자기회귀 계수 (Autoregression Coefficients) |
| `correlation()` | 두 신호 간 상관계수 |
| `maxInds()` | 최대 크기를 가진 주파수 성분의 인덱스 |
| `meanFreq()` | 가중 평균 주파수 |
| `skewness()` | 주파수 도메인 왜도 |
| `kurtosis()` | 주파수 도메인 첨도 |
| `bandsEnergy()` | 주파수 대역별 에너지 |
| `angle()` | 벡터 간 각도 |

---

## 파일 구조

```
UCI HAR Dataset/
├── README.txt                    # 데이터셋 설명
├── features_info.txt             # 특성 정보 상세 설명
├── features.txt                  # 561개 특성 목록
├── activity_labels.txt           # 6개 활동 레이블
├── train/                        # 훈련 데이터 (70%)
│   ├── X_train.txt              # 훈련 특성 데이터 (7352 x 561)
│   ├── y_train.txt              # 훈련 레이블 (7352 x 1)
│   ├── subject_train.txt        # 훈련 참가자 ID (7352 x 1)
│   └── Inertial Signals/        # 원시 센서 데이터
│       ├── body_acc_x_train.txt
│       ├── body_acc_y_train.txt
│       ├── body_acc_z_train.txt
│       ├── body_gyro_x_train.txt
│       ├── body_gyro_y_train.txt
│       ├── body_gyro_z_train.txt
│       ├── total_acc_x_train.txt
│       ├── total_acc_y_train.txt
│       └── total_acc_z_train.txt
└── test/                         # 테스트 데이터 (30%)
    ├── X_test.txt               # 테스트 특성 데이터 (2947 x 561)
    ├── y_test.txt               # 테스트 레이블 (2947 x 1)
    ├── subject_test.txt         # 테스트 참가자 ID (2947 x 1)
    └── Inertial Signals/        # 원시 센서 데이터
        └── ...
```

---

## 활용 분야

### 1. 헬스케어 및 웰니스
- 노인 낙상 감지 시스템
- 재활 환자 활동 모니터링
- 일상 활동량 추적

### 2. 스마트 환경
- 스마트홈 자동화 (활동 기반 조명, 온도 조절)
- 컨텍스트 인식 서비스

### 3. 스포츠 및 피트니스
- 운동 자세 분석
- 칼로리 소모량 추정
- 트레이닝 최적화

### 4. 연구 및 벤치마크
- 분류 알고리즘 성능 비교
- 딥러닝 모델 평가
- 시계열 분석 연구

---

## 권장 분석 방법

### 머신러닝 모델
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)

### 딥러닝 모델
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- CNN-LSTM Hybrid
- Transformer

### 평가 지표
- Accuracy
- Precision, Recall, F1-Score (클래스별)
- Confusion Matrix
- Cross-Validation

---

## 데이터 로드 예시 (Python)

```python
import pandas as pd
import numpy as np

# 특성 이름 로드
features = pd.read_csv('features.txt', sep='\s+', header=None, names=['idx', 'name'])

# 훈련 데이터 로드
X_train = pd.read_csv('train/X_train.txt', sep='\s+', header=None, names=features['name'])
y_train = pd.read_csv('train/y_train.txt', sep='\s+', header=None, names=['activity'])
subject_train = pd.read_csv('train/subject_train.txt', sep='\s+', header=None, names=['subject'])

# 테스트 데이터 로드
X_test = pd.read_csv('test/X_test.txt', sep='\s+', header=None, names=features['name'])
y_test = pd.read_csv('test/y_test.txt', sep='\s+', header=None, names=['activity'])
subject_test = pd.read_csv('test/subject_test.txt', sep='\s+', header=None, names=['subject'])

# 활동 레이블
activity_labels = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

print(f"훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
```

---

## 인용 (Citation)

```bibtex
@misc{misc_human_activity_recognition_using_smartphones_240,
  author       = {Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge},
  title        = {{Human Activity Recognition Using Smartphones}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C54S4K}
}
```

### 관련 논문

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
> **A Public Domain Dataset for Human Activity Recognition Using Smartphones.**
> 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

---

## 참고 링크

- **UCI Repository**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Kaggle**: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

---

## 데이터셋 제작자

| 이름 | 소속 |
|------|------|
| Davide Anguita | Università degli Studi di Genova |
| Alessandro Ghio | Università degli Studi di Genova |
| Luca Oneto | Università degli Studi di Genova |
| Xavier Parra | CETpD - Universitat Politècnica de Catalunya |
| Jorge L. Reyes-Ortiz | CETpD - Universitat Politècnica de Catalunya |
