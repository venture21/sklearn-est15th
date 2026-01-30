# 서울시 따릉이 공유 자전거 수요 예측

서울시 공유 자전거(따릉이)의 시간대별 대여 수요를 예측하는 머신러닝 프로젝트입니다.

## 프로젝트 개요

기상 데이터(온도, 습도, 강수량, 미세먼지 등)와 시간 정보를 활용하여 자전거 대여 수를 예측합니다. AutoGluon과 H2O AutoML 두 가지 AutoML 프레임워크를 사용하여 모델을 구축하고 성능을 비교합니다.

## 디렉토리 구조

```
seoulBike/
├── data/
│   ├── train.csv           # 학습 데이터
│   ├── test.csv            # 테스트 데이터
│   └── submission.csv      # 제출 양식
├── seoulbike_with_autogluon.ipynb  # AutoGluon 모델링 노트북
├── seoulbike_with_h2o.ipynb        # H2O AutoML 모델링 노트북
├── submission_autogluon.csv        # AutoGluon 예측 결과 (생성됨)
├── submission_h2o.csv              # H2O 예측 결과 (생성됨)
└── README.md
```

## 데이터 설명

### 입력 변수 (Features)

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `id` | 고유 식별자 | - |
| `hour` | 시간 | 0-23 |
| `hour_bef_temperature` | 1시간 전 기온 | °C |
| `hour_bef_precipitation` | 1시간 전 강수 여부 | 0(없음), 1(있음) |
| `hour_bef_windspeed` | 1시간 전 풍속 | m/s |
| `hour_bef_humidity` | 1시간 전 습도 | % |
| `hour_bef_visibility` | 1시간 전 시정 | m |
| `hour_bef_ozone` | 1시간 전 오존 농도 | ppm |
| `hour_bef_pm10` | 1시간 전 미세먼지(PM10) | μg/m³ |
| `hour_bef_pm2.5` | 1시간 전 초미세먼지(PM2.5) | μg/m³ |

### 타겟 변수 (Target)

| 컬럼명 | 설명 |
|--------|------|
| `count` | 자전거 대여 수 |

## 노트북 구성

두 노트북 모두 동일한 구조로 구성되어 있습니다:

### 1. 데이터 로드 및 탐색
- Train/Test 데이터 로드
- 데이터 크기 및 타입 확인
- 결측치 현황 파악

### 2. 컬럼별 데이터 분석 및 시각화
- **타겟 변수 분석**: 대여 수 분포, 왜도/첨도 확인
- **시간 분석**: 시간대별 평균 대여 수, 박스플롯
- **기상 변수 분석**: 각 변수의 분포 및 대여 수와의 관계
- **상관관계 분석**: 히트맵을 통한 변수 간 상관관계 시각화
- **강수량 분석**: 비 유무에 따른 대여 수 차이

### 3. 전처리
- **결측치 처리**: 시간별 평균값으로 대체
- **이상치 분석**: IQR 방식으로 이상치 탐지

### 4. 특성 엔지니어링

새로 생성되는 특성들:

| 특성명 | 설명 |
|--------|------|
| `is_rush_hour` | 출퇴근 시간 여부 (7-9시, 18-20시) |
| `is_daytime` | 낮/밤 구분 (6-18시: 낮) |
| `time_period` | 시간대 구분 (0:새벽, 1:아침, 2:점심, 3:오후, 4:저녁, 5:밤) |
| `hour_sin` | 시간의 sin 변환 (주기성 반영) |
| `hour_cos` | 시간의 cos 변환 (주기성 반영) |
| `feels_like` | 체감온도 (온도, 습도 고려) |
| `discomfort_index` | 불쾌지수 |
| `good_weather` | 날씨 좋음 지표 (비X, 적정 온도, 낮은 미세먼지) |
| `air_quality_index` | 대기질 지수 (PM10, PM2.5 가중 평균) |
| `visibility_group` | 시정 구간화 (0-3) |
| `temp_group` | 온도 구간화 (0-4) |
| `temp_humidity` | 온도 × 습도 상호작용 |
| `temp_wind` | 온도 × 풍속 상호작용 |

### 5. 모델링

#### AutoGluon (`seoulbike_with_autogluon.ipynb`)
- `TabularPredictor` 사용
- `best_quality` 프리셋으로 최고 품질 모델 학습
- 자동 모델 선택 및 하이퍼파라미터 튜닝

#### H2O AutoML (`seoulbike_with_h2o.ipynb`)
- `H2OAutoML` 사용
- GBM, XGBoost, DRF, GLM, StackedEnsemble 알고리즘 포함
- 최대 20개 모델 학습

### 6. 앙상블
- 내장 앙상블 모델 분석
- 개별 모델 성능 비교
- 기본 모델 기여도 확인

### 7. 모델 비교 평가
- 실제값 vs 예측값 산점도
- 잔차 분포 분석
- 시간대별 잔차 분석
- 평가 지표: RMSE, MAE, R² Score

### 8. submission.csv 파일 생성
- 테스트 데이터 예측
- 음수 예측값 처리
- CSV 파일 저장

## 설치 및 실행

### 필수 라이브러리

```bash
# 기본 라이브러리
pip install pandas numpy matplotlib seaborn scikit-learn

# AutoGluon
pip install autogluon

# H2O
pip install h2o
```

### 실행 방법

1. Jupyter Notebook 실행
```bash
jupyter notebook
```

2. 원하는 노트북 파일 열기
   - `seoulbike_with_autogluon.ipynb` (AutoGluon 사용)
   - `seoulbike_with_h2o.ipynb` (H2O 사용)

3. 셀 순서대로 실행

## 평가 지표

| 지표 | 설명 |
|------|------|
| **RMSE** | Root Mean Squared Error - 예측 오차의 제곱근 평균 |
| **MAE** | Mean Absolute Error - 예측 오차의 절대값 평균 |
| **R² Score** | 결정계수 - 모델의 설명력 (1에 가까울수록 좋음) |

## 참고 사항

- 모델 학습 시간은 약 10분으로 설정되어 있습니다.
- H2O 노트북 실행 시 Java Runtime Environment(JRE)가 필요합니다.
- GPU가 있는 경우 AutoGluon이 자동으로 활용합니다.
- 결과 재현을 위해 `random_state=42`를 사용합니다.

## 출력 파일

| 파일명 | 설명 |
|--------|------|
| `submission_autogluon.csv` | AutoGluon 모델 예측 결과 |
| `submission_h2o.csv` | H2O 모델 예측 결과 |
