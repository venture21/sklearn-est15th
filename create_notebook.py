import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AutoGluon Ensemble: GBM, LGBM, XGBoost, CatBoost, RF, Voting, Stacking\n",
    "\n",
    "이 노트북은 AutoGluon을 사용하여 다양한 회귀 모델을 학습하고 앙상블(Voting, Stacking) 성능을 테스트합니다.\n",
    "참조: `Plus_7_ensemble.ipynb`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# AutoGluon 설치 (필요한 경우 주석 해제 후 실행)\n",
    "# !pip install autogluon"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from autogluon.tabular import TabularDataset, TabularPredictor\n",
    "\n",
    "np.set_printoptions(precision=4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. 데이터 로드 및 전처리"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = fetch_california_housing()\n",
    "df = pd.DataFrame(data.data, columns=data.feature_names)\n",
    "df['MedHouseVal'] = data.target\n",
    "\n",
    "print(\"데이터 크기:\", df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 학습/테스트 데이터 분리\n",
    "train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)\n",
    "\n",
    "print(\"Train size:\", train_data.shape)\n",
    "print(\"Test size:\", test_data.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. 모델 학습 설정 (Hyperparameters & Ensemble)\n",
    "\n",
    "- **사용 모델**: GBM (LightGBM), XGB (XGBoost), CAT (CatBoost), RF (Random Forest)\n",
    "- **앙상블 전략**:\n",
    "    - **Voting (Weighted Ensemble)**: AutoGluon은 기본적으로 성능이 좋은 모델들의 가중 평균(Weighted Ensemble)을 생성합니다.\n",
    "    - **Stacking**: `num_bag_folds`를 설정하여 K-Fold Bagging 및 Stacking을 수행합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "label = 'MedHouseVal'\n",
    "eval_metric = 'mean_squared_error'  # 회귀 문제 평가 지표\n",
    "\n",
    "# 1. 특정 모델 지정 (GBM, XGB, CAT, RF)\n",
    "# AutoGluon에서 'GBM'은 LightGBM을 의미합니다.\n",
    "hyperparameters = {\n",
    "    'GBM': {}, \n",
    "    'XGB': {}, \n",
    "    'CAT': {}, \n",
    "    'RF': {}\n",
    "}\n",
    "\n",
    "# 2. Stacking 설정\n",
    "# num_bag_folds >= 2 이면 스태킹(Stacking)이 활성화됩니다.\n",
    "num_bag_folds = 5  \n",
    "num_stack_levels = 1 # 스태킹 레이어 수 (0이면 Bagging만, 1 이상이면 Stacking)\n",
    "\n",
    "save_path = 'ag_models_california_housing'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. 학습 실행"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path).fit(\n",
    "    train_data,\n",
    "    hyperparameters=hyperparameters,\n",
    "    num_bag_folds=num_bag_folds,\n",
    "    num_stack_levels=num_stack_levels,\n",
    "    fit_weighted_ensemble=True  # Weighted Ensemble (Voting 유사 효과) 활성화\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. 성능 평가 및 리더보드 확인\n",
    "\n",
    "학습된 모델들의 성능을 비교합니다. \n",
    "- `WeightedEnsemble_L2` (또는 L3): 앙상블 모델 (Voting 역할)\n",
    "- `LightGBM_BAG_L1` 등: Bagging/Stacking된 개별 모델들"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 리더보드 출력 (테스트 데이터 기준 평가)\n",
    "leaderboard = predictor.leaderboard(test_data, silent=False)\n",
    "\n",
    "print(\"\\n--- Top Models ---\")\n",
    "print(leaderboard.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. 최종 평가\n",
    "\n",
    "테스트 데이터 전체에 대한 최종 평가 점수(MSE)를 확인합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = predictor.evaluate(test_data)\n",
    "print(\"Final Test Results:\", results)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. 예측 결과 확인"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = predictor.predict(test_data)\n",
    "print(\"First 5 predictions:\\n\", y_pred.head())\n",
    "print(\"\\nFirst 5 actual values:\\n\", test_data[label].head())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

file_path = "c:\\Users\\park0\\iCloudDrive\\2025\\EST_AI_MODEL_DEVEL\\ESTSoft\\DataScience\\scikit-learn\\Plus_7_ensemble_gemini.ipynb"
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"Notebook created at: {file_path}")
