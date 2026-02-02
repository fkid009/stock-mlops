# Stock MLOps

주식 가격 움직임 예측(상승/하락) MLOps 파이프라인

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Services

- **Airflow**: http://localhost:8080 (admin/admin)
- **API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Web**: http://localhost:3000

## Pipeline

| DAG | Schedule | Description |
|-----|----------|-------------|
| weekly_train | Sunday 00:00 | 3개 모델 학습 및 최적 모델 선택 |
| daily_predict | Weekdays 09:00 | 일일 예측 생성 |
| drift_check | Daily 10:00 | 성능 드리프트 감지 |

## Models

- Logistic Regression (L1)
- LightGBM
- SVM (RBF kernel)

## Features

- return_1d, return_5d
- volume_ratio, high_low_range
- ma_5_20_ratio, rsi_14, volatility_20

자세한 내용은 [claude.md](claude.md) 참조
