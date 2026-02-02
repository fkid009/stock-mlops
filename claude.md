# Stock MLOps Project

주식 가격 움직임 예측(상승/하락)을 위한 MLOps 파이프라인

## 구조

```
stock-mlops/
├── src/                    # 핵심 ML 코드
│   ├── common/            # 설정, 로깅
│   ├── data/              # 데이터 수집, 캐시, DB
│   ├── features/          # 피처 엔지니어링
│   ├── models/            # 학습, 예측, MLflow 연동
│   └── evaluation/        # 검증, 드리프트 감지
├── airflow/dags/          # Airflow DAG
│   ├── weekly_train.py    # 주간 학습 (일요일)
│   ├── daily_predict.py   # 일일 예측 (평일)
│   └── drift_check.py     # 드리프트 체크 (매일)
├── api/                   # FastAPI 백엔드
├── web/                   # Next.js 프론트엔드
├── configs/               # YAML 설정 파일
└── docker-compose.yml     # Docker 구성
```

## 서버 설정 (Ubuntu)

### 1. Docker 설치

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Docker GPG 키
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Docker 저장소
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 권한 (재로그인 필요)
sudo usermod -aG docker $USER
```

### 2. 프로젝트 클론

```bash
git clone https://github.com/fkid009/stock-mlops.git
cd stock-mlops
```

### 3. 환경 설정

```bash
cp .env.example .env
# 필요시 .env 수정
```

### 4. 실행

```bash
docker compose up -d
```

## 서비스 URL

| 서비스 | URL | 비고 |
|--------|-----|------|
| Airflow | http://localhost:8080 | admin / admin |
| FastAPI | http://localhost:8000/docs | Swagger UI |
| MLflow | http://localhost:5000 | 실험 추적 |
| Web | http://localhost:3000 | 대시보드 |

## 파이프라인

### Weekly Training (일요일 00:00)
1. 데이터 수집 (150일)
2. 피처 계산 + 스케일러 fit
3. 3개 모델 학습 (Logistic L1, LightGBM, SVM)
4. 최적 모델 선택 → MLflow 등록

### Daily Prediction (평일 09:00)
1. 최신 데이터 수집
2. 피처 계산 + 스케일링
3. 예측 생성 → DB 저장
4. 전일 예측 정확도 업데이트

### Drift Detection (매일 10:00)
- 최근 5일 vs 이전 30일 정확도 비교
- 5% 이상 하락 시 튜닝 트리거

## 모델

| 모델 | 설명 |
|------|------|
| logistic_l1 | L1 정규화 로지스틱 회귀 |
| lightgbm | Gradient Boosting |
| svm | RBF 커널 SVM |

선택 기준: `0.7 * accuracy + 0.3 * stability`

## 피처 (7개)

- return_1d: 1일 수익률
- return_5d: 5일 수익률
- volume_ratio: 거래량 비율 (20일 평균 대비)
- high_low_range: 고저 변동폭
- ma_5_20_ratio: 5일/20일 이동평균 비율
- rsi_14: RSI 14일
- volatility_20: 20일 변동성

## 개발 명령어

```bash
# 로그 확인
docker compose logs -f airflow-webserver
docker compose logs -f api

# 재시작
docker compose restart api

# 전체 중지
docker compose down

# 볼륨 포함 삭제
docker compose down -v
```

## 로컬 개발

```bash
# Python 환경
pip install -e .

# API 실행
uvicorn api.main:app --reload

# 웹 실행
cd web && npm install && npm run dev
```
