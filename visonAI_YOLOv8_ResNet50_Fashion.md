
# 패션 룩 인식 및 객체 감지를 위한 YOLOv8 및 ResNet50 기반 시스템

본 프로젝트는 YOLOv8을 활용한 객체 감지와 ResNet50을 이용한 세부 이미지 분석을 통해 다양한 패션 룩을 인식하는 시스템을 구축하는 것을 목표로 합니다.
## 프로젝트 개요

- **목적**: 다양한 패션 룩을 자동으로 인식하고 분류하여 패션 트렌드 분석 및 개인화된 스타일 추천 시스템 개발.
- **주요 기능**:
  - 객체 감지를 통한 인물 및 패션 아이템 식별
  - 이미지 분류를 통한 패션 룩의 세부 분류

## 사용 기술 및 도구

- **프레임워크**: PyTorch
- **모델**:
  - Ultralytics YOLOv8 (객체 감지)
  - ResNet50 (이미지 분류)
- **라이브러리**:
  - torchvision (데이터 전처리 및 모델 로딩)
  - sklearn (평가 지표 계산)
  - seaborn & matplotlib (데이터 시각화)
- **개발 환경**: CUDA 지원 GPU (학습 속도 향상)

## 데이터셋 구성

- **디렉토리 구조**:
  - `train/images`: 학습용 이미지
  - `valid/images`: 검증용 이미지
  - `test/images`: 테스트용 이미지
- **클래스 수**: 7개
- **클래스명**: ['casual_look', 'classic_look', 'minimal_look','chic_look', 'sporty_look', 'street_look', 'person']
- **데이터 형식**: ImageFolder 형식으로 구성하여 각 클래스별 폴더에 이미지 저장

## YOLOv8을 이용한 객체 감지

- **모델 로드 및 설정**:
  - `YOLO('yolov8n.pt')`을 통해 사전 학습된 YOLOv8 모델 로드
  - GPU 사용 가능 시 모델을 GPU로 이동 (`model.to(device)`)
- **학습 설정**:
  - 데이터 경로: `data.yaml` 파일 지정
  - 학습 파라미터: 에폭 500, 이미지 크기 640, 배치 크기 16, 학습률 0.01 등 설정
- **학습 과정**:
  - 객체 감지를 통해 이미지 내 인물 및 패션 아이템 식별
  - 학습 과정 중 배치마다 손실 출력 및 진행 상황 모니터링

## ResNet50을 이용한 이미지 분류

- **데이터 전처리**:
  - `transforms.Compose`를 사용하여 이미지 텐서 변환 및 정규화
  - ImageNet 데이터셋의 평균 및 표준편차를 사용하여 정규화
- **모델 설정**:
  - 사전 학습된 ResNet50 모델 로드 (`resnet50(weights=ResNet50_Weights.DEFAULT)`)
  - 마지막 완전 연결층 수정: 드롭아웃 적용 후 클래스 수에 맞게 출력 변경
  - 모델을 GPU로 이동 (`model.to(device)`)
- **손실 함수 및 옵티마이저**:
  - 손실 함수: `CrossEntropyLoss`
  - 옵티마이저: `Adam` (학습률 0.0001, L2 정규화 0.001)
- **조기 종료(Early Stopping)**:
  - 검증 손실이 개선되지 않을 경우 학습 중단
  - `patience`를 10으로 설정하여 10 에폭 동안 개선이 없으면 학습 종료

## 모델 학습 및 검증

- **학습 함수 (train_model)**:
  - 지정된 에폭 동안 학습 및 검증 단계 반복
  - 각 배치마다 손실 및 정확도 계산
  - 조기 종료 조건 모니터링
  - 최고의 검증 정확도를 기록하고 모델 저장
- **Fine-tuning**:
  - 초기에는 마지막 레이어만 학습
  - 필요 시 전체 네트워크를 fine-tuning하여 성능 향상

## 모델 평가 및 시각화

- **테스트 데이터 평가**:
  - 테스트 데이터셋 로드 및 예측 수행
  - 실제 레이블과 예측 레이블 비교
- **평가 지표**:
  - 혼동 행렬 (Confusion Matrix)
  - 정밀도-재현율 곡선 (Precision-Recall Curve)
  - ROC 곡선 및 AUC (ROC Curve & AUC)
  - F1 Score 및 Classification Report
- **시각화 및 저장**:
  - seaborn과 matplotlib을 이용해 각 평가 지표 시각화
  - 모든 결과는 지정된 `answers` 폴더에 저장

## 결과 저장 및 활용

- **모델 저장**:
  - 최종 모델의 가중치를 `resnet50_finetuned.pth` 파일로 저장
- **시각화 파일 저장**:
  - 혼동 행렬, 정밀도-재현율 곡선, ROC 곡선, 분류 리포트 등 시각화 파일을 `answers` 폴더에 저장
- **활용 방안**:
  - 저장된 모델을 이용한 실시간 패션 룩 인식
  - 시각화된 평가 지표를 통해 모델 성능 분석 및 개선

## 실행 방법

- **환경 설정**:
  - Python 3.8 이상 설치
- **필요한 라이브러리 설치**:
  ```bash
  pip install torch torchvision ultralytics sklearn seaborn matplotlib
  ```
- **데이터 준비**:
  - 데이터셋을 지정된 디렉토리 구조에 맞게 배치
- **학습 실행**:
  - 메인 스크립트 실행:
    ```bash
    python main.py
    ```
- **결과 확인**:
  - `answers` 폴더에서 평가 지표 및 시각화 결과 확인

## 참고 자료

- Ultralytics YOLOv8 Documentation
- PyTorch Documentation
- ResNet50 Paper
- Scikit-learn Metrics