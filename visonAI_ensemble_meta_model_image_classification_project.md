# 프로젝트: 앙상블 기반 메타 모델을 이용한 이미지 분류 성능 향상 시도

## 1. 프로젝트 개요
- **목표**: 여러 기본 인식 모델(YOLOv8, ResNet50 등)을 앙상블하여 이미지 분류 성능을 향상
- **사용 기술**:
  - **프레임워크**: PyTorch
  - **모델**: YOLOv8n, YOLOv8s, ResNet50 (두 개 버전)
  - **기타 라이브러리**: Ultralytics YOLO, Torchvision, Scikit-learn, NumPy, JSON

## 2. 환경 설정 및 하이퍼파라미터
- **데이터 디렉토리**: `dataset_classify/`
- **모델 파일 경로**:
  - `models/path_to_yolov8n_cls.pth`
  - `models/path_to_yolov8s_cls.pth`
  - `models/path_to_resnet50_1.pth`
  - `models/path_to_resnet50_2.pth`
- **하이퍼파라미터**:
  - 배치 크기: 32
  - 메타 모델 학습 에포크: 20
  - 학습률: 0.001
  - 랜덤 시드: 42
  - 디바이스 설정: GPU 사용 가능 시 GPU 사용, 아니면 CPU

## 3. 데이터 로드 및 전처리
- **전처리 변환**:
  - 텐서 변환
  - 정규화: 평균 [0.485, 0.456, 0.406], 표준편차 [0.229, 0.224, 0.225]
- **데이터셋**:
  - 훈련: `dataset_classify/train`
  - 검증: `dataset_classify/val`
  - 테스트: `dataset_classify/test`
- **DataLoader 설정**:
  - 배치 크기: 32
  - 셔플: 훈련 데이터만 True
  - 워커 수: 4

## 4. 기본 모델 로드
- **모델 종류**:
  - ResNet50: 커스텀 클래스 수에 맞게 최종 계층 수정
  - YOLOv8n, YOLOv8s: Ultralytics YOLO 라이브러리 사용
- **모델 로드 함수**:
  - 모델 타입에 따라 적절한 로딩 방식 적용
  - 모델을 디바이스로 이동
  - 평가 모드로 설정 (ResNet50)

## 5. 기본 모델 예측 수집
- **예측 수집 함수**:
  - 각 기본 모델에 대해 데이터셋 전체에 대한 예측 확률 수집
  - 소프트맥스 적용하여 확률 계산
  - NumPy 배열로 변환 후 저장
- **메타 피처 생성**:
  - 기본 모델의 예측 확률을 수평으로 결합
  - 훈련, 검증, 테스트 세트 각각에 대해 메타 피처 생성
- **메타 피처 저장**:
  - JSON 형식으로 `X_meta_train.json`, `X_meta_val.json`, `X_meta_test.json` 저장

## 6. 메타 모델 정의 및 학습
- **메타 모델 아키텍처**:
  - 간단한 신경망 (입력층, 은닉층, 출력층)
  - 활성화 함수: ReLU
- **학습 설정**:
  - 손실 함수: CrossEntropyLoss
  - 최적화 알고리즘: Adam
  - 학습 데이터: 메타 피처 및 레이블
- **학습 과정**:
  - 에포크별 손실 출력
  - 학습 모드 활성화
  - 역전파 및 파라미터 업데이트

## 7. 메타 모델 평가
- **평가 함수**:
  - 모델을 평가 모드로 설정
  - 예측 결과와 실제 레이블 비교하여 정확도 계산
- **평가 결과**:
  - 검증 정확도 출력
  - 테스트 정확도 출력

## 8. 전체 프로세스 요약
- **데이터 로드 및 전처리**:
  - ImageFolder를 사용하여 데이터셋 로드
  - transforms 적용하여 이미지 전처리
  - DataLoader로 배치 단위로 데이터 준비
- **기본 모델 로드**:
  - 사전 학습된 네 개의 기본 모델(YOLOv8n, YOLOv8s, ResNet50_1, ResNet50_2) 로드
- **기본 모델 예측 수집**:
  - 각 기본 모델로부터 예측 확률 수집
  - 메타 피처로 결합
- **메타 모델 정의 및 학습**:
  - 신경망 기반 메타 모델 정의
  - 메타 피처를 사용하여 메타 모델 학습
- **메타 모델 평가**:
  - 검증 및 테스트 세트에 대한 정확도 평가
- **결과 출력**:
  - 최종 검증 및 테스트 정확도 출력

## 9. 문제점 및 개선 사항
- **시간 부족으로 인한 실패**:
  - 여러 모델을 앙상블하여 메타 모델을 학습하는 과정에서 시간 부족으로 프로젝트를 완수하지 못함
- **개선 방안**:
  - 모델 로드 및 예측 수집 과정을 최적화
  - 메타 모델의 복잡도 조절 및 하이퍼파라미터 튜닝
  - 병렬 처리 활용하여 예측 속도 향상
  - 다양한 메타 모델 실험 (예: Random Forest, Gradient Boosting)

## 10. 추가 고려 사항
- **모델 아키텍처**:
  - 실제 YOLOv8 모델의 정확한 아키텍처와 로드 방식 반영 필요
- **데이터 전처리**:
  - 모델 학습 시 동일한 전처리 과정 적용
- **모델 저장 형식**:
  - `torch.save(model.state_dict(), PATH)` 형식 사용 여부 확인
- **오버피팅 방지**:
  - 메타 모델의 복잡도 조절 및 정규화 기법 적용
- **교차 검증**:
  - 교차 검증 도입으로 메타 모델의 견고성 강화
- **다양한 메타 모델 실험**:
  - 신경망 외 다른 머신러닝 모델과의 성능 비교

## 11. 참고 자료
- 스테킹 앙상블: [Stacking - Wikipedia](https://en.wikipedia.org/wiki/Stacking)
- PyTorch 공식 문서: [PyTorch Documentation](https://pytorch.org/docs/)
- TorchVision 공식 문서: [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- Scikit-learn 공식 문서: [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

## 프로젝트 요약
여러 기본 인식 모델을 앙상블하여 메타 모델을 통해 분류 성능을 향상시키려는 시도를 했으나, 시간 부족으로 인해 완성하지 못함. 향후 개선 사항을 반영하여 프로젝트를 재개할 계획.
