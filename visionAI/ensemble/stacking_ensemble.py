import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import numpy as np  
import os
import json

# ----------------------------
# 2. 설정 및 하이퍼파라미터
# ----------------------------

# 데이터 디렉토리 경로 설정
DATA_DIR = 'dataset_classify/'

# 기본 모델 파일 경로
MODEL_YOLOV8N_PATH = 'models/path_to_yolov8n_cls.pth'
MODEL_YOLOV8S_PATH = 'models/path_to_yolov8s_cls.pth'
MODEL_RESNET50_1_PATH = 'models/path_to_resnet50_1.pth'
MODEL_RESNET50_2_PATH = 'models/path_to_resnet50_2.pth'

# 하이퍼파라미터 설정
BATCH_SIZE = 32  # 배치 크기
META_MODEL_EPOCHS = 20  # 메타 모델 학습 에포크 수
LEARNING_RATE = 0.001  # 메타 모델 학습률
RANDOM_STATE = 42  # 랜덤 시드

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {DEVICE}")

# ----------------------------
# 3. 데이터 로드 및 전처리
# ----------------------------

# 이미지 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# PyTorch의 ImageFolder를 사용하여 데이터 로드
train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')
test_dir = os.path.join(DATA_DIR, 'test')

# ImageFolder는 폴더 구조를 기반으로 자동으로 레이블을 할당
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 클래스 수 설정 (데이터셋에 맞게 수정)
NUM_CLASSES = len(train_dataset.classes)

# ----------------------------
# 4. 기본 모델 로드
# ----------------------------

def load_base_model(model_path, model_type):
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    elif model_type =='yolov8s':
        model = YOLO('yolov8s-cls.yaml').load(model_path)
    elif model_type == 'yolov8n':
        model = YOLO('yolov8n-cls.yaml').load(model_path)
    else:
        raise ValueError("Unsupported model type")

    model = model.to(DEVICE)
    if model_type == 'resnet50':
        model.eval()
    return model

# 사전 학습된 네 개의 기본 모델 로드
model_yolov8n_cls = load_base_model(MODEL_YOLOV8N_PATH, model_type='yolov8n')
model_yolov8s_cls = load_base_model(MODEL_YOLOV8S_PATH, model_type='yolov8s')
model_resnet50_1 = load_base_model(MODEL_RESNET50_1_PATH, model_type='resnet50')
model_resnet50_2 = load_base_model(MODEL_RESNET50_2_PATH, model_type='resnet50')

# 기본 모델들을 리스트로 관리
base_models = [
    model_yolov8n_cls,
    model_yolov8s_cls,
    model_resnet50_1,
    model_resnet50_2
]

# ----------------------------
# 5. 기본 모델 예측 수집
# ----------------------------

def get_model_predictions(models, data_loader):
    predictions = []  # 예측 저장 리스트
    for model in models:
        model_preds = []  # 현재 모델의 예측 저장 리스트
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for inputs, _ in data_loader:
                inputs = inputs.to(DEVICE)  # 입력 데이터를 디바이스로 이동
                outputs = model(inputs)  # 모델 예측
                probs = F.softmax(outputs, dim=1)  # 소프트맥스 적용하여 확률 계산
                model_preds.append(probs.cpu().numpy())  # CPU로 이동 후 NumPy 배열로 변환하여 저장
        model_preds = np.concatenate(model_preds, axis=0)  # 모든 배치의 예측을 하나로 결합
        predictions.append(model_preds)  # 현재 모델의 예측을 전체 리스트에 추가
    return predictions  # 모든 모델의 예측 리스트 반환

train_preds = get_model_predictions(base_models, train_loader)
val_preds = get_model_predictions(base_models, val_loader)
test_preds = get_model_predictions(base_models, test_loader)

# 기본 모델 예측을 메타 피처로 결합 (수평으로 이어 붙임)
X_meta_train = np.hstack(train_preds)
X_meta_val = np.hstack(val_preds)
X_meta_test = np.hstack(test_preds)

print(f"메타피처 갯수 확인 - Train: {X_meta_train.shape}, Val: {X_meta_val.shape}, Test: {X_meta_test.shape}")

# ----------------------------
# 메타 피처를 JSON 형식으로 저장
# ----------------------------

# NumPy 배열을 리스트로 변환
X_meta_train_list = X_meta_train.tolist()
X_meta_val_list = X_meta_val.tolist()
X_meta_test_list = X_meta_test.tolist()

# JSON 파일로 저장
with open('X_meta_train.json', 'w') as f:
    json.dump(X_meta_train_list, f)
with open('X_meta_val.json', 'w') as f:
    json.dump(X_meta_val_list, f)
with open('X_meta_test.json', 'w') as f:
    json.dump(X_meta_test_list, f)

print("메타 피처가 JSON 형식으로 저장되었습니다.")



# ----------------------------
# 6. 메타 모델 정의 및 학습
# ----------------------------

# 메타 모델을 위한 신경망 클래스 정의
class MetaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        메타 모델 정의 (간단한 신경망).

        Args:
            input_dim (int): 입력 피처의 차원 수
            hidden_dim (int): 은닉층의 차원 수
            output_dim (int): 출력 클래스 수
        """
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 첫 번째 선형 계층
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 두 번째 선형 계층

    def forward(self, x):
        """
        순전파 정의.

        Args:
            x (torch.Tensor): 입력 데이터

        Returns:
            torch.Tensor: 모델 출력
        """
        out = self.fc1(x)  # 첫 번째 선형 계층 통과
        out = self.relu(out)  # ReLU 활성화
        out = self.fc2(out)  # 두 번째 선형 계층 통과
        return out  # 최종 출력 반환

# 메타 모델 인스턴스 생성
meta_input_dim = X_meta_train.shape[1]  # 메타 피처의 차원 수
meta_hidden_dim = 128  # 은닉층의 차원 수 (하이퍼파라미터)
meta_output_dim = NUM_CLASSES  # 출력 클래스 수

meta_model = MetaModel(input_dim=meta_input_dim,
                       hidden_dim=meta_hidden_dim,
                       output_dim=meta_output_dim)
meta_model = meta_model.to(DEVICE)

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류용 손실 함수
optimizer = torch.optim.Adam(meta_model.parameters(), lr=LEARNING_RATE)  # Adam 옵티마이저

# 메타 모델 학습을 위한 데이터 준비
# NumPy 배열을 PyTorch 텐서로 변환
X_meta_train_tensor = torch.tensor(X_meta_train).float().to(DEVICE)  # 학습 메타 피처
y_train_tensor = torch.tensor([label for _, label in train_dataset.samples]).long().to(DEVICE)  # 학습 레이블

X_meta_val_tensor = torch.tensor(X_meta_val).float().to(DEVICE)  # 검증 메타 피처
y_val_tensor = torch.tensor([label for _, label in val_dataset.samples]).long().to(DEVICE)  # 검증 레이블

X_meta_test_tensor = torch.tensor(X_meta_test).float().to(DEVICE)  # 테스트 메타 피처
y_test_tensor = torch.tensor([label for _, label in test_dataset.samples]).long().to(DEVICE)  # 테스트 레이블

# 메타 데이터셋 및 DataLoader 정의
meta_train_dataset = torch.utils.data.TensorDataset(X_meta_train_tensor, y_train_tensor)
meta_val_dataset = torch.utils.data.TensorDataset(X_meta_val_tensor, y_val_tensor)
meta_test_dataset = torch.utils.data.TensorDataset(X_meta_test_tensor, y_test_tensor)

meta_train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
meta_val_loader = DataLoader(meta_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
meta_test_loader = DataLoader(meta_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 메타 모델 학습 함수 정의
def train_meta_model(model, loader, criterion, optimizer, epochs):
    """
    메타 모델을 학습합니다.

    Args:
        model (nn.Module): 학습할 메타 모델
        loader (DataLoader): 학습 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        epochs (int): 학습 에포크 수
    """
    model.train()  # 학습 모드로 설정
    for epoch in range(epochs):
        running_loss = 0.0  # 에포크 손실 누적 변수
        for inputs, labels in loader:
            optimizer.zero_grad()  # 기울기 초기화
            outputs = model(inputs)  # 메타 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 파라미터 업데이트
            running_loss += loss.item() * inputs.size(0)  # 배치 손실 누적
        epoch_loss = running_loss / len(loader.dataset)  # 에포크 당 평균 손실 계산
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")  # 손실 출력

# 메타 모델 학습
print("Training meta model...")
train_meta_model(meta_model, meta_train_loader, criterion, optimizer, META_MODEL_EPOCHS)

# ----------------------------
# 7. 메타 모델 평가
# ----------------------------

def evaluate_model(model, loader):
    """
    주어진 모델과 데이터 로더를 사용하여 정확도를 평가합니다.

    Args:
        model (nn.Module): 평가할 모델
        loader (DataLoader): 평가 데이터 로더

    Returns:
        float: 정확도
    """
    model.eval()  # 평가 모드로 설정
    correct = 0  # 올바르게 예측한 샘플 수
    total = 0  # 전체 샘플 수
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, labels in loader:
            outputs = model(inputs)  # 모델 예측
            _, preds = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 선택
            correct += (preds == labels).sum().item()  # 올바른 예측 수 누적
            total += labels.size(0)  # 전체 샘플 수 누적
    accuracy = correct / total  # 정확도 계산
    return accuracy  # 정확도 반환

# 검증 데이터에 대한 정확도 평가
val_accuracy = evaluate_model(meta_model, meta_val_loader)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 테스트 데이터에 대한 정확도 평가
test_accuracy = evaluate_model(meta_model, meta_test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# ----------------------------
# 8. 전체 프로세스 요약
# ----------------------------

"""
프로세스 요약:

1. 데이터 로드 및 전처리:
   - PyTorch의 ImageFolder를 사용하여 train, val, test 데이터를 로드합니다.
   - 이미지 전처리를 위해 transforms를 적용합니다.
   - DataLoader를 사용하여 데이터를 배치 단위로 로드할 수 있도록 준비합니다.

2. 기본 모델 로드:
   - 사전 학습된 네 개의 기본 모델(Yolov8n-cls, Yolov8s-cls, ResNet50_1, ResNet50_2)을 PyTorch로 로드합니다.
   - 각 모델은 동일한 입력 및 출력 차원을 가져야 합니다.

3. 기본 모델 예측 수집:
   - 각 기본 모델을 사용하여 학습, 검증, 테스트 세트에 대한 예측 확률을 수집합니다.
   - 소프트맥스를 적용하여 클래스별 확률을 계산합니다.
   - 예측 확률을 수평으로 결합하여 메타 피처를 생성합니다.

4. 메타 모델 정의 및 학습:
   - PyTorch를 사용하여 간단한 신경망 메타 모델을 정의합니다.
   - 메타 모델을 학습 데이터의 메타 피처로 학습시킵니다.
   - 검증 데이터를 사용하여 학습 과정을 모니터링합니다.

5. 메타 모델 평가:
   - 학습된 메타 모델을 사용하여 검증 및 테스트 세트에 대한 정확도를 평가합니다.

6. 결과 출력:
   - 최종 검증 정확도와 테스트 정확도를 출력합니다.
"""

# ----------------------------
# 9. 추가 고려 사항
# ----------------------------

"""
- **모델 아키텍처**: 실제 Yolov8 모델을 사용하려면, 해당 모델의 정확한 아키텍처와 로드 방식을 반영해야 합니다. 여기서는 예시로 ResNet50을 사용했지만, 실제 Yolov8 모델로 대체해야 합니다.
- **데이터 전처리**: 입력 데이터의 전처리 과정(정규화, 증강 등)이 모델 학습 시와 동일해야 합니다.
- **모델 저장 형식**: 모델을 저장할 때 `torch.save(model.state_dict(), PATH)` 형식을 사용했는지 확인하고, 로드 시 동일한 구조를 유지해야 합니다.
- **하이퍼파라미터 튜닝**: 메타 모델의 하이퍼파라미터(은닉층 크기, 학습률 등)를 조정하여 성능을 최적화할 수 있습니다.
- **오버피팅 방지**: 메타 모델이 너무 복잡해지지 않도록 주의하고, 필요시 드롭아웃 등의 정규화 기법을 사용할 수 있습니다.
- **교차 검증**: 더 견고한 메타 모델을 위해 교차 검증을 도입할 수 있습니다.
- **다양한 메타 모델 실험**: 현재는 신경망을 메타 모델로 사용했지만, Random Forest, Gradient Boosting 등 다른 머신러닝 모델을 메타 모델로 사용하여 성능을 비교할 수 있습니다.
"""

# ----------------------------
# 10. 참고 자료
# ----------------------------

"""
- **스테킹 앙상블**: [Stacking - Wikipedia](https://en.wikipedia.org/wiki/Stacking_(machine_learning))
- **PyTorch 공식 문서**: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- **TorchVision 공식 문서**: [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- **Scikit-learn 공식 문서**: [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
"""
