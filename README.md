#  AI Vision Model Trainer API 

## 1. 프로젝트 개요 및 목적

이 프로젝트는 FastAPI를 백엔드 프레임워크로 사용하여 사용자로부터 이미지 데이터셋을 받아 PyTorch 기반의 딥러닝 모델을 학습하고, 학습된 모델로 새로운 이미지에 대한 예측을 수행하는 AI 비전 모델 트레이너 API를 구현합니다.

### 주요 기능:
- 사용자 정의 두 가지 클래스(레이블)에 대한 이미지 파일 업로드.
- 업로드된 데이터를 80% 학습(Train), 20% 검증(Test) 데이터셋으로 자동 분할.
- 전이 학습(Transfer Learning) 기법을 활용한 ResNet-18 모델 학습.
- 학습된 모델 파일(trained_model.pth) 저장 및 예측 서비스 제공.

### 기술 스택:
- 백엔드: Python, FastAPI
- 딥러닝: PyTorch (torch, torchvision)
- 데이터 처리: PIL (Pillow), os, shutil, uuid

---

## 2. 환경 설정 및 실행 방법

이 API는 Uvicorn 웹 서버를 통해 실행됩니다.

### 필수 라이브러리 설치

```
pip install fastapi uvicorn torch torchvision pillow python-multipart
```


### API 서버 실행

프로젝트 파일 이름이 main.py라고 가정할 때, 커맨드 프롬프트(혹은 터미널)에서 다음 명령어를 입력하여 서버를 시작합니다.

uvicorn main:app --reload

### 설명

main:app: main.py 파일 내의 app 객체를 실행하라는 의미입니다.

--reload: 개발 모드에 유용한 옵션으로, 소스 코드 변경 사항이 감지되면 서버를 자동으로 재시작합니다.

### 서버 실행 결과:

서버가 성공적으로 시작되면 보통 아래와 같은 메시지가 출력됩니다.

INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)


## 3. 주요 API 엔드포인트

| 경로     | 메서드 | 설명                                      | 요청 본문/파라미터                                  | 반환 (JSON)             |
|----------|--------|-------------------------------------------|---------------------------------------------------|--------------------------|
| /train   | POST   | 이미지 파일을 업로드하여 모델 학습 및 검증 | label1, label2, epochs (Form), files1, files2 (File List) | message, log (학습 과정 로그) |
| /predict | POST   | 학습된 모델로 단일 이미지 예측             | file (UploadFile)                                  | prediction, confidence  |
| /        | GET    | API 상태 확인                              | 없음                                              | message                  |



## 4. 딥러닝 핵심 구현 상세

### 4.1. 전이 학습 (Transfer Learning)

기반 모델:
ResNet-18 모델을 사용하며, 이는 대규모 ImageNet 데이터셋으로 사전 학습된 가중치(ResNet18_Weights.IMAGENET1K_V1)를 가져와 사용합니다.

동결 (Freezing):
for param in model.parameters(): param.requires_grad = False

코드를 통해 사전 학습된 계층의 가중치를 고정(동결)하고 학습에서 제외합니다. 이는 학습 시간을 줄이고 데이터가 적을 때 과적합을 방지하는 효과가 있습니다.

분류기 교체:
모델의 마지막 완전 연결 계층(model.fc)을 사용자가 제공한 클래스 개수(num_classes)에 맞게
새로운 계층(nn.Linear(num_ftrs, num_classes))으로 교체합니다.
새로운 분류 계층만 학습됩니다.


## 4.2. 데이터 전처리 및 증강 (Data Augmentation)

### 데이터 증강 (train_transforms):
학습 데이터셋의 다양성을 높여 모델의 일반화 성능을 향상시키기 위해
무작위 뒤집기, 회전, 색상 왜곡 등을 적용합니다.

### 정규화 (Normalize):
모델이 ImageNet 데이터셋으로 학습될 때 사용된
표준 평균 (0.485, 0.456, 0.406) 및
표준 편차 (0.229, 0.224, 0.225)로 입력 이미지를 정규화합니다.


## 4.3. 학습 및 검증 루프 (train_and_evaluate_model)

### 데이터 분할:
업로드된 이미지는 train_endpoint에서
80%는 학습(train)용으로,
20%는 검증(test)용으로 나뉩니다.

### 검증의 역할:
각 Epoch이 끝날 때마다 분리된 검증 데이터셋을 사용하여
Test Accuracy를 측정합니다.
이는 모델이 학습에 사용하지 않은 데이터에 대해
얼마나 잘 작동하는지(즉, 일반화 성능)를 객관적으로 평가합니다.

### 모델 저장:
학습이 완료되면 모델의 상태 정보(model_state_dict)와
클래스 이름(class_names)을 포함하는 체크포인트 파일을
trained_model.pth로 저장합니다.


## 5. 프로젝트의 장점

### 신속한 AI 서비스 구축:
FastAPI의 비동기 처리와 PyTorch의 딥러닝 기능을 결합하여,
이미지 분류 모델을 API 형태로 빠르게 구축하고 배포할 수 있습니다.

### 리소스 효율성:
전이 학습을 사용하여 초기 학습 시간이 오래 걸리는
대규모 모델 학습 과정을 생략하고,
소규모 데이터로도 준수한 성능을 달성할 수 있습니다.

### 전문가 모드 (Train/Test Split):
데이터를 자동으로 학습/검증 데이터로 분할하고
검증 정확도를 제공하여,
사용자가 모델의 과적합 여부를 쉽게 파악할 수 있도록 돕습니다.


## 6. 향후 개선 방안

### 비동기 학습 처리:
현재 train_and_evaluate_model은 동기적으로 실행되어
학습 시간이 길어질 경우 API 응답이 지연됩니다.
Celery 등의 비동기 작업 큐를 사용하여
학습을 백그라운드에서 처리하고,
별도의 엔드포인트로 학습 상태를 확인할 수 있도록 개선합니다.

### 더 많은 클래스 지원:
현재는 두 가지 클래스만 지원합니다.
사용자 입력에 따라 동적으로 N개의 클래스를 처리할 수 있도록 확장합니다.

### GPU 활용 최적화:
학습 장치(device)를 cuda:0을 우선적으로 사용하도록 설정했지만,
학습 시 데이터 로더의 num_workers를 조정하여
GPU 활용도를 높일 수 있습니다.






