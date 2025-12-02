import os
import shutil
import uuid
import random
from typing import List
import io

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="AI Vision Model Trainer API")

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# --- 딥러닝 모델 학습 및 평가 함수 ---
def train_and_evaluate_model(base_dir: str, num_epochs: int) -> List[str]:
    logs = []
    logs.append("--- 전문가 모드 안내 ---")
    logs.append("1. 업로드된 이미지를 학습(80%) 및 검증(20%)용으로 자동 분할합니다.")
    logs.append("2. 매 학습마다 검증용 데이터로 '테스트 정확도'를 측정하여 모델의 일반화 성능을 평가합니다.")
    logs.append("-" * 20)

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # --- 1. 데이터 전처리 및 로더 설정 ---
    logs.append("[단계 1/5] 데이터셋 준비 (학습/검증 분할)")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, train_transforms)
    test_dataset = ImageFolder(test_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    logs.append(f"데이터셋 분할 완료. 클래스: {class_names}")
    logs.append(f"학습 이미지: {len(train_dataset)}개 / 검증 이미지: {len(test_dataset)}개")

    # --- 2. 모델 설정 ---
    logs.append("[단계 2/5] 전이 학습 모델 설정")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logs.append(f"학습 장치: {device}")

    # --- 3. 손실 함수 및 옵티마이저 정의 ---
    logs.append("[단계 3/5] 손실 함수와 옵티마이저 정의")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # --- 4. 모델 학습 및 검증 루프 ---
    logs.append(f"[단계 4/5] 모델 학습 및 검증 시작 (총 {num_epochs}회 반복)")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_acc = corrects.double() / len(test_dataset)

        epoch_log = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.2%}"
        logs.append(epoch_log)
        print(epoch_log)

    logs.append("[단계 5/5] 모델 학습 완료!")
    # --- 5. 모델 저장 ---
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    # "wb" (write binary) 모드를 사용하여 파일을 저장합니다.
    with open("trained_model.pth", "wb") as f:
        torch.save(checkpoint, f)
    logs.append("모델이 'trained_model.pth' 파일로 안전하게 저장되었습니다.")

    return logs


# --- 이미지 예측 함수 ---
def predict_image(image_bytes: bytes) -> (str, float):
    if not os.path.exists("trained_model.pth"):
        raise HTTPException(status_code=400, detail="학습된 모델이 없습니다. 먼저 모델을 학습시켜 주세요.")

    # "rb" (read binary) 모드와 map_location을 사용하여 안정적으로 모델을 불러옵니다.
    with open("trained_model.pth", "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    class_names = checkpoint['class_names']

    num_classes = len(class_names)
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # PIL.Image.open은 바이트 스트림을 직접 받을 수 있습니다.
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        prediction = class_names[top_catid[0]]
        confidence = top_prob[0].item()
    return prediction, confidence


# --- API 엔드포인트 ---
@app.post("/train")
async def train_endpoint(
        label1: str = Form(...),
        label2: str = Form(...),
        epochs: int = Form(...),
        files1: List[UploadFile] = File(...),
        files2: List[UploadFile] = File(...)
):
    session_id = str(uuid.uuid4())
    base_dir = "temp_images"
    session_dir = os.path.join(base_dir, session_id)

    for folder in ["train", "test"]:
        for label in [label1, label2]:
            os.makedirs(os.path.join(session_dir, folder, label), exist_ok=True)

    try:
        logs = []
        for i, files in enumerate([files1, files2]):
            label = [label1, label2][i]
            valid_files = []
            for file in files:
                try:
                    file.file.seek(0)
                    Image.open(file.file)
                    file.file.seek(0)
                    valid_files.append(file)
                except Exception:
                    logs.append(f"경고: '{file.filename}'은(는) 유효한 이미지 파일이 아니므로 제외합니다.")

            random.shuffle(valid_files)
            split_idx = int(len(valid_files) * 0.8)
            train_files = valid_files[:split_idx]
            test_files = valid_files[split_idx:]

            for file in train_files:
                file_path = os.path.join(session_dir, "train", label, file.filename)
                with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
            for file in test_files:
                file_path = os.path.join(session_dir, "test", label, file.filename)
                with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

        training_logs = train_and_evaluate_model(session_dir, num_epochs=epochs)
        logs.extend(training_logs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
    finally:
        if os.path.exists(session_dir): shutil.rmtree(session_dir)

    return {"message": "모델 학습 및 검증이 성공적으로 완료되었습니다.", "log": logs}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        prediction, confidence = predict_image(image_bytes)
    except HTTPException as e:
        raise e
    except Exception as e:
        # 오류 발생 시 더 자세한 정보를 클라이언트에 전달
        import traceback
        error_details = traceback.format_exc()
        print(error_details)  # 서버 콘솔에 전체 오류 출력
        raise HTTPException(status_code=500, detail=f"예측 중 심각한 오류 발생: {e}")
    return {"prediction": prediction, "confidence": confidence}


@app.get("/")
def read_root():
    return {"message": "AI Vision Platform API"}
#command prompt에서 쳐야함
#uvicorn main:app --reload
