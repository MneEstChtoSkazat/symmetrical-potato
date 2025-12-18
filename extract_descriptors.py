import torch
import torchvision.transforms as T
from torch.hub import load
import faiss
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import pickle

# DINOv2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

dinov2 = (
    load("facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True)
    .eval()
    .to(device)
)
# dinov2_vitb14_reg

transform = T.Compose(
    [
        T.Resize((518, 518)),  # Оптимальный размер для DINOv2
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_descriptor(img_bgr):
    """Извлекает нормализованный дескриптор (768-dim) из изображения"""
    if img_bgr is None:
        raise ValueError("Изображение не загружено!")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = T.ToPILImage()(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        desc = dinov2(input_tensor)  # (1, 768)
    desc = desc.cpu().numpy()[0]
    desc /= np.linalg.norm(desc)  # L2-нормализация для косинусного поиска
    return desc.astype(np.float32)


# КЭШ
CACHE_DIR = "dinov2_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_FILE = os.path.join(CACHE_DIR, "faiss.index")
POS_FILE = os.path.join(CACHE_DIR, "positions.npy")
NAMES_FILE = os.path.join(CACHE_DIR, "img_names.pkl")


# Построение базы
def build_database():
    print("Строим базу DINOv2 дескрипторов...")
    img_dir = "airsim_dataset/images"
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Не найдена папка {img_dir}")

    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    print(f"Найдено изображений: {len(img_names)}")

    descriptors = []
    for name in tqdm(img_names, desc="Извлечение дескрипторов"):
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Пропуск {name} (не удалось загрузить)")
            continue
        desc = get_descriptor(img)
        descriptors.append(desc)

    if len(descriptors) == 0:
        raise ValueError("Не удалось извлечь ни одного дескриптора!")

    descriptors = np.array(descriptors)
    print(f"Дескрипторы: {descriptors.shape}")

    # FAISS индекс
    dimension = descriptors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(descriptors)
    faiss.write_index(index, INDEX_FILE)
    print(f"Индекс сохранён: {INDEX_FILE}")

    # Позиции из CSV
    poses = pd.read_csv("airsim_dataset/poses.csv")
    if len(poses) != len(img_names):
        print("Кол-во поз не равно кол-ву изображений")
    positions = poses[["x", "y"]].values.astype(np.float32)[: len(img_names)]
    np.save(POS_FILE, positions)
    print(f"Позиции сохранены: {POS_FILE}")

    with open(NAMES_FILE, "wb") as f:
        pickle.dump(img_names, f)


# Загрузка базы
def load_database():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"База не построена!")
    index = faiss.read_index(INDEX_FILE)
    positions = np.load(POS_FILE)
    with open(NAMES_FILE, "rb") as f:
        img_names = pickle.load(f)
    print(f"База загружена: {len(img_names)} изображений")
    return index, positions, img_names


if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        build_database()

    global index, db_positions, img_names
    index, db_positions, img_names = load_database()
