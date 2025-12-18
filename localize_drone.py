import airsim
import cv2
import numpy as np
import time
import os
import faiss
import torch
import torchvision.transforms as T
from torch.hub import load
from PIL import Image
import math
import random

# DINOv2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")

dinov2 = (
    load("facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True)
    .eval()
    .to(device)
)

transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_north_up_image(img_bgr):
    pose = client.simGetVehiclePose()
    yaw_rad = airsim.to_eularian_angles(pose.orientation)[2]  # только yaw

    yaw_deg = math.degrees(yaw_rad)
    yaw_deg = (yaw_deg + 360) % 360

    rotation_angle = -yaw_deg

    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    rotated = cv2.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return rotated


def get_descriptor(img_bgr):
    north_up_img = get_north_up_image(img_bgr)

    img_rgb = cv2.cvtColor(north_up_img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        desc = dinov2(tensor)[0].cpu().numpy()
    desc /= np.linalg.norm(desc)
    return desc.astype("float32")


# Загрузка базы
CACHE_DIR = "dinov2_cache"
index = faiss.read_index(os.path.join(CACHE_DIR, "faiss.index"))
db_positions = np.load(os.path.join(CACHE_DIR, "positions.npy"))
print(f"База загружена: {len(db_positions)} точек")

# AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Камера строго вниз
client.simSetCameraPose(
    "front_camera", airsim.Pose(orientation_val=airsim.to_quaternion(-np.pi / 2, 0, 0))
)

print("Взлёт...")
client.takeoffAsync().join()
client.moveToZAsync(-145.0, 3).join()
time.sleep(3)


x_min, x_max = -212, 315
y_min, y_max = -180, 127

waypoints = []
for _ in range(5):
    x = round(random.uniform(x_min, x_max), 1)
    y = round(random.uniform(y_min, y_max), 1)
    waypoints.append((x, y))

print(f"Точки:")
for i, (x, y) in enumerate(waypoints, 1):
    print(f"  {i}. ({x:+.1f}, {y:+.1f})")

# Полет по точкам
all_errors = []


print("\nСТАРТ\n")

for idx, (target_x, target_y) in enumerate(waypoints, 1):
    print(f"ТОЧКА {idx}/5 → Летим к ({target_x:+.1f}, {target_y:+.1f})")

    client.moveToPositionAsync(float(target_x), float(target_y), -145.0, 10.0)

    point_errors = []

    while True:
        # Получаем изображение
        responses = client.simGetImages(
            [airsim.ImageRequest("front_camera", airsim.ImageType.Scene, False, False)]
        )
        resp = responses[0]
        if not resp.image_data_uint8:
            time.sleep(0.1)
            continue

        img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(720, 1280, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Истинная позиция
        pose = client.simGetVehiclePose()
        true_x = pose.position.x_val
        true_y = pose.position.y_val

        # Предсказание
        try:
            desc = get_descriptor(img_bgr).reshape(1, -1)
            D, I = index.search(desc, k=1)
            pred_x, pred_y = db_positions[I[0][0]]
            score = float(D[0][0])
        except:
            print("Ошибка дескриптора")
            time.sleep(0.1)
            continue

        error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        point_errors.append(error)
        all_errors.append(error)

        print(
            f"  True: ({true_x:7.1f}, {true_y:7.1f}) → Pred: ({pred_x:7.1f}, {pred_y:7.1f}) | "
            f"Ошибка: {error:5.2f} м | Score: {score:.4f} → Цель: ({target_x:+.1f}, {target_y:+.1f})"
        )

        # Проверяем, прилетели ли
        dist_to_target = np.sqrt((true_x - target_x) ** 2 + (true_y - target_y) ** 2)
        if dist_to_target < 8.0:  # достаточно близко
            print(f"Прибыли\n")
            break

        time.sleep(1.5)


print("Посадка...")
client.hoverAsync().join()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
