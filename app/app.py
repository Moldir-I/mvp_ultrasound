import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# === Добавляем путь к корню проекта (чтобы видеть models/) ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.hybrid_unet import HybridUNetClassifierClean  # модель

# === Настройки страницы ===
st.set_page_config(page_title="MVP Ultrasound", layout="centered")
st.title(" Ultrasound Brain Image Classifier (MVP)")
st.markdown("Модель: U-Net + Classifier · Обучена на УЗИ головного мозга у детей с особыми потребностями")

# === Загрузка модели ===
@st.cache_resource
def load_model():
    model = HybridUNetClassifierClean(num_classes=3)
    model_path = PROJECT_ROOT / "models" / "mvp_model.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("⚠️ Пропущенные ключи:", missing)
    if unexpected:
        print("⚠️ Лишние ключи:", unexpected)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ Модель загружена успешно")
except Exception as e:
    st.warning(f"⚠️ Не удалось загрузить модель: {e}")

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

label_names = ["benign", "malignant", "normal"]

# === Интерфейс загрузки ===
uploaded = st.file_uploader("📤 Загрузите УЗИ изображение", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Исходное изображение", use_container_width=True)

    # Преобразуем и прогоняем через модель
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        seg, logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_class = int(np.argmax(probs))
        mask = seg.squeeze().numpy()

    # === Обработка маски ===
    mask = np.clip(mask, 0, 1)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask = mask ** 3
    mask_bin = (mask > 0.6).astype(np.uint8) * 255

    # Масштабируем под исходное изображение
    mask_img = Image.fromarray(mask_bin)
    mask_resized = mask_img.resize(image.size, resample=Image.BILINEAR)
    mask_resized = np.array(mask_resized, dtype=np.uint8)

    # === Наложение (красная маска поверх исходного УЗИ) ===
    overlay = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_color = np.zeros_like(overlay, dtype=np.uint8)
    mask_color[..., 0] = mask_resized
    blended = cv2.addWeighted(overlay, 0.7, mask_color, 0.6, 0)

    # === Отображение результатов ===
    st.markdown("## 📊 Результаты анализа")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🩻 *Предсказанный диагноз:*")
        st.subheader(f"{label_names[pred_class].capitalize()}")
    with col2:
        st.markdown("📍 *Сегментация (наложение маски):*")
        st.image(blended, caption="Сегментированная область", use_container_width=True)

    # === Диаграмма уверенности ===
    st.markdown("### 🔎 Уверенность модели по классам")

    # Цвета: выделяем выбранный класс ярко, остальные — серее
    colors = ["#6EC1E4", "#E74C3C", "#2ECC71"]  # голубой, красный, зелёный
    alpha = [0.3 if i != pred_class else 1.0 for i in range(len(label_names))]

    fig, ax = plt.subplots(figsize=(4, 2.8))
    for i, (label, p) in enumerate(zip(label_names, probs)):
        ax.bar(label, p * 100, color=colors[i], alpha=alpha[i])
        ax.text(i, p * 100 + 1, f"{p*100:.1f}%", ha='center', fontsize=10,
                fontweight='bold' if i == pred_class else 'normal')

    ax.set_ylim(0, 100)
    ax.set_ylabel("Уверенность (%)")
    ax.set_title("Вероятности классов")
    st.pyplot(fig)

    # === Отладочная визуализация маски ===
    with st.expander("🔍 Показать маску модели"):
        st.image(mask * 255, clamp=True, caption="Сырая маска (из модели)")