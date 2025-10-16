# Архитектура гибридной модели

## 1. Общая идея
- Общий энкодер: **ResNet34** (предобученный на ImageNet).
- Ветка сегментации: **U-Net** (decoder + skip connections).
- Ветка классификации: **GlobalAveragePooling2D → Linear**.

## 2. Сегментация (U-Net)
- Библиотека: `segmentation-models-pytorch`
- Вход: 1 канал (grayscale), 512×512 пикселей
- Выход: 1 канал (сигмоида), бинарная маска патологии или области мозга

## 3. Классификация
Признаки из bottleneck энкодера → усреднение → линейный слой:
python
nn.AdaptiveAvgPool2d(1)
nn.Flatten()
nn.Linear(512, num_classes)
`

где `512` — число каналов выходного слоя энкодера.

## 4. Обучение

Совместная оптимизация двух ветвей:

text
Loss = loss_seg + loss_cls


* `loss_seg`: DiceLoss + BCEWithLogitsLoss
* `loss_cls`: CrossEntropyLoss

Оптимизатор: AdamW
Начальный learning rate: 1e-3
Аугментации: флипы, повороты, сдвиги, масштабирование.

## 5. Метрики

* Dice, IoU — для сегментации
* Accuracy, F1 — для классификации
