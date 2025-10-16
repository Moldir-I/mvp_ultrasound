# Data Directory — Ultrasound Brain MVP

## 📂 Overview
This folder contains **local data** used for training, validation, and testing of the hybrid model for ultrasound image analysis.

The dataset includes ultrasound images of the fetal or neonatal brain and related medical annotations used for segmentation and classification experiments.

---

## ⚠️ Data Storage Policy
Raw medical data **is not stored in the public repository** due to:
- Ethical and legal data protection requirements;
- File size limitations (GitHub limit: 100 MB);
- Compliance with medical research confidentiality standards.

All datasets are stored **locally** on the researcher’s computer and referenced during training and evaluation.

---

## 🧩 Folder Structure


data/
│
├── raw/            # Original datasets (downloaded from Kaggle, Zenodo, or OpenNeuro)
│   ├── ultrasound-fetus-dataset/
│   └── ...
│
├── processed/      # Preprocessed and normalized data (512×512 grayscale images)
│
└── annotations/    # Segmentation masks or CSV label files

`

---

## 📥 Data Sources
Example dataset used in this MVP:
- **Ultrasound Fetus Dataset** — [Kaggle](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset)

Other potential sources:
- [Zenodo](https://zenodo.org)
- [OpenNeuro](https://openneuro.org)
- [Medical Segmentation Decathlon](http://medicaldecathlon.com)

---

## 🧠 Usage Notes
- All data must be downloaded manually or via Kaggle API before running any notebooks.
- Ensure the correct folder paths in your Jupyter notebooks:
  python
  DATA_PATH = "E:/work/mvp_ultrasound/data/raw/ultrasound-fetus-dataset"
`

* The data folder is excluded from version control (`.gitignore`).

---

*Last updated: October 2025*



---