# Student Performance ML System

End-to-end machine learning project to **predict student performance** and provide **profile clustering** for interpretability.

## What’s inside
- **Model training & evaluation** (Random Forest, XGBoost, Decision Tree, KNN, etc.)
- **Hyperparameter tuning** (RandomizedSearchCV)
- **Clustering** with KMeans + silhouette score
- **GUI app** for predictions (runs locally)

## Repository structure
```
student-performance-ml-system/
├── src/
│   ├── model_evaluation.py   # training + evaluation
│   └── app.py                # GUI app (loads trained models)
├── data/                     # (optional) small sample data
├── assets/                   # screenshots (optional)
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run training / evaluation
```bash
python src/model_evaluation.py
```

> **Note:** Do NOT commit large datasets or `.pkl` model files. Keep them local or generate them during training.

## Run the app
```bash
python src/app.py
```

## Notes for GitHub
- Add a screenshot of the GUI into `assets/` and reference it here.
- If your dataset is big, put only a small sample in `data/` and link the full dataset source in this README.
