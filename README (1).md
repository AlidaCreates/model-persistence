# Day 18 — Model Persistence & MLflow

Готовый проект по Day 18: persistence моделей, `joblib`, schema validation, metadata, MLflow и inference.

Production pipeline использует feature engineering, preprocessing и `HistGradientBoostingRegressor`; ожидаемый test R² — около `0.8427`. fileciteturn1file0L14-L24

## Структура

```text
day18_persistence_project/
├── persistence.py
├── README.md
├── requirements.txt
└── .gitignore
```

После запуска создаются `model.joblib`, `baseline.joblib` и `model.metadata.json`.

## Установка

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python persistence.py
```

Задание использует `joblib`, `scikit-learn` и `mlflow`; в примере ожидаются версии около `1.4.2`, `1.4.2` и `2.13.2` соответственно. fileciteturn1file0L1-L16

## Exercises

### Exercise 0
Обучается production pipeline:

```text
FunctionTransformer
→ ColumnTransformer
→ HistGradientBoostingRegressor
```

Ожидаемый R²: `0.8427`. fileciteturn1file0L18-L24

### Exercise 1 — joblib
Pipeline сохраняется в `model.joblib`, загружается обратно и проверяется идентичность predictions и R². fileciteturn1file0L31-L59

### Exercise 2 — baseline
Создаётся облегчённый pipeline с `LinearRegression`, после чего сравниваются размеры `baseline.joblib` и `model.joblib`. fileciteturn1file0L65-L85

### Exercise 3 — schema validation
Проверяются missing columns, extra columns и неправильные dtypes. Ошибки выдаются через `ValueError` до обращения к модели. fileciteturn1file0L91-L117

### Exercise 4 — metadata
Создаётся `model.metadata.json` с версией модели, датой обучения, preprocessing, схемой входа, метрикой и размерами train/test. fileciteturn1file0L123-L143

### Exercise 5 — MLflow
Модель логируется в experiment `california-housing` с run name `hgb-production-v1`, параметрами и `test_r2`, затем загружается по `runs:/<run_id>/model`. fileciteturn1file0L149-L169

### Exercise 6 — inference
`predict_house_value()` выполняет:

```text
load metadata
→ validate input
→ load model
→ select expected columns
→ predict
```

Также проверяется отказ на DataFrame без `median_income`. fileciteturn1file0L176-L204

## MLflow UI

После запуска:

```bash
mlflow ui
```

Открой локальный адрес, который выведет MLflow.

## Git

```bash
git init
git add .
git commit -m "feat: implement day 18 model persistence"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

## Production workflow

```text
Training
  ↓
Pipeline
  ↓
model.joblib + model.metadata.json
  ↓
Schema validation
  ↓
Inference
```

Отдельно:

```text
Training
  ↓
MLflow Run
  ↓
Logged Model
  ↓
runs:/<run_id>/model
```
