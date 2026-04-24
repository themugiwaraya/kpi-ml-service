# Деплой KPI ML Service на Render

## Структура файлов

```
kpi-ml-service/          ← отдельный репо на GitHub
  config/
    settings.py
    urls.py
    wsgi.py
  predictions/
    models.py            ← KPIRecord, ModelVersion, PredictionLog
    views.py             ← /predict, /finalize, /metrics, /health
    services.py          ← логика обучения и предсказания
    serializers.py
    admin.py             ← кнопки Train в Django Admin
    urls.py
    management/
      commands/
        load_dataset.py  ← загрузка CSV в БД
        train_model.py   ← обучение из консоли
  requirements.txt
  Dockerfile
  render.yaml
  .env.example
```

---

## Шаг 1 — Supabase настройка

### 1.1 Создать bucket для модели
Supabase Dashboard → Storage → New bucket
- Name: `ml-models`
- Public: НЕТ (private)

### 1.2 Получить ключи
Supabase → Project Settings → API:
- `SUPABASE_URL` = Project URL
- `SUPABASE_SERVICE_KEY` = service_role (не anon!)

### 1.3 Получить DATABASE_URL
Supabase → Project Settings → Database → Connection string → URI
Выбери: **Transaction pooler** (порт 6543)

---

## Шаг 2 — Деплой на Render

1. Создай новый репо `kpi-ml-service` на GitHub
2. Залей все файлы
3. Render Dashboard → New → Web Service → подключи репо
4. Runtime: **Docker**
5. Environment Variables (добавь вручную):

```
SECRET_KEY          = (сгенерируй случайную строку)
DEBUG               = False
ALLOWED_HOSTS       = kpi-ml-service.onrender.com
DATABASE_URL        = postgresql://postgres.USER:PASSWORD@...
SUPABASE_URL        = https://xxx.supabase.co
SUPABASE_SERVICE_KEY = eyJhbGci...
SUPABASE_BUCKET     = ml-models
ML_API_KEY          = (любая случайная строка — скопируй в основной проект)
ML_MAX_PREDICT_BATCH = 1000
ML_EXCLUDED_ROLES   = HR,ADMIN,COMMISSION
ML_BOOTSTRAP_ON_START = true
ML_BOOTSTRAP_LOAD_DATASET = true
ML_BOOTSTRAP_TRAIN   = true
ML_BOOTSTRAP_MODEL_TYPES = linear_regression,random_forest
ML_BOOTSTRAP_PRIMARY_MODEL_TYPE = random_forest
ML_BOOTSTRAP_STRICT  = false
```

6. Deploy

---

## Шаг 3 — Что происходит автоматически при деплое

Shell на Render не нужен:

1. Контейнер запускает `python manage.py migrate --noinput`
2. Затем запускается `python manage.py bootstrap_ml`
3. `bootstrap_ml` делает idempotent шаги:
  - загружает `kpi_dataset.csv` в `kpi_dataset`, только если таблица пуста
  - обучает стартовые модели из `ML_BOOTSTRAP_MODEL_TYPES`, только если нет активной версии модели
  - модель из `ML_BOOTSTRAP_PRIMARY_MODEL_TYPE` тренируется последней и становится активной
4. После этого стартует Gunicorn

Если нужно отключить любой шаг, поменяй env:

- `ML_BOOTSTRAP_LOAD_DATASET=false`
- `ML_BOOTSTRAP_TRAIN=false`
- `ML_BOOTSTRAP_ON_START=false`

---

## Шаг 4 — Основной Django проект

Добавь в `.env` основного проекта:
```
ML_SERVICE_URL=https://kpi-ml-service.onrender.com
ML_API_KEY=тот-же-ключ-что-в-ml-сервисе
```

Положи `ml_client_for_main_project.py` в:
```
your_main_project/kpi/services/ml_client.py
```

---

## Шаг 5 — Использование в основном проекте

### Предсказание для одного преподавателя:
```python
from kpi.services.ml_client import predict_kpi

result = predict_kpi(
    teacher=teacher_obj,          # Django model
    kpi_blocks={"block_1": 18.5, "block_2": 62.0, "block_3": 0, "block_4": 0},
    year=2026
)
# result = { "predicted_kpi": 83.2, "gap": 2.7, "gap_interpretation": "On track" }
```

### Финализация конца года (из Django Admin основного проекта):
```python
from kpi.services.ml_client import finalize_year

result = finalize_year(
    year=2025,
    evaluations_queryset=KPIEvaluation.objects.filter(year=2025, status="finalized")
)
```

---

## Django Admin ML сервиса

После деплоя: `https://kpi-ml-service.onrender.com/admin/`

Там доступно:
- **KPI Records** — просмотр датасета, фильтрация по году/роли
- **Model Versions** — история моделей + кнопки "Train RF" и "Train LR"
- **Prediction Logs** — лог всех предсказаний

### Кнопки обучения:
1. Открой `/admin/predictions/modelversion/`
2. Выбери любую запись (или все)
3. Actions → "🚀 Train NEW Random Forest model"
4. Нажми Go

---

## API Endpoints

```
GET  /api/health/    — статус сервиса и активная модель
POST /api/predict/   — предсказание (X-API-Key header обязателен)
POST /api/finalize/  — финализация года (X-API-Key header обязателен)
GET  /api/metrics/   — история версий моделей
```

Для `POST /api/finalize/` обязателен дополнительный header:

```
X-Idempotency-Key: <uuid>
```
