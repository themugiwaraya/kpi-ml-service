"""
ML Service Logic
- Загрузка/сохранение модели через Supabase Storage
- Обучение модели на данных из БД
- Предсказание
"""
import io
import logging
import os
import threading
import numpy as np
import pandas as pd
import joblib

from datetime import datetime
from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.utils import OperationalError, ProgrammingError
from django.utils import timezone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .role_block_config import (
    BLOCK_FEATURES,
    applicable_blocks_for_role,
    normalize_role_name,
)

logger = logging.getLogger(__name__)

FEATURES = ["role", "department", "experience_years", *BLOCK_FEATURES]
TARGET   = "total_kpi"

CAT_FEATURES = ["role", "department"]
NUM_FEATURES = ["experience_years", *BLOCK_FEATURES]


def _normalize_role_name(role: str) -> str:
    return normalize_role_name(role)


def _applicable_blocks_for_role(role: str) -> set[str]:
    return applicable_blocks_for_role(role)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_record_blocks(record: dict) -> dict:
    normalized = dict(record)
    applicable = _applicable_blocks_for_role(normalized.get("role", ""))
    for block in BLOCK_FEATURES:
        value = _safe_float(normalized.get(block, 0), default=0.0)
        normalized[block] = value if block in applicable else 0.0
    return normalized


def _normalize_dataframe_blocks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    records = df.to_dict(orient="records")
    normalized = [_normalize_record_blocks(r) for r in records]
    return pd.DataFrame(normalized)


def _excluded_roles() -> set[str]:
    raw = getattr(settings, "ML_EXCLUDED_ROLES", ["HR", "ADMIN", "COMMISSION"])
    if isinstance(raw, str):
        return {_normalize_role_name(r) for r in raw.split(",") if str(r).strip()}
    return {_normalize_role_name(str(r)) for r in raw if str(r).strip()}


def _storage_enabled() -> bool:
    return bool(getattr(settings, "SUPABASE_URL", "") and getattr(settings, "SUPABASE_SERVICE_KEY", ""))

# ──────────────────────────────────────────────
# SUPABASE STORAGE
# ──────────────────────────────────────────────

def _get_supabase_client():
    from supabase import create_client
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def upload_model_to_storage(pipeline, filename: str) -> str:
    """Сериализует pipeline и загружает в Supabase Storage. Возвращает путь."""
    local_path = os.path.join(settings.MODEL_CACHE_DIR, filename)
    if not _storage_enabled():
        joblib.dump(pipeline, local_path)
        logger.info(f"Model saved to local storage: {local_path}")
        return filename

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    buf.seek(0)

    sb = _get_supabase_client()
    sb.storage.from_(settings.SUPABASE_BUCKET).upload(
        path=filename,
        file=buf.getvalue(),
        file_options={"content-type": "application/octet-stream", "upsert": "true"},
    )
    logger.info(f"Model uploaded to Supabase Storage: {filename}")
    return filename


def download_model_from_storage(filename: str):
    """Скачивает модель из Supabase Storage, кэширует локально."""
    local_path = os.path.join(settings.MODEL_CACHE_DIR, filename)

    # Если уже есть в кэше — не качаем снова
    if os.path.exists(local_path):
        logger.info(f"Loading model from local cache: {local_path}")
        return joblib.load(local_path)

    if not _storage_enabled():
        raise FileNotFoundError(f"Model not found in local cache: {local_path}")

    logger.info(f"Downloading model from Supabase Storage: {filename}")
    sb = _get_supabase_client()
    data = sb.storage.from_(settings.SUPABASE_BUCKET).download(filename)

    with open(local_path, "wb") as f:
        f.write(data)

    return joblib.load(local_path)


def invalidate_model_cache(filename: str):
    """Удаляет локальный кэш (нужно после переобучения)."""
    local_path = os.path.join(settings.MODEL_CACHE_DIR, filename)
    if os.path.exists(local_path):
        os.remove(local_path)
        logger.info(f"Cache invalidated: {local_path}")


# ──────────────────────────────────────────────
# ACTIVE MODEL
# ──────────────────────────────────────────────

_active_pipeline = None   # in-memory кэш активной модели
_active_model_version = None
_active_lock = threading.Lock()


def get_active_pipeline():
    """
    Возвращает активный pipeline.
    Загружает из Supabase Storage при первом вызове или после инвалидации.
    """
    global _active_pipeline, _active_model_version
    if _active_pipeline is not None:
        return _active_pipeline

    with _active_lock:
        if _active_pipeline is not None:
            return _active_pipeline

        from .models import ModelVersion
        try:
            version = ModelVersion.objects.filter(status="active").order_by("-trained_at").first()
        except (ProgrammingError, OperationalError) as exc:
            logger.exception("Active model lookup failed due to DB schema/table issue")
            raise RuntimeError(
                "Model tables are unavailable in current DB schema. "
                "Check DB_SEARCH_PATH and run migrations/bootstrap in the target schema."
            ) from exc

        if version is None:
            raise RuntimeError("No active model found. Run training first via Django Admin.")

        _active_pipeline = download_model_from_storage(version.storage_path)
        _active_model_version = version
        logger.info(f"Active model loaded: {version.version}")
        return _active_pipeline


def get_active_model_version():
    global _active_model_version
    if _active_model_version is not None:
        return _active_model_version
    get_active_pipeline()
    return _active_model_version


def invalidate_active_pipeline():
    """Сбрасывает in-memory кэш после переобучения."""
    global _active_pipeline, _active_model_version
    _active_pipeline = None
    _active_model_version = None


def check_storage_access() -> bool:
    """Проверяет доступность Supabase Storage bucket."""
    if not _storage_enabled():
        return False
    try:
        sb = _get_supabase_client()
        sb.storage.from_(settings.SUPABASE_BUCKET).list(path="", options={"limit": 1})
        return True
    except Exception:
        logger.exception("Supabase storage healthcheck failed")
        return False


# ──────────────────────────────────────────────
# PIPELINE BUILDER
# ──────────────────────────────────────────────

def _build_pipeline(model_type: str):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "linear_regression":
        estimator = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("pre", preprocessor), ("model", estimator)])


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train_and_save(train_years: list[int], test_year: int, model_type: str = "random_forest"):
    """
    Обучает модель на train_years, оценивает на test_year.
    Сохраняет в Supabase Storage, создаёт запись ModelVersion.
    Возвращает ModelVersion instance.
    """
    from .models import KPIRecord, ModelVersion

    excluded = _excluded_roles()

    # Загружаем данные из БД
    train_qs = KPIRecord.objects.filter(year__in=train_years).exclude(role__in=excluded).values(*FEATURES, TARGET)
    test_qs  = KPIRecord.objects.filter(year=test_year).exclude(role__in=excluded).values(*FEATURES, TARGET)

    df_train = pd.DataFrame(train_qs)
    df_test  = pd.DataFrame(test_qs)

    if df_train.empty:
        raise ValueError(f"No training data for years {train_years}")
    if df_test.empty:
        raise ValueError(f"No test data for year {test_year}")

    df_train = _normalize_dataframe_blocks(df_train)
    df_test = _normalize_dataframe_blocks(df_test)

    logger.info(f"Training {model_type}: {len(df_train)} train / {len(df_test)} test records")

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]

    pipeline = _build_pipeline(model_type)
    pipeline.fit(X_train, y_train)

    # Метрики
    y_pred = pipeline.predict(X_test)
    mae    = round(mean_absolute_error(y_test, y_pred), 3)
    rmse   = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3)
    r2     = round(r2_score(y_test, y_pred), 4)

    logger.info(f"Metrics → MAE={mae}  RMSE={rmse}  R²={r2}")

    # Версия
    now        = timezone.now()
    version_id = f"v{now.strftime('%Y%m%d_%H%M%S')}_{model_type[:2].upper()}"
    filename   = f"{version_id}.pkl"

    old_active_paths = list(ModelVersion.objects.filter(status="active").values_list("storage_path", flat=True))

    # Загружаем в Supabase Storage
    storage_path = upload_model_to_storage(pipeline, filename)

    # Создаём запись в БД атомарно
    with transaction.atomic():
        ModelVersion.objects.filter(status="active").update(status="archived")
        version = ModelVersion.objects.create(
            version      = version_id,
            model_type   = model_type,
            storage_path = storage_path,
            status       = "active",
            train_years  = f"{min(train_years)}-{max(train_years)}",
            train_records= len(df_train),
            test_year    = test_year,
            mae          = mae,
            rmse         = rmse,
            r2           = r2,
            trained_at   = now,
        )

    invalidate_active_pipeline()
    for old_path in old_active_paths:
        invalidate_model_cache(old_path)

    logger.info(f"New active model: {version_id}")
    return version


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────

def predict_one(data: dict) -> dict:
    """
    data: {teacher_id, role, department, experience_years, block_1..4}
    Возвращает prediction dict.
    """
    normalized_data = _normalize_record_blocks(data)
    role = _normalize_role_name(normalized_data.get("role", ""))
    if role in _excluded_roles():
        raise ValueError(f"Role '{role}' is excluded from ML predictions")

    pipeline = get_active_pipeline()
    model_version = get_active_model_version()
    df = pd.DataFrame([normalized_data])[FEATURES]

    predicted = float(pipeline.predict(df)[0])
    predicted = round(min(max(predicted, 0), 100), 2)

    blocks      = [normalized_data.get(f"block_{i}", 0) for i in range(1, 5)]
    current_sum = round(sum(blocks), 2)
    gap         = round(predicted - current_sum, 2)

    if gap > 3:
        interpretation = "Growth potential"
    elif abs(gap) <= 3:
        interpretation = "On track"
    else:
        interpretation = "Risk of decline"

    return {
        "teacher_id":       normalized_data.get("teacher_id"),
        "role":             normalized_data.get("role"),
        "department":       normalized_data.get("department"),
        "predicted_kpi":    predicted,
        "current_sum":      current_sum,
        "gap":              gap,
        "gap_interpretation": interpretation,
        "model_version":    model_version.version if model_version else None,
    }


def predict_batch(records: list[dict]) -> list[dict]:
    excluded = _excluded_roles()
    normalized_records = [_normalize_record_blocks(r) for r in records]
    unsupported = sorted({_normalize_role_name(r.get("role", "")) for r in normalized_records if _normalize_role_name(r.get("role", "")) in excluded})
    if unsupported:
        raise ValueError(f"Excluded roles in batch: {', '.join(unsupported)}")

    pipeline = get_active_pipeline()
    model_version = get_active_model_version()
    df = pd.DataFrame(normalized_records)[FEATURES]

    preds = pipeline.predict(df)
    results = []

    for i, rec in enumerate(normalized_records):
        predicted   = round(min(max(float(preds[i]), 0), 100), 2)
        blocks      = [rec.get(f"block_{j}", 0) for j in range(1, 5)]
        current_sum = round(sum(blocks), 2)
        gap         = round(predicted - current_sum, 2)

        results.append({
            "teacher_id":    rec.get("teacher_id"),
            "role":          rec.get("role"),
            "department":    rec.get("department"),
            "predicted_kpi": predicted,
            "current_sum":   current_sum,
            "gap":           gap,
            "model_version": model_version.version if model_version else None,
            "gap_interpretation": (
                "Growth potential" if gap > 3 else
                "On track"         if abs(gap) <= 3 else
                "Risk of decline"
            ),
        })

    return results


# ──────────────────────────────────────────────
# ANALYTICS (historical + prediction)
# ──────────────────────────────────────────────

def get_analytics(filters: dict) -> dict:
    """
    Возвращает historical и prediction по фильтрам.
    Поддерживаемые фильтры: department, role, teacher_id.
    """
    from .models import KPIRecord

    department = (filters.get("department") or "").strip()
    role = (filters.get("role") or "").strip()
    teacher_id = filters.get("teacher_id")

    qs = KPIRecord.objects.all()
    if department:
        qs = qs.filter(department__iexact=department)
    if role:
        qs = qs.filter(role__iexact=role)
    if teacher_id is not None:
        qs = qs.filter(teacher_id=teacher_id)

    hist_df = pd.DataFrame(qs.values("year", TARGET))
    if hist_df.empty:
        raise ValueError("No KPI records found for given filters")

    grouped = (
        hist_df.groupby("year", as_index=False)[TARGET]
        .mean()
        .sort_values("year")
    )

    historical = {
        str(int(row["year"])): round(float(row[TARGET]), 2)
        for _, row in grouped.iterrows()
    }

    latest_year = int(grouped["year"].max())
    next_year = latest_year + 1

    prediction_value = None
    prediction_source = "trend_fallback"
    model_version = None

    # Пытаемся использовать активную ML-модель на последнем году выборки.
    try:
        pipeline = get_active_pipeline()
        active_version = get_active_model_version()
        latest_df = pd.DataFrame(qs.filter(year=latest_year).values(*FEATURES))
        latest_df = _normalize_dataframe_blocks(latest_df)
        if not latest_df.empty:
            preds = pipeline.predict(latest_df[FEATURES])
            preds = np.clip(preds, 0, 100)
            prediction_value = round(float(np.mean(preds)), 2)
            prediction_source = "active_model_mean_prediction"
            model_version = active_version.version if active_version else None
    except Exception as exc:
        logger.warning(f"Analytics model prediction unavailable, fallback to trend: {exc}")

    # Если нет активной модели, даем стабильный fallback по линейному тренду.
    if prediction_value is None:
        years = grouped["year"].to_numpy(dtype=float)
        values = grouped[TARGET].to_numpy(dtype=float)
        if len(values) >= 2:
            slope, intercept = np.polyfit(years, values, 1)
            raw_pred = float(slope * next_year + intercept)
            prediction_value = round(min(max(raw_pred, 0.0), 100.0), 2)
            prediction_source = "trend_fallback"

    applied_filters = {}
    if department:
        applied_filters["department"] = department
    if role:
        applied_filters["role"] = role
    if teacher_id is not None:
        applied_filters["teacher_id"] = teacher_id

    return {
        "filters": applied_filters,
        "historical": historical,
        "prediction": {
            "year": next_year,
            "value": prediction_value,
            "source": prediction_source,
            "model_version": model_version,
        },
    }


# ──────────────────────────────────────────────
# SNAPSHOTS (precomputed predictions for fast reads)
# ──────────────────────────────────────────────

def _snapshot_scope_key(scope: str, *, department: str = "", role: str = "", teacher_id: int | None = None) -> str:
    if scope == "overall":
        return "overall"
    if scope == "department":
        return f"department:{department}"
    if scope == "role":
        return f"role:{role}"
    if scope == "department_role":
        return f"department_role:{department}|{role}"
    if scope == "teacher":
        return f"teacher:{teacher_id}"
    raise ValueError(f"Unsupported snapshot scope: {scope}")


def rebuild_prediction_snapshots(base_year: int | None = None, target_year: int | None = None) -> dict:
    """
    Пересчитывает и сохраняет snapshot-прогнозы на target_year.
    Источник признаков: KPIRecord за base_year.
    """
    from .models import KPIRecord, PredictionSnapshot

    all_years = list(KPIRecord.objects.order_by("year").values_list("year", flat=True).distinct())
    if not all_years:
        raise ValueError("No KPI data found to build snapshots")

    if base_year is None:
        base_year = all_years[-1]
    if target_year is None:
        target_year = base_year + 1

    excluded = _excluded_roles()
    
    # Фильтруем преподавателей, у которых есть хотя бы 2 года истории
    from django.db.models import Count
    valid_teachers = set(
        KPIRecord.objects.values("teacher_id")
        .annotate(years_count=Count("year", distinct=True))
        .filter(years_count__gte=2)
        .values_list("teacher_id", flat=True)
    )

    base_qs = KPIRecord.objects.filter(
        year=base_year,
        teacher_id__in=valid_teachers
    ).exclude(role__in=excluded).values(
        "teacher_id",
        "department",
        "role",
        "experience_years",
        *BLOCK_FEATURES,
    )

    base_df = pd.DataFrame(base_qs)
    if base_df.empty:
        raise ValueError(f"No KPI records found for base_year={base_year}")

    base_df = _normalize_dataframe_blocks(base_df)

    pipeline = get_active_pipeline()
    model_version = get_active_model_version()
    predicted = pipeline.predict(base_df[FEATURES])
    predicted = np.clip(predicted, 0, 100)

    pred_df = base_df.copy()
    pred_df["predicted_kpi"] = np.round(predicted.astype(float), 2)

    snapshots_to_create = []

    # Overall snapshot
    overall_value = round(float(pred_df["predicted_kpi"].mean()), 2)
    snapshots_to_create.append(
        PredictionSnapshot(
            target_year=target_year,
            base_year=base_year,
            scope="overall",
            scope_key=_snapshot_scope_key("overall"),
            predicted_kpi=overall_value,
            records_count=int(len(pred_df)),
            model_version=model_version,
        )
    )

    # Department snapshots
    dep_count = 0
    dep_groups = pred_df.groupby("department", dropna=False)["predicted_kpi"].agg(["mean", "count"]).reset_index()
    for _, row in dep_groups.iterrows():
        department = str(row["department"] or "").strip()
        snapshots_to_create.append(
            PredictionSnapshot(
                target_year=target_year,
                base_year=base_year,
                scope="department",
                scope_key=_snapshot_scope_key("department", department=department),
                department=department,
                predicted_kpi=round(float(row["mean"]), 2),
                records_count=int(row["count"]),
                model_version=model_version,
            )
        )
        dep_count += 1

    # Role snapshots
    role_count = 0
    role_groups = pred_df.groupby("role", dropna=False)["predicted_kpi"].agg(["mean", "count"]).reset_index()
    for _, row in role_groups.iterrows():
        role = str(row["role"] or "").strip()
        snapshots_to_create.append(
            PredictionSnapshot(
                target_year=target_year,
                base_year=base_year,
                scope="role",
                scope_key=_snapshot_scope_key("role", role=role),
                role=role,
                predicted_kpi=round(float(row["mean"]), 2),
                records_count=int(row["count"]),
                model_version=model_version,
            )
        )
        role_count += 1

    # Department + role snapshots
    dep_role_count = 0
    dep_role_groups = (
        pred_df.groupby(["department", "role"], dropna=False)["predicted_kpi"]
        .agg(["mean", "count"])
        .reset_index()
    )
    for _, row in dep_role_groups.iterrows():
        department = str(row["department"] or "").strip()
        role = str(row["role"] or "").strip()
        snapshots_to_create.append(
            PredictionSnapshot(
                target_year=target_year,
                base_year=base_year,
                scope="department_role",
                scope_key=_snapshot_scope_key("department_role", department=department, role=role),
                department=department,
                role=role,
                predicted_kpi=round(float(row["mean"]), 2),
                records_count=int(row["count"]),
                model_version=model_version,
            )
        )
        dep_role_count += 1

    # Per-teacher snapshots (for point lookup on dashboard)
    teacher_count = 0
    for _, row in pred_df.iterrows():
        teacher_id = int(row["teacher_id"])
        department = str(row.get("department") or "").strip()
        role = str(row.get("role") or "").strip()
        snapshots_to_create.append(
            PredictionSnapshot(
                target_year=target_year,
                base_year=base_year,
                scope="teacher",
                scope_key=_snapshot_scope_key("teacher", teacher_id=teacher_id),
                department=department,
                role=role,
                teacher_id=teacher_id,
                predicted_kpi=round(float(row["predicted_kpi"]), 2),
                records_count=1,
                model_version=model_version,
            )
        )
        teacher_count += 1

    with transaction.atomic():
        PredictionSnapshot.objects.filter(target_year=target_year).delete()
        PredictionSnapshot.objects.bulk_create(snapshots_to_create, batch_size=1000)

    return {
        "status": "ok",
        "base_year": base_year,
        "target_year": target_year,
        "model_version": model_version.version if model_version else None,
        "total_snapshots": len(snapshots_to_create),
        "counts": {
            "overall": 1,
            "department": dep_count,
            "role": role_count,
            "department_role": dep_role_count,
            "teacher": teacher_count,
        },
    }


def get_prediction_snapshots(
    *,
    target_year: int,
    department: str = "",
    role: str = "",
    teacher_id: int | None = None,
) -> dict:
    """Возвращает предрасчитанные snapshot-прогнозы на target_year."""
    from .models import PredictionSnapshot

    qs = PredictionSnapshot.objects.filter(target_year=target_year)
    if not qs.exists():
        raise ValueError(f"No prediction snapshots found for year={target_year}")

    def _row(item: PredictionSnapshot) -> dict:
        return {
            "scope": item.scope,
            "predicted_kpi": item.predicted_kpi,
            "records_count": item.records_count,
            "department": item.department,
            "role": item.role,
            "teacher_id": item.teacher_id,
        }

    overall_item = qs.filter(scope="overall").order_by("-updated_at").first()
    overall = _row(overall_item) if overall_item else None

    by_department = [_row(i) for i in qs.filter(scope="department").order_by("department")]
    by_role = [_row(i) for i in qs.filter(scope="role").order_by("role")]
    by_department_role = [_row(i) for i in qs.filter(scope="department_role").order_by("department", "role")]

    filtered_prediction = None
    dep = (department or "").strip()
    rl = (role or "").strip()
    if dep and rl:
        item = qs.filter(scope="department_role", department__iexact=dep, role__iexact=rl).first()
        filtered_prediction = _row(item) if item else None
    elif dep:
        item = qs.filter(scope="department", department__iexact=dep).first()
        filtered_prediction = _row(item) if item else None
    elif rl:
        item = qs.filter(scope="role", role__iexact=rl).first()
        filtered_prediction = _row(item) if item else None

    teacher_prediction = None
    if teacher_id is not None:
        item = qs.filter(scope="teacher", teacher_id=teacher_id).first()
        teacher_prediction = _row(item) if item else None

    latest = qs.order_by("-updated_at").first()
    model_version = None
    if latest and latest.model_version:
        model_version = latest.model_version.version

    return {
        "year": target_year,
        "model_version": model_version,
        "generated_at": latest.updated_at if latest else None,
        "overall": overall,
        "by_department": by_department,
        "by_role": by_role,
        "by_department_role": by_department_role,
        "filtered_prediction": filtered_prediction,
        "teacher_prediction": teacher_prediction,
    }


def _refresh_year_snapshots_safe(year: int) -> dict:
    try:
        return rebuild_prediction_snapshots(base_year=year, target_year=year + 1)
    except Exception as exc:
        logger.exception("Snapshot refresh failed after finalize")
        return {
            "status": "failed",
            "target_year": year + 1,
            "error": str(exc),
        }


# ──────────────────────────────────────────────
# FINALIZATION (конец года)
# ──────────────────────────────────────────────

def finalize_year(year: int, records: list[dict], idempotency_key: str) -> dict:
    """
    Принимает финализированные оценки за year,
    сохраняет в KPIRecord, переобучает модель.
    """
    from .models import FinalizeRequest, KPIRecord

    if not idempotency_key:
        raise ValueError("X-Idempotency-Key is required")

    existing_request = FinalizeRequest.objects.filter(idempotency_key=idempotency_key).first()
    if existing_request:
        if existing_request.status == "completed":
            replay_payload = dict(existing_request.response_payload or {})
            replay_payload["idempotency_replay"] = True
            return replay_payload
        if existing_request.status == "processing":
            return {
                "status": "processing",
                "message": "Finalize request with this idempotency key is already processing",
                "idempotency_key": idempotency_key,
            }
        return {
            "status": "failed",
            "message": existing_request.error_message or "Previous finalize attempt failed",
            "idempotency_key": idempotency_key,
        }

    try:
        request_row = FinalizeRequest.objects.create(
            idempotency_key=idempotency_key,
            year=year,
            status="processing",
        )
    except IntegrityError:
        replay = FinalizeRequest.objects.get(idempotency_key=idempotency_key)
        payload = dict(replay.response_payload or {})
        payload["idempotency_replay"] = True
        return payload

    excluded = _excluded_roles()
    filtered_records = [r for r in records if _normalize_role_name(r.get("role", "")) not in excluded]

    if not filtered_records:
        payload = {
            "status": "skipped",
            "message": "All incoming records were excluded by role policy",
            "new_records": 0,
            "idempotency_key": idempotency_key,
        }
        request_row.status = "completed"
        request_row.response_payload = payload
        request_row.save(update_fields=["status", "response_payload", "updated_at"])
        return payload

    existing_ids = set(
        KPIRecord.objects.filter(year=year).values_list("teacher_id", flat=True)
    )

    new_records = [r for r in filtered_records if r.get("teacher_id") not in existing_ids]
    new_records = [_normalize_record_blocks(r) for r in new_records]

    if not new_records:
        snapshot_refresh = _refresh_year_snapshots_safe(year)
        payload = {
            "status": "skipped",
            "message": f"All records for {year} already exist",
            "new_records": 0,
            "idempotency_key": idempotency_key,
            "snapshots": snapshot_refresh,
        }
        request_row.status = "completed"
        request_row.response_payload = payload
        request_row.save(update_fields=["status", "response_payload", "updated_at"])
        return payload

    try:
        with transaction.atomic():
            KPIRecord.objects.bulk_create([
                KPIRecord(
                    teacher_id       = r["teacher_id"],
                    full_name        = r.get("full_name", ""),
                    department       = r["department"],
                    role             = r["role"],
                    year             = year,
                    experience_years = r.get("experience_years", 1),
                    block_1          = _safe_float(r.get("block_1", 0), default=0.0),
                    block_2          = _safe_float(r.get("block_2", 0), default=0.0),
                    block_3          = _safe_float(r.get("block_3", 0), default=0.0),
                    block_4          = _safe_float(r.get("block_4", 0), default=0.0),
                    total_kpi        = float(r["total_kpi"]),
                )
                for r in new_records
            ], ignore_conflicts=True)

        logger.info(f"Saved up to {len(new_records)} new records for year {year}")

        # Все доступные годы для обучения
        from .models import KPIRecord as KR
        all_years = list(
            KR.objects.order_by("year").values_list("year", flat=True).distinct()
        )
        train_years = [y for y in all_years if y < year]
        test_year   = year

        if len(train_years) < 2:
            snapshot_refresh = _refresh_year_snapshots_safe(year)
            payload = {
                "status":  "data_saved",
                "message": f"Saved {len(new_records)} records. Not enough years to retrain yet.",
                "new_records": len(new_records),
                "idempotency_key": idempotency_key,
                "snapshots": snapshot_refresh,
            }
            request_row.status = "completed"
            request_row.response_payload = payload
            request_row.save(update_fields=["status", "response_payload", "updated_at"])
            return payload

        version = train_and_save(train_years, test_year, model_type="random_forest")
        snapshot_refresh = _refresh_year_snapshots_safe(year)

        payload = {
            "status":       "retrained",
            "new_records":  len(new_records),
            "model_version": version.version,
            "idempotency_key": idempotency_key,
            "metrics": {
                "MAE":  version.mae,
                "RMSE": version.rmse,
                "R2":   version.r2,
            },
            "snapshots": snapshot_refresh,
        }
        request_row.status = "completed"
        request_row.response_payload = payload
        request_row.save(update_fields=["status", "response_payload", "updated_at"])
        return payload
    except Exception as exc:
        request_row.status = "failed"
        request_row.error_message = str(exc)
        request_row.save(update_fields=["status", "error_message", "updated_at"])
        raise
