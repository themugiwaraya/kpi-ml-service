"""
ФАЙЛ ДЛЯ ОСНОВНОГО DJANGO ПРОЕКТА
Положи в: your_main_project/kpi/services/ml_client.py

Установи в основной проект:
  pip install requests

Добавь в settings.py основного проекта:
  ML_SERVICE_URL = env("ML_SERVICE_URL")   # https://kpi-ml-service.onrender.com
  ML_API_KEY     = env("ML_API_KEY")       # тот же ключ что и в ML сервисе

Добавь в .env основного проекта:
  ML_SERVICE_URL=https://kpi-ml-service.onrender.com
  ML_API_KEY=super-secret-key-shared-with-main-backend
"""
import logging
import uuid
import requests
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ML_SERVICE_URL = getattr(settings, "ML_SERVICE_URL", "http://localhost:8001")
ML_API_KEY     = getattr(settings, "ML_API_KEY",     "")
ML_CONNECT_TIMEOUT = float(getattr(settings, "ML_CONNECT_TIMEOUT", 2))
ML_READ_TIMEOUT = float(getattr(settings, "ML_READ_TIMEOUT", 10))


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


def _headers(extra: dict | None = None):
    headers = {
        "X-API-Key": ML_API_KEY,
        "X-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


# ──────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────

def predict_kpi(teacher, kpi_blocks: dict, year: int) -> dict | None:
    """
    teacher    — Django model instance или dict с полями: id, role, department, experience_years
    kpi_blocks — dict: { "block_1": 20.0, "block_2": 65.0, "block_3": 0, "block_4": 0 }
    year       — текущий год

    Возвращает:
    {
        "teacher_id":       42,
        "predicted_kpi":    85.3,
        "current_sum":      83.0,
        "gap":              2.3,
        "gap_interpretation": "On track"
    }
    """
    def get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    payload = {
        "teacher_id":      get(teacher, "id"),
        "role":            get(teacher, "role"),
        "department":      get(teacher, "department", "N/A"),
        "experience_years": get(teacher, "experience_years", 1),
        "year":            year,
        "block_1":         float(kpi_blocks.get("block_1", 0)),
        "block_2":         float(kpi_blocks.get("block_2", 0)),
        "block_3":         float(kpi_blocks.get("block_3", 0)),
        "block_4":         float(kpi_blocks.get("block_4", 0)),
    }

    try:
        resp = _SESSION.post(
            f"{ML_SERVICE_URL}/api/predict/",
            json=payload,
            headers=_headers(),
            timeout=(ML_CONNECT_TIMEOUT, ML_READ_TIMEOUT),
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"ML service unavailable: {e}")
        return None   # graceful degradation — UI просто не показывает предикшн


def predict_kpi_batch(items: list[dict]) -> list[dict]:
    """
    items = [
        { "teacher": teacher_obj, "blocks": {...}, "year": 2026 },
        ...
    ]
    """
    payloads = []
    for item in items:
        t = item["teacher"]
        b = item["blocks"]

        def get(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

        payloads.append({
            "teacher_id":      get(t, "id"),
            "role":            get(t, "role"),
            "department":      get(t, "department", "N/A"),
            "experience_years": get(t, "experience_years", 1),
            "year":            item.get("year", 0),
            "block_1":         float(b.get("block_1", 0)),
            "block_2":         float(b.get("block_2", 0)),
            "block_3":         float(b.get("block_3", 0)),
            "block_4":         float(b.get("block_4", 0)),
        })

    try:
        resp = _SESSION.post(
            f"{ML_SERVICE_URL}/api/predict/",
            json=payloads,
            headers=_headers(),
            timeout=(ML_CONNECT_TIMEOUT, max(ML_READ_TIMEOUT, 30)),
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"ML service batch predict failed: {e}")
        return []


# ──────────────────────────────────────────────
# FINALIZE (вызывается из Django Admin основного проекта)
# ──────────────────────────────────────────────

def finalize_year(year: int, evaluations_queryset) -> dict:
    """
    evaluations_queryset — QuerySet финализированных оценок из основного Django.
    Каждый объект должен иметь поля:
      teacher.id, teacher.role, teacher.department, teacher.experience_years,
      block_1, block_2, block_3, block_4, total_kpi

    Пример вызова из Django Admin основного проекта:
      from kpi.services.ml_client import finalize_year
      result = finalize_year(2025, KPIEvaluation.objects.filter(year=2025, status="finalized"))
    """
    records = []
    for ev in evaluations_queryset:
        records.append({
            "teacher_id":      ev.teacher.id,
            "full_name":       getattr(ev.teacher, "full_name", ""),
            "department":      ev.teacher.department,
            "role":            ev.teacher.role,
            "experience_years": getattr(ev.teacher, "experience_years", 1),
            "block_1":         float(getattr(ev, "block_1", 0) or 0),
            "block_2":         float(getattr(ev, "block_2", 0) or 0),
            "block_3":         float(getattr(ev, "block_3", 0) or 0),
            "block_4":         float(getattr(ev, "block_4", 0) or 0),
            "total_kpi":       float(getattr(ev, "total_kpi", 0) or 0),
        })

    try:
        resp = _SESSION.post(
            f"{ML_SERVICE_URL}/api/finalize/",
            json={"year": year, "records": records},
            headers=_headers({"X-Idempotency-Key": str(uuid.uuid4())}),
            timeout=(ML_CONNECT_TIMEOUT, max(ML_READ_TIMEOUT, 20)),
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"ML service finalize failed: {e}")
        raise


def get_health() -> dict:
    try:
        resp = _SESSION.get(
            f"{ML_SERVICE_URL}/api/health/",
            headers=_headers(),
            timeout=(ML_CONNECT_TIMEOUT, 5),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unavailable"}


def get_kpi_analytics(department: str | None = None, role: str | None = None, teacher_id: int | None = None) -> dict | None:
    """
    Возвращает historical + prediction из ML сервиса.

    Пример ответа:
    {
      "historical": {"2021": 86.1, "2022": 87.4},
      "prediction": {"year": 2026, "value": 92.5, ...}
    }
    """
    params = {}
    if department:
        params["department"] = department
    if role:
        params["role"] = role
    if teacher_id is not None:
        params["teacher_id"] = int(teacher_id)

    if not params:
        raise ValueError("At least one filter is required: department, role, teacher_id")

    try:
        resp = _SESSION.get(
            f"{ML_SERVICE_URL}/api/analytics/",
            params=params,
            headers=_headers(),
            timeout=(ML_CONNECT_TIMEOUT, ML_READ_TIMEOUT),
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"ML service analytics failed: {e}")
        return None
