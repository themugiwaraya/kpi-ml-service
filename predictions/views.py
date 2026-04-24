import logging
import hmac
import time
import uuid
from functools import wraps
from django.conf import settings
from django.db import connection
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import FinalizeInputSerializer, PredictInputSerializer
from . import services

logger = logging.getLogger(__name__)
DEFAULT_MAX_PREDICT_BATCH = 1000


# ──────────────────────────────────────────────
# API KEY AUTH (простой middleware для сервис-к-сервис)
# ──────────────────────────────────────────────

def require_api_key(func):
    @wraps(func)
    def wrapper(self, request, *args, **kwargs):
        expected = getattr(settings, "ML_API_KEY", "")
        key = request.headers.get("X-API-Key", "")
        if not expected or not hmac.compare_digest(key, expected):
            return Response({"error": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)
        return func(self, request, *args, **kwargs)
    return wrapper


def _request_id(request) -> str:
    return request.headers.get("X-Request-Id") or str(uuid.uuid4())


def _response(payload, request_id: str, status_code=status.HTTP_200_OK, extra_headers=None):
    resp = Response(payload, status=status_code)
    resp["X-Request-Id"] = request_id
    if extra_headers:
        for key, value in extra_headers.items():
            resp[key] = str(value)
    return resp


# ──────────────────────────────────────────────
# VIEWS
# ──────────────────────────────────────────────

class HealthView(APIView):
    """GET /health/ — проверка доступности сервиса"""

    def get(self, request):
        from .models import ModelVersion
        request_id = _request_id(request)
        active = None
        try:
            active = ModelVersion.objects.filter(status="active").order_by("-trained_at").first()
        except Exception:
            logger.exception("Health active model check failed")
        db_ok = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception:
            db_ok = False

        storage_configured = bool(getattr(settings, "SUPABASE_URL", "") and getattr(settings, "SUPABASE_SERVICE_KEY", ""))
        storage_ok = services.check_storage_access() if storage_configured else False

        if db_ok and (storage_ok or not storage_configured):
            status_value = "ok"
        else:
            status_value = "degraded"

        return _response({
            "status": status_value,
            "checks": {
                "db": "ok" if db_ok else "failed",
                "storage": (
                    "ok" if storage_ok else
                    "not_configured" if not storage_configured else
                    "failed"
                ),
            },
            "active_model": {
                "version":    active.version    if active else None,
                "model_type": active.model_type if active else None,
                "r2":         active.r2         if active else None,
                "trained_at": active.trained_at if active else None,
            }
        }, request_id=request_id)


class PredictView(APIView):
    """
    POST /predict/
    Принимает одного сотрудника или список.
    Header: X-API-Key: <ML_API_KEY>

    Single:
        { "teacher_id": 1, "role": "LECTURER", ... }

    Batch:
        [{ ... }, { ... }]
    """

    @require_api_key
    def post(self, request):
        request_id = _request_id(request)
        started_at = time.perf_counter()
        data = request.data

        # Определяем batch или single
        is_batch = isinstance(data, list)
        records  = data if is_batch else [data]

        max_batch = int(getattr(settings, "ML_MAX_PREDICT_BATCH", DEFAULT_MAX_PREDICT_BATCH))
        if is_batch and len(records) > max_batch:
            return _response(
                {"error": f"Batch size exceeds limit ({max_batch})"},
                request_id=request_id,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        serializer = PredictInputSerializer(data=records, many=True)
        if not serializer.is_valid():
            return _response(serializer.errors, request_id=request_id, status_code=status.HTTP_400_BAD_REQUEST)

        try:
            if is_batch:
                results = services.predict_batch(serializer.validated_data)
            else:
                results = [services.predict_one(serializer.validated_data[0])]

            # Логируем (fire-and-forget — ошибки не критичны)
            try:
                _log_predictions(results, request, request_id)
            except Exception as e:
                logger.warning(f"Prediction logging failed: {e}")

            latency_ms = int((time.perf_counter() - started_at) * 1000)
            model_version = results[0].get("model_version") if results else ""
            payload = results if is_batch else results[0]

            return _response(
                payload,
                request_id=request_id,
                extra_headers={
                    "X-Model-Version": model_version,
                    "X-Latency-Ms": latency_ms,
                },
            )

        except ValueError as e:
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as e:
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            logger.exception("Prediction error")
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class FinalizeView(APIView):
    """
    POST /finalize/
    Конец года: принять оценки, сохранить в БД, переобучить модель.
    Header: X-API-Key: <ML_API_KEY>

    Body:
    {
        "year": 2025,
        "records": [ { teacher_id, role, department, ..., total_kpi }, ... ]
    }
    """

    @require_api_key
    def post(self, request):
        request_id = _request_id(request)
        idempotency_key = request.headers.get("X-Idempotency-Key", "").strip()
        if not idempotency_key:
            return _response(
                {"error": "X-Idempotency-Key header is required"},
                request_id=request_id,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        serializer = FinalizeInputSerializer(data=request.data)
        if not serializer.is_valid():
            return _response(serializer.errors, request_id=request_id, status_code=status.HTTP_400_BAD_REQUEST)

        year    = serializer.validated_data["year"]
        records = serializer.validated_data["records"]

        logger.info(f"Finalize requested: request_id={request_id}, year={year}, records={len(records)}")

        try:
            result = services.finalize_year(year, records, idempotency_key=idempotency_key)
            status_code = status.HTTP_202_ACCEPTED if result.get("status") == "processing" else status.HTTP_200_OK
            return _response(result, request_id=request_id, status_code=status_code)
        except ValueError as e:
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("Finalize error")
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MetricsView(APIView):
    """GET /metrics/ — история версий моделей"""

    def get(self, request):
        request_id = _request_id(request)
        from .models import ModelVersion
        versions = ModelVersion.objects.all()[:10]
        data = [
            {
                "version":     v.version,
                "model_type":  v.model_type,
                "status":      v.status,
                "train_years": v.train_years,
                "test_year":   v.test_year,
                "mae":         v.mae,
                "rmse":        v.rmse,
                "r2":          v.r2,
                "trained_at":  v.trained_at,
            }
            for v in versions
        ]
        return _response(data, request_id=request_id)


class AnalyticsView(APIView):
    """
    GET /analytics/?department=...&role=...&teacher_id=...
    Возвращает historical и prediction по фильтрам.
    """

    @require_api_key
    def get(self, request):
        request_id = _request_id(request)

        department = (request.query_params.get("department") or "").strip()
        role = (request.query_params.get("role") or "").strip()
        teacher_id_raw = (request.query_params.get("teacher_id") or "").strip()

        teacher_id = None
        if teacher_id_raw:
            try:
                teacher_id = int(teacher_id_raw)
            except ValueError:
                return _response(
                    {"error": "teacher_id must be an integer"},
                    request_id=request_id,
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

        if not any([department, role, teacher_id is not None]):
            return _response(
                {"error": "At least one filter is required: department, role, teacher_id"},
                request_id=request_id,
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        try:
            payload = services.get_analytics(
                {
                    "department": department,
                    "role": role,
                    "teacher_id": teacher_id,
                }
            )
            return _response(payload, request_id=request_id)
        except ValueError as e:
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception("Analytics error")
            return _response({"error": str(e)}, request_id=request_id, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _log_predictions(results: list, request, request_id: str):
    from .models import ModelVersion, PredictionLog

    active = ModelVersion.objects.filter(status="active").first()
    year   = request.data[0].get("year") if isinstance(request.data, list) else request.data.get("year")

    PredictionLog.objects.bulk_create([
        PredictionLog(
            teacher_id    = r.get("teacher_id"),
            role          = r.get("role", ""),
            department    = r.get("department", ""),
            year          = year or 0,
            predicted_kpi = r["predicted_kpi"],
            current_sum   = r["current_sum"],
            gap           = r["gap"],
            request_id    = request_id,
            model_version = active,
        )
        for r in results
    ], ignore_conflicts=True)
