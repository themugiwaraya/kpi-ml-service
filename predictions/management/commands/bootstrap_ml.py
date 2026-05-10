import os
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection

from predictions.models import KPIRecord, ModelVersion


TRUE_VALUES = {"1", "true", "yes", "on"}
VALID_MODEL_TYPES = ("random_forest", "linear_regression", "decision_tree")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def _parse_model_types(raw: str) -> list[str]:
    if not raw:
        return []

    parsed: list[str] = []
    for item in raw.split(","):
        model_type = item.strip().lower()
        if not model_type:
            continue
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Allowed: {', '.join(VALID_MODEL_TYPES)}"
            )
        if model_type not in parsed:
            parsed.append(model_type)
    return parsed


class Command(BaseCommand):
    help = "Bootstrap ML service data/model on startup without interactive shell"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            type=str,
            default="kpi_dataset.csv",
            help="Path to seed dataset CSV",
        )
        parser.add_argument(
            "--model-types",
            type=str,
            default=os.getenv("ML_BOOTSTRAP_MODEL_TYPES", ""),
            help="Comma-separated model types to train on first bootstrap",
        )
        parser.add_argument(
            "--model-type",
            type=str,
            default=os.getenv("ML_BOOTSTRAP_MODEL_TYPE", "random_forest"),
            choices=VALID_MODEL_TYPES,
            help="Initial model type to train if no active model exists",
        )
        parser.add_argument(
            "--primary-model-type",
            type=str,
            default=os.getenv("ML_BOOTSTRAP_PRIMARY_MODEL_TYPE", ""),
            choices=VALID_MODEL_TYPES,
            help="Model type that should remain active after first bootstrap",
        )
        parser.add_argument("--skip-load", action="store_true", help="Skip dataset load step")
        parser.add_argument("--skip-train", action="store_true", help="Skip initial training step")

    def handle(self, *args, **options):
        strict = _env_bool("ML_BOOTSTRAP_STRICT", False)
        load_enabled = _env_bool("ML_BOOTSTRAP_LOAD_DATASET", True) and not options["skip_load"]
        train_enabled = _env_bool("ML_BOOTSTRAP_TRAIN", True) and not options["skip_train"]

        dataset_path = Path(options["dataset"])
        model_types = _parse_model_types(options.get("model_types") or "")
        fallback_model_type = options["model_type"]
        if not model_types:
            model_types = [fallback_model_type]

        primary_model_type = (options.get("primary_model_type") or fallback_model_type).strip().lower()
        if primary_model_type not in model_types:
            model_types.append(primary_model_type)

        self.stdout.write("[bootstrap_ml] Starting bootstrap checks")
        self._log_db_context()

        try:
            if load_enabled:
                self._load_dataset_if_needed(dataset_path)
            else:
                self.stdout.write("[bootstrap_ml] Dataset load step skipped")

            if train_enabled:
                self._train_models_if_needed(model_types, primary_model_type)
            else:
                self.stdout.write("[bootstrap_ml] Model training step skipped")
        except Exception as exc:  # pragma: no cover - defensive startup guard
            if strict:
                raise
            self.stderr.write(self.style.WARNING(f"[bootstrap_ml] Non-fatal error: {exc}"))

        self.stdout.write(self.style.SUCCESS("[bootstrap_ml] Bootstrap finished"))

    def _log_db_context(self):
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT current_database(), current_user, current_schema(), current_setting('search_path')"
                )
                current_database, current_user, current_schema, search_path = cursor.fetchone()
            self.stdout.write(
                "[bootstrap_ml] DB context: "
                f"database={current_database}, user={current_user}, "
                f"current_schema={current_schema}, search_path={search_path}"
            )
        except Exception as exc:
            self.stderr.write(self.style.WARNING(f"[bootstrap_ml] DB context probe failed: {exc}"))

    def _load_dataset_if_needed(self, dataset_path: Path):
        if KPIRecord.objects.exists():
            self.stdout.write("[bootstrap_ml] KPIRecord already has data; skip load")
            return

        if not dataset_path.exists():
            self.stderr.write(
                self.style.WARNING(
                    f"[bootstrap_ml] Seed CSV not found: {dataset_path}. Skipping dataset load."
                )
            )
            return

        self.stdout.write(f"[bootstrap_ml] Loading seed dataset from {dataset_path}")
        call_command("load_dataset", csv=str(dataset_path))

    def _train_models_if_needed(self, model_types: list[str], primary_model_type: str):
        existing_types = set(
            ModelVersion.objects.values_list("model_type", flat=True).distinct()
        )
        missing_types = [model_type for model_type in model_types if model_type not in existing_types]

        if not missing_types:
            self.stdout.write("[bootstrap_ml] All configured model types already exist; skip training")
            self._ensure_primary_active(primary_model_type)
            return

        years = list(
            KPIRecord.objects.order_by("year").values_list("year", flat=True).distinct()
        )
        if len(years) < 3:
            self.stderr.write(
                self.style.WARNING(
                    "[bootstrap_ml] Need at least 3 distinct years to train first model; skipping training"
                )
            )
            return

        ordered_types = [m for m in missing_types if m != primary_model_type]
        if primary_model_type in missing_types:
            ordered_types.append(primary_model_type)

        self.stdout.write(f"[bootstrap_ml] Training missing models: {', '.join(ordered_types)}")
        for model_type in ordered_types:
            self.stdout.write(f"[bootstrap_ml] Training initial model ({model_type})")
            call_command("train_model", model_type=model_type)

        self._ensure_primary_active(primary_model_type)

    def _ensure_primary_active(self, primary_model_type: str):
        latest_primary = ModelVersion.objects.filter(model_type=primary_model_type).order_by(
            "-trained_at", "-created_at"
        ).first()
        if not latest_primary:
            self.stderr.write(
                self.style.WARNING(
                    f"[bootstrap_ml] Primary model type '{primary_model_type}' not found; keep current active model"
                )
            )
            return

        current_active = ModelVersion.objects.filter(status="active").order_by(
            "-trained_at", "-created_at"
        ).first()
        if current_active and current_active.id == latest_primary.id:
            self.stdout.write(
                f"[bootstrap_ml] Primary model already active: {latest_primary.version}"
            )
            return

        ModelVersion.objects.filter(status="active").exclude(id=latest_primary.id).update(status="archived")
        if latest_primary.status != "active":
            latest_primary.status = "active"
            latest_primary.save(update_fields=["status"])

        self.stdout.write(
            f"[bootstrap_ml] Active model set to primary type '{primary_model_type}': {latest_primary.version}"
        )
