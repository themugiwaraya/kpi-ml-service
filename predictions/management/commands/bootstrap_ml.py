import os
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand

from predictions.models import KPIRecord, ModelVersion


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


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
            "--model-type",
            type=str,
            default=os.getenv("ML_BOOTSTRAP_MODEL_TYPE", "random_forest"),
            choices=["random_forest", "linear_regression"],
            help="Initial model type to train if no active model exists",
        )
        parser.add_argument("--skip-load", action="store_true", help="Skip dataset load step")
        parser.add_argument("--skip-train", action="store_true", help="Skip initial training step")

    def handle(self, *args, **options):
        strict = _env_bool("ML_BOOTSTRAP_STRICT", False)
        load_enabled = _env_bool("ML_BOOTSTRAP_LOAD_DATASET", True) and not options["skip_load"]
        train_enabled = _env_bool("ML_BOOTSTRAP_TRAIN", True) and not options["skip_train"]

        dataset_path = Path(options["dataset"])
        model_type = options["model_type"]

        self.stdout.write("[bootstrap_ml] Starting bootstrap checks")

        try:
            if load_enabled:
                self._load_dataset_if_needed(dataset_path)
            else:
                self.stdout.write("[bootstrap_ml] Dataset load step skipped")

            if train_enabled:
                self._train_model_if_needed(model_type)
            else:
                self.stdout.write("[bootstrap_ml] Model training step skipped")
        except Exception as exc:  # pragma: no cover - defensive startup guard
            if strict:
                raise
            self.stderr.write(self.style.WARNING(f"[bootstrap_ml] Non-fatal error: {exc}"))

        self.stdout.write(self.style.SUCCESS("[bootstrap_ml] Bootstrap finished"))

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

    def _train_model_if_needed(self, model_type: str):
        if ModelVersion.objects.filter(status="active").exists():
            self.stdout.write("[bootstrap_ml] Active model already exists; skip training")
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

        self.stdout.write(f"[bootstrap_ml] Training initial model ({model_type})")
        call_command("train_model", model_type=model_type)
