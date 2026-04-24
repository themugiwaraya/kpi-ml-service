from django.core.management.base import BaseCommand, CommandError

from predictions.models import KPIRecord
from predictions.services import train_and_save


class Command(BaseCommand):
    help = "Train and activate ML model from KPIRecord data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-type",
            type=str,
            default="random_forest",
            choices=["random_forest", "linear_regression"],
            help="Model type",
        )

    def handle(self, *args, **options):
        model_type = options["model_type"]

        # Reset default ordering before distinct to get truly unique years.
        all_years = list(
            KPIRecord.objects.order_by("year").values_list("year", flat=True).distinct()
        )
        if len(all_years) < 3:
            raise CommandError("Need at least 3 distinct years in KPIRecord to train/test reliably")

        train_years = all_years[:-1]
        test_year = all_years[-1]

        version = train_and_save(train_years=train_years, test_year=test_year, model_type=model_type)
        self.stdout.write(
            self.style.SUCCESS(
                f"Trained {version.version} ({version.model_type}) with R2={version.r2}, RMSE={version.rmse}, MAE={version.mae}"
            )
        )
