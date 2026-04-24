from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from predictions.models import KPIRecord


class Command(BaseCommand):
    help = "Load KPI dataset CSV into KPIRecord table"

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv",
            type=str,
            default=str(Path("kpi_dataset.csv")),
            help="Path to CSV file",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["csv"])
        if not csv_path.exists():
            raise CommandError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {
            "teacher_id",
            "full_name",
            "department",
            "role",
            "year",
            "experience_years",
            "block_1",
            "block_2",
            "block_3",
            "block_4",
            "total_kpi",
        }
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise CommandError(f"Missing required columns: {missing}")

        rows = []
        for _, row in df.iterrows():
            rows.append(
                KPIRecord(
                    teacher_id=int(row["teacher_id"]),
                    full_name=str(row.get("full_name", "")),
                    department=str(row["department"]),
                    role=str(row["role"]),
                    year=int(row["year"]),
                    experience_years=int(row.get("experience_years", 1)),
                    block_1=float(row.get("block_1", 0) or 0),
                    block_2=float(row.get("block_2", 0) or 0),
                    block_3=float(row.get("block_3", 0) or 0),
                    block_4=float(row.get("block_4", 0) or 0),
                    total_kpi=float(row["total_kpi"]),
                )
            )

        KPIRecord.objects.bulk_create(rows, ignore_conflicts=True, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(f"Loaded rows (with ignore_conflicts): {len(rows)}"))
