from django.db import models


class KPIRecord(models.Model):
    """
    Датасет KPI — хранится в Supabase PostgreSQL.
    Сюда грузится начальный CSV и дописываются данные каждый год.
    """
    teacher_id       = models.IntegerField()
    full_name        = models.CharField(max_length=200, blank=True)
    department       = models.CharField(max_length=200)
    role             = models.CharField(max_length=50)
    year             = models.IntegerField()
    experience_years = models.IntegerField(default=1)

    block_1 = models.FloatField(default=0)
    block_2 = models.FloatField(default=0)
    block_3 = models.FloatField(default=0)
    block_4 = models.FloatField(default=0)

    total_kpi = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "kpi_dataset"
        unique_together = ("teacher_id", "year")
        ordering = ["year", "teacher_id"]

    def __str__(self):
        return f"{self.full_name} | {self.role} | {self.year} | KPI={self.total_kpi}"


class ModelVersion(models.Model):
    """
    История версий обученной модели.
    pkl-файл хранится в Supabase Storage.
    """
    STATUS_CHOICES = [
        ("training", "Training"),
        ("active",   "Active"),
        ("archived", "Archived"),
    ]

    version       = models.CharField(max_length=50)
    model_type    = models.CharField(max_length=50)
    storage_path  = models.CharField(max_length=255)
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default="training")

    train_years   = models.CharField(max_length=100)
    train_records = models.IntegerField(default=0)

    test_year     = models.IntegerField(null=True, blank=True)
    mae           = models.FloatField(null=True, blank=True)
    rmse          = models.FloatField(null=True, blank=True)
    r2            = models.FloatField(null=True, blank=True)

    created_at    = models.DateTimeField(auto_now_add=True)
    trained_at    = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "ml_model_versions"
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["version"], name="uq_ml_model_versions_version"),
        ]

    def __str__(self):
        return f"{self.version} | {self.model_type} | {self.status} | R2={self.r2}"


class PredictionLog(models.Model):
    """
    Лог предсказаний для аналитики.
    """
    teacher_id    = models.IntegerField()
    role          = models.CharField(max_length=50)
    department    = models.CharField(max_length=200)
    year          = models.IntegerField()

    block_1       = models.FloatField(default=0)
    block_2       = models.FloatField(default=0)
    block_3       = models.FloatField(default=0)
    block_4       = models.FloatField(default=0)

    predicted_kpi = models.FloatField()
    current_sum   = models.FloatField()
    gap           = models.FloatField()
    request_id    = models.CharField(max_length=64, blank=True, db_index=True)

    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="predictions"
    )
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "ml_prediction_log"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["teacher_id", "created_at"], name="idx_ml_pred_teacher_created"),
            models.Index(fields=["year", "created_at"], name="idx_ml_pred_year_created"),
        ]


class FinalizeRequest(models.Model):
    """
    Идемпотентность и аудит finalize-запросов.
    Один X-Idempotency-Key должен приводить к одному итоговому результату.
    """
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    idempotency_key = models.CharField(max_length=100, unique=True)
    year            = models.IntegerField()
    status          = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    response_payload = models.JSONField(default=dict, blank=True)
    error_message   = models.TextField(blank=True)
    created_at      = models.DateTimeField(auto_now_add=True)
    updated_at      = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ml_finalize_requests"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["year", "created_at"], name="idx_ml_finalize_year_created"),
        ]

    def __str__(self):
        return f"{self.idempotency_key} | {self.year} | {self.status}"


class PredictionSnapshot(models.Model):
    """
    Предрасчитанные прогнозы на целевой год по разным срезам.
    Используются для быстрых ответов фронту без повторного запуска модели.
    """

    SCOPE_CHOICES = [
        ("overall", "Overall"),
        ("department", "Department"),
        ("role", "Role"),
        ("department_role", "Department + Role"),
        ("teacher", "Teacher"),
    ]

    target_year = models.IntegerField()
    base_year = models.IntegerField()
    scope = models.CharField(max_length=30, choices=SCOPE_CHOICES)
    scope_key = models.CharField(max_length=255)

    department = models.CharField(max_length=200, blank=True)
    role = models.CharField(max_length=50, blank=True)
    teacher_id = models.IntegerField(null=True, blank=True)

    predicted_kpi = models.FloatField()
    records_count = models.IntegerField(default=0)

    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="snapshots",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ml_prediction_snapshots"
        ordering = ["-target_year", "scope", "scope_key"]
        constraints = [
            models.UniqueConstraint(
                fields=["target_year", "scope", "scope_key"],
                name="uq_ml_prediction_snapshots_target_scope_key",
            ),
        ]
        indexes = [
            models.Index(fields=["target_year", "scope"], name="idx_ml_snapshots_year_scope"),
            models.Index(fields=["department", "target_year"], name="idx_ml_snapshots_department_year"),
            models.Index(fields=["role", "target_year"], name="idx_ml_snapshots_role_year"),
            models.Index(fields=["teacher_id", "target_year"], name="idx_ml_snapshots_teacher_year"),
        ]

    def __str__(self):
        return f"{self.target_year} | {self.scope} | {self.scope_key} | {self.predicted_kpi}"
