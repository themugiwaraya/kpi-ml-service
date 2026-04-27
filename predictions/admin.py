from django.contrib import admin
from django.utils.html import format_html
from django.contrib import messages
from .models import FinalizeRequest, KPIRecord, ModelVersion, PredictionLog, PredictionSnapshot
from . import services


# ──────────────────────────────────────────────
# KPIRecord
# ──────────────────────────────────────────────

@admin.register(KPIRecord)
class KPIRecordAdmin(admin.ModelAdmin):
    list_display   = ("teacher_id", "full_name", "department", "role", "year",
                       "block_1", "block_2", "block_3", "block_4", "total_kpi")
    list_filter    = ("year", "role", "department")
    search_fields  = ("full_name", "teacher_id")
    ordering       = ("-year", "teacher_id")
    list_per_page  = 50


# ──────────────────────────────────────────────
# ModelVersion — основная админка для управления моделями
# ──────────────────────────────────────────────

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display  = ("version", "model_type", "status_badge", "train_years",
                      "test_year", "mae", "rmse", "r2", "trained_at")
    list_filter   = ("status", "model_type")
    readonly_fields = ("version", "model_type", "storage_path", "status",
                        "train_years", "train_records", "test_year",
                        "mae", "rmse", "r2", "created_at", "trained_at")
    ordering      = ("-created_at",)
    actions       = ["train_random_forest", "train_linear_regression"]

    def status_badge(self, obj):
        colors = {"active": "green", "archived": "grey", "training": "orange"}
        color  = colors.get(obj.status, "black")
        return format_html('<b style="color:{}">{}</b>', color, obj.status.upper())
    status_badge.short_description = "Status"

    # ── ACTIONS (кнопки в админке) ──────────────

    @admin.action(description="🚀 Train NEW Random Forest model")
    def train_random_forest(self, request, queryset):
        self._run_training(request, "random_forest")

    @admin.action(description="📈 Train NEW Linear Regression model")
    def train_linear_regression(self, request, queryset):
        self._run_training(request, "linear_regression")

    def _run_training(self, request, model_type: str):
        from .models import KPIRecord
        import datetime

        all_years = list(
            KPIRecord.objects.order_by("year").values_list("year", flat=True).distinct()
        )
        if len(all_years) < 2:
            self.message_user(request, "Need at least 2 years of data to train.", messages.ERROR)
            return

        train_years = all_years[:-1]
        test_year   = all_years[-1]

        try:
            version = services.train_and_save(train_years, test_year, model_type)
            self.message_user(
                request,
                f"✓ Model trained: {version.version} | "
                f"Train: {version.train_years} | Test: {test_year} | "
                f"MAE={version.mae}  RMSE={version.rmse}  R²={version.r2}",
                messages.SUCCESS,
            )
        except Exception as e:
            self.message_user(request, f"Training failed: {e}", messages.ERROR)

    # ── Кнопка "Финализация года" на странице списка ──

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["show_finalize_hint"] = True
        return super().changelist_view(request, extra_context)


# ──────────────────────────────────────────────
# PredictionLog
# ──────────────────────────────────────────────

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display  = ("teacher_id", "role", "department", "year",
                      "predicted_kpi", "current_sum", "gap", "created_at")
    list_filter   = ("role", "year")
    readonly_fields = [f.name for f in PredictionLog._meta.fields]
    ordering      = ("-created_at",)
    list_per_page = 50

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(FinalizeRequest)
class FinalizeRequestAdmin(admin.ModelAdmin):
    list_display = ("idempotency_key", "year", "status", "created_at", "updated_at")
    list_filter = ("status", "year")
    search_fields = ("idempotency_key",)
    readonly_fields = [f.name for f in FinalizeRequest._meta.fields]
    ordering = ("-created_at",)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PredictionSnapshot)
class PredictionSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        "target_year",
        "base_year",
        "scope",
        "department",
        "role",
        "teacher_id",
        "predicted_kpi",
        "records_count",
        "model_version",
        "updated_at",
    )
    list_filter = ("target_year", "scope", "department", "role")
    search_fields = ("scope_key", "department", "role", "teacher_id")
    readonly_fields = [f.name for f in PredictionSnapshot._meta.fields]
    ordering = ("-target_year", "scope", "scope_key")

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False
