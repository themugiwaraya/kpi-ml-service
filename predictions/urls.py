from django.urls import path

from .views import AnalyticsView, FinalizeView, HealthView, MetricsView, PredictView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("finalize/", FinalizeView.as_view(), name="finalize"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
    path("analytics/", AnalyticsView.as_view(), name="analytics"),
]
