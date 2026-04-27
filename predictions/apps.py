import logging

from django.apps import AppConfig
from django.conf import settings
from django.db.backends.signals import connection_created


logger = logging.getLogger(__name__)


def _apply_search_path(sender, connection, **kwargs):
    search_path = (getattr(settings, "DB_SEARCH_PATH", "") or "").strip()
    if not search_path:
        return

    engine = connection.settings_dict.get("ENGINE", "")
    if "postgresql" not in engine:
        return

    # Set search_path per DB session; this is reliable even when pooler ignores startup options.
    with connection.cursor() as cursor:
        cursor.execute("SELECT set_config('search_path', %s, false)", [search_path])

    logger.info("Applied DB search_path for connection: %s", search_path)


class PredictionsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictions"

    def ready(self):
        connection_created.connect(_apply_search_path, dispatch_uid="predictions.apply_search_path")
