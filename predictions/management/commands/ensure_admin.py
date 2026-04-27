import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.utils import OperationalError, ProgrammingError


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


class Command(BaseCommand):
    help = "Create/update Django superuser from environment variables"

    def _user_table_available(self, table_name: str) -> bool:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass(%s)", [table_name])
                row = cursor.fetchone()
            return bool(row and row[0])
        except (ProgrammingError, OperationalError):
            return False

    def handle(self, *args, **options):
        enabled = _env_bool("DJANGO_SUPERUSER_CREATE", False)
        if not enabled:
            self.stdout.write("[ensure_admin] Skipped (DJANGO_SUPERUSER_CREATE is false)")
            return

        username = (os.getenv("DJANGO_SUPERUSER_USERNAME") or "").strip()
        email = (os.getenv("DJANGO_SUPERUSER_EMAIL") or "").strip()
        password = os.getenv("DJANGO_SUPERUSER_PASSWORD") or ""
        strict = _env_bool("DJANGO_SUPERUSER_STRICT", False)
        update_password = _env_bool("DJANGO_SUPERUSER_UPDATE_PASSWORD", False)

        if not username or not password:
            message = (
                "[ensure_admin] DJANGO_SUPERUSER_USERNAME and DJANGO_SUPERUSER_PASSWORD are required "
                "when DJANGO_SUPERUSER_CREATE=true"
            )
            if strict:
                raise ValueError(message)
            self.stderr.write(self.style.WARNING(message))
            return

        User = get_user_model()
        if not self._user_table_available(User._meta.db_table):
            message = (
                f"[ensure_admin] User table '{User._meta.db_table}' is not visible in current schema/search_path. "
                "Skipping admin bootstrap."
            )
            if strict:
                raise RuntimeError(message)
            self.stderr.write(self.style.WARNING(message))
            return

        try:
            user = User.objects.filter(username=username).first()
        except (ProgrammingError, OperationalError) as exc:
            message = f"[ensure_admin] User lookup failed: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            self.stderr.write(self.style.WARNING(message))
            return

        if user is None:
            User.objects.create_superuser(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f"[ensure_admin] Superuser created: {username}"))
            return

        changed_fields = []
        if email and getattr(user, "email", "") != email:
            user.email = email
            changed_fields.append("email")

        if not user.is_staff:
            user.is_staff = True
            changed_fields.append("is_staff")

        if not user.is_superuser:
            user.is_superuser = True
            changed_fields.append("is_superuser")

        if update_password:
            user.set_password(password)
            changed_fields.append("password")

        if changed_fields:
            user.save()
            self.stdout.write(
                self.style.SUCCESS(
                    f"[ensure_admin] Superuser updated: {username} ({', '.join(changed_fields)})"
                )
            )
        else:
            self.stdout.write(f"[ensure_admin] Superuser already exists: {username}")
