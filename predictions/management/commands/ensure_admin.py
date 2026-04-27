import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


class Command(BaseCommand):
    help = "Create/update Django superuser from environment variables"

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
        user = User.objects.filter(username=username).first()

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
