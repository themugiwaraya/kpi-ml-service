import os
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DEBUG=(bool, False),
)

env_file = BASE_DIR / ".env"
if env_file.exists():
    environ.Env.read_env(str(env_file))

SECRET_KEY = env("SECRET_KEY", default="unsafe-dev-secret-key-change-me")
DEBUG = env.bool("DEBUG", default=False)
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["*"])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "predictions",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

SESSION_COOKIE_NAME = "ml_sessionid"
CSRF_COOKIE_NAME = "ml_csrftoken"

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

db_config = env.db(
    "DATABASE_URL",
    default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
)

db_search_path = env("DB_SEARCH_PATH", default="").strip()
DB_SEARCH_PATH = db_search_path
if "postgresql" in (db_config.get("ENGINE") or "") and db_search_path:
    db_options = db_config.setdefault("OPTIONS", {})
    existing_options = (db_options.get("options") or "").strip()
    search_path_option = f"-c search_path={db_search_path}"
    db_options["options"] = (
        f"{existing_options} {search_path_option}".strip()
        if existing_options
        else search_path_option
    )

DATABASES = {
    "default": db_config,
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

ML_API_KEY = env("ML_API_KEY", default="local-ml-key")
SUPABASE_URL = env("SUPABASE_URL", default="")
SUPABASE_SERVICE_KEY = env("SUPABASE_SERVICE_KEY", default="")
SUPABASE_BUCKET = env("SUPABASE_BUCKET", default="ml-models")

MODEL_CACHE_DIR = env("MODEL_CACHE_DIR", default=str(BASE_DIR / ".model_cache"))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

ML_MAX_PREDICT_BATCH = env.int("ML_MAX_PREDICT_BATCH", default=1000)
ML_EXCLUDED_ROLES = env.list("ML_EXCLUDED_ROLES", default=["HR", "ADMIN", "COMMISSION"])
