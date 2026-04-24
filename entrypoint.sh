#!/bin/sh
set -e

python manage.py migrate --noinput

BOOTSTRAP_ON_START=$(printf '%s' "${ML_BOOTSTRAP_ON_START:-true}" | tr '[:upper:]' '[:lower:]')
if [ "$BOOTSTRAP_ON_START" = "1" ] || [ "$BOOTSTRAP_ON_START" = "true" ] || [ "$BOOTSTRAP_ON_START" = "yes" ] || [ "$BOOTSTRAP_ON_START" = "on" ]; then
	python manage.py bootstrap_ml
fi

exec gunicorn config.wsgi:application --bind 0.0.0.0:8001 --workers 2 --timeout 120
