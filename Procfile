web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
worker: celery -A celery_worker.celery_app worker --loglevel=info
beat: celery -A celery_worker.celery_app beat --loglevel=info
