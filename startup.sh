#!/bin/bash
# Startup command for the Azure Web App (Linux).
#
# --timeout 600: a crew run takes far longer than gunicorn's 30s default,
#   which would otherwise kill the worker mid-request.
# ${PORT:-8000}: Azure injects PORT; 8000 is the fallback for local runs.
gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} --timeout 600 main:app
