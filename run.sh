cd /teamspace/studios/this_studio/UI-Venus

# Source environment overrides from .env when present.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Default tensor parallelism to a single GPU when not explicitly set.
export UI_VENUS_TP="${UI_VENUS_TP:-1}"

uvicorn autonomous_api:app --host 0.0.0.0 --port 8000
