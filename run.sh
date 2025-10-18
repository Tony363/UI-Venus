cd /teamspace/studios/this_studio/UI-Venus

# Default tensor parallelism to a single GPU when not explicitly set.
export UI_VENUS_TP="${UI_VENUS_TP:-1}"

uvicorn autonomous_api:app --host 0.0.0.0 --port 8000
