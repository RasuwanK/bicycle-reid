#!/usr/bin/env bash
set -e

# If no arguments provided, default to starting Streamlit Web UI
if [ $# -eq 0 ]; then
    exec streamlit run src/bicycle_reid/app/streamlit_app.py
fi

# If argument starts with '-' (e.g. --demo, --source-a, --ui), pass to bicycle-reid CLI
if [ "${1#-}" != "$1" ]; then
    exec bicycle-reid "$@"
fi

# Otherwise execute the given command directly (e.g. pytest, bash, streamlit, bicycle-reid)
exec "$@"
