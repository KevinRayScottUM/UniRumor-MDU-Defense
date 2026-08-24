#!/usr/bin/env bash

set -euo pipefail

readonly DICC_VENV="/scr/user/kevin2002/TensorCat/.venv310"
readonly DEFENSE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

: "${WEB_RUNTIME_ROOT:?WEB_RUNTIME_ROOT is required}"
: "${PRODUCTION_RUNTIME_CONFIG_PATH:?PRODUCTION_RUNTIME_CONFIG_PATH is required}"
: "${ALLOWED_ORIGINS:?ALLOWED_ORIGINS is required}"

if [[ ! -x "${DICC_VENV}/bin/python" ]]; then
  echo "The configured DICC Python environment is unavailable." >&2
  exit 2
fi
if [[ ! -d "${WEB_RUNTIME_ROOT}" || -L "${WEB_RUNTIME_ROOT}" ]]; then
  echo "WEB_RUNTIME_ROOT must be an existing non-symlink directory." >&2
  exit 2
fi
if [[ ! -f "${PRODUCTION_RUNTIME_CONFIG_PATH}" ]]; then
  echo "PRODUCTION_RUNTIME_CONFIG_PATH must be an existing file." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${DICC_VENV}/bin/activate"
cd "${DEFENSE_ROOT}"

exec python - <<'PY'
import os
from pathlib import Path

import uvicorn

from webapp.api import create_app
from webapp.api_config import APIConfig


allowed_origins = tuple(
    origin.strip()
    for origin in os.environ["ALLOWED_ORIGINS"].split(",")
    if origin.strip()
)
config = APIConfig(
    web_runtime_root=Path(os.environ["WEB_RUNTIME_ROOT"]),
    allowed_origins=allowed_origins,
    production_runtime_config_path=Path(
        os.environ["PRODUCTION_RUNTIME_CONFIG_PATH"]
    ),
)
app = create_app(config)

uvicorn.run(
    app,
    host=os.environ.get("API_HOST", "127.0.0.1"),
    port=int(os.environ.get("API_PORT", "8000")),
    workers=1,
)
PY
