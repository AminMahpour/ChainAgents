#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
scratch_root="$(mktemp -d "${TMPDIR:-/tmp}/chainagents-wheel-check.XXXXXX")"
trap 'rm -rf "${scratch_root}"' EXIT

distribution_directory="${scratch_root}/dist"
requirements_file="${scratch_root}/runtime-requirements.txt"
virtual_environment="${scratch_root}/venv"
user_workspace="${scratch_root}/user-workspace"

mkdir -p "${distribution_directory}" "${user_workspace}"
cd "${repository_root}"

uv build --wheel --no-sources --out-dir "${distribution_directory}"
uv export \
  --locked \
  --no-dev \
  --no-emit-project \
  --no-annotate \
  --output-file "${requirements_file}" \
  >/dev/null
uv venv --python 3.12 "${virtual_environment}"
uv pip install \
  --python "${virtual_environment}/bin/python" \
  --requirements "${requirements_file}" \
  --no-progress
uv pip install \
  --python "${virtual_environment}/bin/python" \
  --no-deps \
  "${distribution_directory}"/*.whl

cd "${user_workspace}"

PYTHONPATH="" DEEPAGENT_CONFIG="deepagent.toml" \
  "${virtual_environment}/bin/python" \
  "${repository_root}/scripts/installed_wheel_smoke.py"
PYTHONPATH="" "${virtual_environment}/bin/chainagents" --help >/dev/null
PYTHONPATH="" "${virtual_environment}/bin/chainagents-api" --help >/dev/null

echo "installed wheel CLI and API help checks passed"
