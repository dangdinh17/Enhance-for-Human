#!/usr/bin/env bash
# Install/update the shared controller without changing any ML environment.
set -euo pipefail

researchctl_source="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
researchctl_data="${XDG_DATA_HOME:-$HOME/.local/share}/researchctl"
researchctl_bin="${HOME}/.local/bin"
researchctl_python="${RESEARCHCTL_INSTALL_PYTHON:-python3}"
researchctl_build="$(mktemp -d -t researchctl-wheel-XXXXXXXX)"

"${researchctl_python}" -c 'import sys; assert sys.version_info >= (3,10), "researchctl requires Python >=3.10"'
"${researchctl_python}" -m pip wheel --no-deps --no-build-isolation "${researchctl_source}" -w "${researchctl_build}"
if [[ ! -x "${researchctl_data}/venv/bin/python" ]]; then
  "${researchctl_python}" -m venv "${researchctl_data}/venv"
fi
if ! "${researchctl_data}/venv/bin/python" -c 'import yaml; assert int(yaml.__version__.split(".")[0]) >= 6' 2>/dev/null; then
  "${researchctl_data}/venv/bin/python" -m pip install 'PyYAML>=6.0'
fi
"${researchctl_data}/venv/bin/python" -m pip install --no-deps --force-reinstall "${researchctl_build}"/researchctl-*.whl
mkdir -p "${researchctl_bin}"
if [[ -e "${researchctl_bin}/researchctl" && ! -L "${researchctl_bin}/researchctl" ]]; then
  echo "Refusing to overwrite existing non-symlink ${researchctl_bin}/researchctl" >&2
  exit 1
fi
ln -sfn "${researchctl_data}/venv/bin/researchctl" "${researchctl_bin}/researchctl"
"${researchctl_bin}/researchctl" --version
echo "Wheel retained at ${researchctl_build}; add ${researchctl_bin} to PATH if needed."
