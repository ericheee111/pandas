from __future__ import annotations

import os
import platform

_DISABLED_SETTINGS = {"0", "false", "no", "off"}


def _get_use_boostkit_fastpaths() -> bool:
    machine = platform.machine().lower()
    setting = os.environ.get("PANDAS_BOOSTKIT_FASTPATHS", "auto").strip().lower()
    return machine in {"aarch64", "arm64"} and setting not in _DISABLED_SETTINGS


USE_BOOSTKIT_FASTPATHS = _get_use_boostkit_fastpaths()
