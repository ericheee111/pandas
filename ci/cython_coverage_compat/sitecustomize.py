import os
from pathlib import Path

import Cython.Coverage as cycoverage


_original_find_c_source = cycoverage._find_c_source
_original_find_dep_file_path = cycoverage._find_dep_file_path
_original_configure = cycoverage.Plugin.configure
_build_dir = Path(os.environ["PANDAS_MESON_BUILD_DIR"])
_source_dir = Path(os.environ["PANDAS_SOURCE_DIR"])


def _find_meson_c_source(base_path):
    source = _original_find_c_source(base_path)
    if source is not None:
        return source

    try:
        relative = Path(base_path).resolve().relative_to(_source_dir)
    except ValueError:
        return None

    target_pattern = f"{relative.stem}.cpython-*.so.p"
    for target_dir in (_build_dir / relative.parent).glob(target_pattern):
        generated_base = target_dir / relative.parent / relative.name
        for suffix in (".c", ".cpp", ".cc", ".cxx"):
            candidate = Path(f"{generated_base}{suffix}")
            if candidate.exists():
                return str(candidate)
    return None


def _find_dep_file_path(main_file, file_path, relative_path_search=False):
    if main_file is None:
        main_file = str(_source_dir / file_path)
    return _original_find_dep_file_path(
        main_file,
        file_path,
        relative_path_search,
    )


def _configure(self, config):
    _original_configure(self, config)
    for pattern in ("*.pyx.c", "*.pyx.cpp", "*.pyx.cc", "*.pyx.cxx"):
        for generated_source in _build_dir.rglob(pattern):
            self._read_source_lines(str(generated_source), "")


cycoverage._find_c_source = _find_meson_c_source
cycoverage._find_dep_file_path = _find_dep_file_path
cycoverage.Plugin.configure = _configure
