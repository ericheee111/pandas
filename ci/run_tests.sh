#!/bin/bash -e

set -e

SOURCE_DIR=$(cd "$(dirname "$0")/.." && pwd)

if [[ -z "${PYTEST_WORKERS+x}" ]]; then
  PYTEST_WORKERS=8
fi
if [[ -z "${PYTEST_TARGET+x}" ]]; then
  PYTEST_TARGET=pandas
fi
if [[ -z "${PATTERN+x}" ]]; then
  PATTERN="not network and not clipboard and not db"
fi
if [[ -z "${TEST_ARGS+x}" ]]; then
  TEST_ARGS=$(
    printf "%s " \
      "--ignore=pandas/tests/plotting" \
      "--deselect=pandas/tests/arrays/sparse/test_array.py::TestSparseArrayAnalytics::test_ufunc" \
      "--deselect='pandas/tests/apply/test_str.py::test_apply_np_transformer[transform-log]'" \
      "--deselect='pandas/tests/apply/test_str.py::test_apply_np_transformer[apply-log]'" \
      "--deselect=pandas/tests/io/test_sql.py::test_con_string_import_error" \
      "--deselect=pandas/tests/series/methods/test_argsort.py::TestSeriesArgsort::test_argsort_stable" \
      "--deselect='pandas/tests/extension/test_arrow.py::TestArrowArray::test_from_sequence_of_strings_pa_array[timestamp[s, tz=UTC]]'" \
      "--deselect='pandas/tests/extension/test_arrow.py::TestArrowArray::test_from_sequence_of_strings_pa_array[timestamp[ms, tz=UTC]]'" \
      "--deselect='pandas/tests/extension/test_arrow.py::TestArrowArray::test_from_sequence_of_strings_pa_array[timestamp[us, tz=UTC]]'" \
      "--deselect='pandas/tests/extension/test_arrow.py::TestArrowArray::test_from_sequence_of_strings_pa_array[timestamp[ns, tz=UTC]]'" \
      "--deselect=pandas/tests/extension/test_arrow.py::test_dt_strftime" \
      "--deselect='pandas/tests/resample/test_datetime_index.py::test_arrow_timestamp_resample[UTC]'" \
      "--deselect='pandas/tests/tools/test_to_datetime.py::TestToDatetime::test_to_datetime_arrow[index-None-True]'" \
      "--deselect='pandas/tests/tools/test_to_datetime.py::TestToDatetime::test_to_datetime_arrow[series-None-True]'" \
      "-k 'not ((TestTimestampReplace and test_replace_tzinfo and not test_replace_tzinfo_equiv_tz_localize_none) or (TestTimestampMethod and test_timestamp))'"
  )
fi

# Workaround for pytest-xdist (it collects different tests in the workers if PYTHONHASHSEED is not set)
# https://github.com/pytest-dev/pytest/issues/920
# https://github.com/pytest-dev/pytest/issues/1075
PYTHONHASHSEED=$(python -c 'import random; print(random.randint(1, 4294967295))')
export PYTHONHASHSEED
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

if [[ -z "${PANDAS_CI+x}" ]]; then
  PANDAS_CI=1
fi
export PANDAS_CI
echo "PANDAS_CI=$PANDAS_CI"

python -m pip install -e "$SOURCE_DIR" \
  --no-build-isolation \
  -Csetup-args=-Dpandas_cython_coverage=true

PYTHON_TAG=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
PANDAS_MESON_BUILD_DIR="$SOURCE_DIR/build/$PYTHON_TAG"
if [[ ! -d "$PANDAS_MESON_BUILD_DIR" ]]; then
  echo "Could not find the Meson build directory under $SOURCE_DIR/build" >&2
  exit 1
fi

export COVERAGE_CORE=ctrace
export PANDAS_MESON_BUILD_DIR
export PANDAS_SOURCE_DIR="$SOURCE_DIR"
export PYTHONPATH="$SOURCE_DIR/ci/cython_coverage_compat${PYTHONPATH:+:$PYTHONPATH}"

python -m coverage erase

COVERAGE="-s --cov=pandas --cov-append --cov-report=term --cov-report=xml --cov-config=pyproject.toml"

PYTEST_CMD="MESONPY_EDITABLE_VERBOSE=1 PYTHONDEVMODE=1 PYTHONWARNDEFAULTENCODING=1 python -m pytest -r fE -n $PYTEST_WORKERS --dist=worksteal $TEST_ARGS $COVERAGE $PYTEST_TARGET"

if [[ "$PATTERN" ]]; then
  PYTEST_CMD="$PYTEST_CMD -m \"$PATTERN\""
fi

echo "$PYTEST_CMD"
sh -c "$PYTEST_CMD"

echo "Python coverage:"
python -m coverage report --include="*/pandas/**/*.py" | tail -n 1

echo "Cython coverage:"
python -m coverage report \
  --include="*/pandas/**/*.pyx,*/pandas/**/*.pxi,*/pandas/**/*.pxd" \
  | tail -n 1

coverage_gate_failed=0
echo "Full coverage (minimum ${TOTAL_COVERAGE_MIN:-70}%):"
if python -m coverage report --format=total --fail-under="${TOTAL_COVERAGE_MIN:-70}"; then
  echo "Full coverage gate: passed"
else
  echo "Full coverage gate: failed"
  coverage_gate_failed=1
fi

if [[ -z "${DIFF_COVERAGE_MIN+x}" ]]; then
  DIFF_COVERAGE_MIN=80
fi

if [[ -n "${GIT_BRANCH:-}" ]]; then
  if ! git rev-parse --verify --quiet "$GIT_BRANCH^{commit}" >/dev/null; then
    echo "CI base branch does not exist: $GIT_BRANCH" >&2
    exit 1
  fi

  if ! git rev-parse --verify --quiet "HEAD^2" >/dev/null; then
    echo "CI checkout is not a merge commit; cannot verify incremental coverage base." >&2
    exit 1
  fi

  GIT_BRANCH_COMMIT=$(git rev-parse "$GIT_BRANCH^{commit}")
  MERGE_BASE_COMMIT=$(git rev-parse "HEAD^1")

  if [[ "$GIT_BRANCH_COMMIT" != "$MERGE_BASE_COMMIT" ]]; then
    echo "CI base mismatch:" >&2
    echo "  GIT_BRANCH=$GIT_BRANCH -> $GIT_BRANCH_COMMIT" >&2
    echo "  HEAD^1 -> $MERGE_BASE_COMMIT" >&2
    exit 1
  fi

  COVERAGE_BASE_REF="$MERGE_BASE_COMMIT"
  echo "Incremental coverage base: $COVERAGE_BASE_REF ($GIT_BRANCH)"

  if ! python "$SOURCE_DIR/ci/check_incremental_coverage.py" \
    coverage.xml \
    "$COVERAGE_BASE_REF" \
    --fail-under="$DIFF_COVERAGE_MIN"; then
    coverage_gate_failed=1
  fi
else
  echo "Local environment detected, skipping incremental coverage gate."
  echo "To run incremental coverage in CI, export GIT_BRANCH=<base_ref>."
fi

exit "$coverage_gate_failed"
