import argparse
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


SOURCE_SUFFIXES = {".py", ".pyx", ".pxd", ".pxi"}
HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def parse_diff(diff: str) -> dict[str, set[int]]:
    changed_lines: dict[str, set[int]] = defaultdict(set)
    current_file = None

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            current_file = path[2:] if path.startswith("b/") else None
            continue

        if current_file is None:
            continue

        match = HUNK_PATTERN.match(line)
        if match is None:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or 1)
        changed_lines[current_file].update(range(start, start + count))

    return changed_lines


def merge_changed_lines(
    destination: dict[str, set[int]],
    source: dict[str, set[int]],
) -> None:
    for filename, lines in source.items():
        destination[filename].update(lines)


def get_changed_lines(base_ref: str) -> dict[str, set[int]]:
    merge_base = run_git("merge-base", "HEAD", base_ref).strip()
    changed_lines: dict[str, set[int]] = defaultdict(set)

    for args in (
        ("diff", "--unified=0", "--no-color", f"{merge_base}...HEAD", "--", "pandas"),
        ("diff", "--unified=0", "--no-color", "--", "pandas"),
        ("diff", "--cached", "--unified=0", "--no-color", "--", "pandas"),
    ):
        merge_changed_lines(changed_lines, parse_diff(run_git(*args)))

    untracked = run_git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "pandas",
    ).splitlines()
    for filename in untracked:
        path = Path(filename)
        if path.suffix in SOURCE_SUFFIXES:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            changed_lines[filename].update(range(1, line_count + 1))

    return {
        filename: lines
        for filename, lines in changed_lines.items()
        if filename.startswith("pandas/")
        and Path(filename).suffix in SOURCE_SUFFIXES
        and not filename.startswith("pandas/tests/")
    }


def read_coverage(
    coverage_file: Path,
) -> dict[str, dict[int, int]]:
    coverage: dict[str, dict[int, int]] = {}
    root = ET.parse(coverage_file).getroot()

    for class_node in root.findall(".//class"):
        filename = class_node.attrib["filename"].replace("\\", "/")
        if not filename.startswith("pandas/"):
            filename = f"pandas/{filename}"

        coverage[filename] = {
            int(line.attrib["number"]): int(line.attrib["hits"])
            for line in class_node.findall("./lines/line")
        }

    return coverage


def format_lines(lines: list[int]) -> str:
    ranges: list[str] = []
    start = previous = lines[0]

    for line in lines[1:]:
        if line == previous + 1:
            previous = line
            continue

        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = line

    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_file", type=Path)
    parser.add_argument("base_ref")
    parser.add_argument("--fail-under", type=float, default=80)
    args = parser.parse_args()

    changed_lines = get_changed_lines(args.base_ref)
    if not changed_lines:
        print(
            "Incremental coverage: no measurable source changes "
            f"against {args.base_ref}."
        )
        return 0

    coverage = read_coverage(args.coverage_file)
    measurable = 0
    covered = 0
    missing_by_file: dict[str, list[int]] = {}
    missing_reports: list[str] = []

    print(f"Incremental coverage files against {args.base_ref}:")
    for filename in sorted(changed_lines):
        print(filename)
        file_coverage = coverage.get(filename)
        if file_coverage is None:
            if Path(filename).suffix in {".py", ".pyx"}:
                missing_reports.append(filename)
            continue

        executable = changed_lines[filename] & file_coverage.keys()
        measurable += len(executable)
        covered += sum(file_coverage[line] > 0 for line in executable)
        missing = sorted(line for line in executable if file_coverage[line] == 0)
        if missing:
            missing_by_file[filename] = missing

    if missing_reports:
        print("Missing coverage data for changed source files:")
        for filename in missing_reports:
            print(f"  {filename}")
        return 1

    if measurable == 0:
        print("Incremental coverage: no changed executable lines.")
        return 0

    percent = covered * 100 / measurable
    print(f"Incremental lines: {measurable}")
    print(f"Covered lines: {covered}")
    print(f"Missing lines: {measurable - covered}")
    print(f"Incremental coverage: {percent:.2f}%")

    for filename, lines in missing_by_file.items():
        print(f"  {filename}: {format_lines(lines)}")

    if percent < args.fail_under:
        print(
            f"Incremental coverage is below {args.fail_under:.2f}%.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
