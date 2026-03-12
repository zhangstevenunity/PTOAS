#!/usr/bin/env python3

import argparse
import pathlib
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the ptoas CLI version from the top-level CMakeLists.txt."
    )
    parser.add_argument(
        "--cmake-file",
        default="CMakeLists.txt",
        help="Path to the top-level CMakeLists.txt file.",
    )
    parser.add_argument(
        "--mode",
        choices=("dev", "release"),
        default="dev",
        help="release mode increments the minor component by 1 (for example, 0.7 -> 0.8 and 0.10 -> 0.11).",
    )
    parser.add_argument(
        "--check-tag",
        help="Optional release tag to validate, e.g. v0.8 or 0.8.",
    )
    return parser.parse_args()


def read_base_version(cmake_file: pathlib.Path) -> str:
    content = cmake_file.read_text(encoding="utf-8")
    match = re.search(r"project\s*\(\s*ptoas\s+VERSION\s+([0-9]+\.[0-9]+)\s*\)", content)
    if not match:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    return match.group(1)


def bump_version(base_version: str) -> str:
    major_str, minor_str = base_version.split(".")
    major = int(major_str)
    minor = int(minor_str) + 1
    return f"{major}.{minor}"


def normalize_tag(tag: str) -> str:
    return tag[1:] if tag.startswith("v") else tag


def main() -> int:
    args = parse_args()
    cmake_file = pathlib.Path(args.cmake_file)
    base_version = read_base_version(cmake_file)
    version = bump_version(base_version) if args.mode == "release" else base_version

    if args.check_tag is not None:
        normalized_tag = normalize_tag(args.check_tag.strip())
        if normalized_tag != version:
            print(
                f"release tag '{args.check_tag}' does not match computed version '{version}'",
                file=sys.stderr,
            )
            return 1

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
