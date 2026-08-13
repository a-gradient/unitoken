#!/usr/bin/env python3

import json
import os
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def toml_version(path: str, table: str) -> str:
  with (ROOT / path).open("rb") as file:
    document = tomllib.load(file)
  return str(document[table]["version"])


def package(path: str) -> dict[str, object]:
  with (ROOT / path).open(encoding="utf-8") as file:
    return json.load(file)


def main() -> int:
  with (ROOT / "Cargo.toml").open("rb") as file:
    cargo_manifest = tomllib.load(file)

  versions = {
    "Cargo.toml": toml_version("Cargo.toml", "package"),
    "pyproject.toml": toml_version("pyproject.toml", "project"),
    "crates/ffbpe-wasm/Cargo.toml": toml_version(
      "crates/ffbpe-wasm/Cargo.toml",
      "package",
    ),
    "packages/ffbpe/package.json": str(
      package("packages/ffbpe/package.json")["version"],
    ),
  }
  release_version = versions["Cargo.toml"]

  mismatches = {
    path: version
    for path, version in versions.items()
    if version != release_version
  }
  if mismatches:
    print(
      f"release versions do not match {release_version}: {mismatches}",
      file=sys.stderr,
    )
    return 1

  pat_version = toml_version("crates/ffbpe-pat/Cargo.toml", "package")
  pat_dependency = cargo_manifest["dependencies"]["ffbpe-pat"]["version"]
  if pat_dependency != pat_version:
    print(
      "Cargo.toml requires ffbpe-pat "
      f"{pat_dependency!r}; expected {pat_version!r}",
      file=sys.stderr,
    )
    return 1

  expected_peer = f">={release_version} <0.2.0"
  for path in (
    "packages/ffbpe-inspect/package.json",
    "packages/ffbpe-presets/package.json",
  ):
    peer_version = package(path)["peerDependencies"]["@tokn-ai/ffbpe"]
    if peer_version != expected_peer:
      print(
        f"{path} requires {peer_version!r}; expected {expected_peer!r}",
        file=sys.stderr,
      )
      return 1

  if os.environ.get("GITHUB_REF_TYPE") == "tag":
    tag = os.environ.get("GITHUB_REF_NAME")
    expected_tag = f"v{release_version}"
    if tag != expected_tag:
      print(f"release tag is {tag!r}; expected {expected_tag!r}", file=sys.stderr)
      return 1

  print(f"release metadata is consistent for {release_version}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
