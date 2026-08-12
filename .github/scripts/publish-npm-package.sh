#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 <package-directory>" >&2
  exit 2
fi

package_directory="$1"
package_name="$(node -p "require('./${package_directory}/package.json').name")"
package_version="$(node -p "require('./${package_directory}/package.json').version")"

if registry_output="$(npm view "${package_name}@${package_version}" version --json 2>&1)"; then
  echo "${package_name}@${package_version} is already available on npm."
  exit 0
fi

if [[ "$registry_output" != *"E404"* ]]; then
  echo "$registry_output" >&2
  exit 1
fi

npm publish "${package_directory}" --ignore-scripts --provenance
