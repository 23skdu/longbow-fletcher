#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"
export PATH="/home/linuxbrew/.linuxbrew/bin:$(go env GOPATH 2>/dev/null || echo "$HOME/go")/bin:$PATH"
if ! PKGS=$(go list -f '{{.ImportPath}} {{.Dir}}' ./... 2>/dev/null); then
  echo "ERROR: go list ./... failed"
  exit 2
fi
UNFUZZED=""
TOTAL=0
while read -r pkg dir; do
  [ -z "$pkg" ] && continue
  TOTAL=$((TOTAL + 1))
  if ! grep -qs '^func Fuzz' "$dir"/*_test.go; then
    UNFUZZED="$UNFUZZED $pkg"
  fi
done <<< "$PKGS"
if [ -n "$UNFUZZED" ]; then
  echo "FAIL: packages without fuzz targets:"
  echo "$UNFUZZED" | tr ' ' '\n' | sed '/^$/d'
  exit 1
fi
echo "PASS: 100% package fuzz coverage ($TOTAL packages)"
