#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPERS_DIR="$SCRIPT_DIR/../papers"
mkdir -p "$PAPERS_DIR"

UA="User-Agent: Mozilla/5.0"

# name:eprint_id pairs
PAPERS="
hypernova:2023/573
latticefold:2024/257
latticefold-plus:2025/247
neo:2025/294
superneo:2026/242
salsaa:2025/2124
cyclo:2026/359
"

for entry in $PAPERS; do
  name="${entry%%:*}"
  eprint_id="${entry#*:}"
  dest="$PAPERS_DIR/$name.pdf"

  if [[ -f "$dest" ]]; then
    echo "skip: $name.pdf (already exists)"
    continue
  fi

  url="https://eprint.iacr.org/${eprint_id}.pdf"
  echo "downloading: $name.pdf ..."
  curl -sL -H "$UA" -o "$dest" "$url"

  if file "$dest" | grep -q "PDF"; then
    echo "  ok"
  else
    echo "  FAILED (not a valid PDF, removing)"
    rm -f "$dest"
  fi
done

echo ""
echo "papers in $PAPERS_DIR:"
ls -lh "$PAPERS_DIR"/*.pdf 2>/dev/null || echo "  (none)"
