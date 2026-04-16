#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
REF_DIR="$REPO_DIR/../references"

echo "=== baby-lattice-folding setup ==="
echo ""

# 1. Check Rust toolchain
echo "--- Rust toolchain ---"
if command -v rustc &>/dev/null; then
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
else
  echo "Rust not found. Install via:"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  echo ""
  echo "After installing, restart your shell and re-run this script."
  exit 1
fi

# 2. Clone reference repos
echo ""
echo "--- Reference repos ---"

REFS="
latticefold:https://github.com/NethermindEth/latticefold.git
salsaa:https://github.com/lattice-arguments/salsaa.git
algebra:https://github.com/arkworks-rs/algebra.git
kyber:https://github.com/Argyle-Software/kyber.git
"

mkdir -p "$REF_DIR"

for entry in $REFS; do
  name="${entry%%:*}"
  url="${entry#*:}"
  dest="$REF_DIR/$name"

  if [[ -d "$dest" ]]; then
    echo "skip: $name (already cloned)"
  else
    echo "cloning: $name ..."
    git clone --depth 1 "$url" "$dest"
  fi
done

# 3. Download papers
echo ""
echo "--- Papers ---"
bash "$SCRIPT_DIR/download-papers.sh"

# 4. Verify workspace (if Cargo.toml exists)
echo ""
echo "--- Workspace ---"
if [[ -f "$REPO_DIR/Cargo.toml" ]]; then
  echo "running: cargo check"
  cd "$REPO_DIR" && cargo check
else
  echo "no Cargo.toml yet (skip cargo check)"
fi

echo ""
echo "=== setup complete ==="
