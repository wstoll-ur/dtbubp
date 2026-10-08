#!/usr/bin/env bash
# Run on bluehive3. Read-only node snapshot; archive is outside the Git checkout.
set -euo pipefail
umask 077
repo=$(cd "$(dirname "$0")/.." && pwd)
campaign=02_dft_dataset_r2_pbe-d3bj-c9_dzvpsr
remote=/home/wstoll/dtbubp_runs/Calculations/$campaign/label
destination=${1:-$HOME/dtbubp-transfer}
mkdir -p "$destination"
stage=$(mktemp -d "$destination/r2-transfer.XXXXXXXX")
echo "Snapshot directory: $stage"
mkdir -p "$stage/code" "$stage/Calculations/$campaign" "$stage/Calculations/02_dft_dataset/frame_selection"
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' "$repo/code/" "$stage/code/"
cp "$repo/Calculations/$campaign/config.toml" "$repo/Calculations/$campaign/.config.json" "$stage/Calculations/$campaign/"
cp "$repo/Calculations/02_dft_dataset/frame_selection/frames.extxyz" "$stage/Calculations/02_dft_dataset/frame_selection/"
for node in bhx0125 bhx0124 bhx0123; do
    echo "Pulling inputs and output tables from $node ..."
    mkdir -p "$stage/node_snapshots/$node"
    rsync -a --include '*/' --include 'input.inp' --include 'coord.xyz' \
      --include 'meta.json' --include 'output.out' --include 'chunk_*.txt' --exclude '*' \
      -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' \
      "$node:$remote/" "$stage/node_snapshots/$node/"
done
PYTHONPATH="$stage/code${PYTHONPATH:+:$PYTHONPATH}" python3 "$repo/bluehive/prepare_r2_transfer.py" "$stage"
archive="$stage.tar.gz"
tar -czf "$archive" -C "$destination" "$(basename "$stage")"
(cd "$destination" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
echo "Transfer these two files (keep the unpacked snapshot too):"
printf '%s\n' "$archive" "$archive.sha256"
