#!/usr/bin/env bash
# 从 tiny-llm-stack 同步 972 权重快照到本仓库。
# 用法: ./tools/sync-weights.sh [上游路径]
set -euo pipefail
SRC="${1:-../tiny-llm-stack/code/train_minimal/trained_weights.json}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/trained_weights.json"
[ -f "$SRC" ] || { echo "找不到上游权重: $SRC"; exit 1; }
cp "$SRC" "$DEST"
echo "已同步 → $DEST"
if command -v md5 >/dev/null 2>&1; then md5 -q "$DEST"; else md5sum "$DEST"; fi
