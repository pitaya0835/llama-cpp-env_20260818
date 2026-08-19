#!/usr/bin/env bash
set -euo pipefail

# devcontainer 経由で使う場合、VS Code がデフォルトでこのENTRYPOINTを
# 上書きしてコンテナをアイドル状態で起動する（overrideCommand）。
# そのため以下の自動起動は「docker run」で直接使う場合にのみ実行される。

MODEL_PATH="${MODEL_PATH:-/app/models/model.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
N_CTX="${N_CTX:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
# 画像・動画入力に対応させる場合、同じHFリポジトリにある mmproj-*.gguf の
# パスを指定する（例: unsloth/Qwen3.8-27B-GGUF の mmproj-F16.gguf）。
# 未指定ならテキスト専用として起動する。
MMPROJ_PATH="${MMPROJ_PATH:-}"
NO_MMPROJ_OFFLOAD="${NO_MMPROJ_OFFLOAD:-0}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[ERROR] GGUFモデルが見つかりません: ${MODEL_PATH}" >&2
    echo "        docker run 時に -v <hostのモデルディレクトリ>:/app/models を指定するか、" >&2
    echo "        MODEL_PATH 環境変数でファイル名を合わせてください。" >&2
    exit 1
fi

args=(
    --model "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --ctx-size "${N_CTX}"
    --n-gpu-layers "${N_GPU_LAYERS}"
    --jinja
)

if [ -n "${MMPROJ_PATH}" ]; then
    if [ ! -f "${MMPROJ_PATH}" ]; then
        echo "[ERROR] mmprojファイルが見つかりません: ${MMPROJ_PATH}" >&2
        exit 1
    fi
    args+=(--mmproj "${MMPROJ_PATH}")
    if [ "${NO_MMPROJ_OFFLOAD}" = "1" ]; then
        args+=(--no-mmproj-offload)
    fi
fi

exec /app/llama-server "${args[@]}" "$@"
