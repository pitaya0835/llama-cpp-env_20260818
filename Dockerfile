# syntax=docker/dockerfile:1
#
# llama.cpp (llama-server) を CUDA 対応でビルドし、GGUF モデルをロードするための
# オフライン実行用コンテナ。GitHub Actions (GPUなし) でビルドし、
# 出力イメージを USB 等でオフラインPCに移して docker load で使う想定。
#
# ===== Build stage =====
# Blackwell世代(RTX 50xx, compute capability 120)をネイティブサポートするには
# CUDA Toolkit 12.8 以降が必要（12.8未満はsm_120を認識しない）。
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# GitHub Actions ランナーには物理GPUが無く、CMakeのCUDAドライバ検出が
# libcuda.so.1 not found で失敗するため、CUDA Toolkit付属のスタブを
# ドライバの代わりとして配置する（NVIDIA公式にも案内されている回避策）。
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 || true

WORKDIR /build

# 特定バージョンに固定したい場合は --build-arg LLAMA_CPP_REF=b<番号> や
# タグ名を指定する（例: b4700）。デフォルトは最新の master。
ARG LLAMA_CPP_REF=master
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout ${LLAMA_CPP_REF}

# RTX 5070 Ti / RTX 5060 Ti (どちらもBlackwell世代: compute capability 120) 向けにビルド。
# 別GPUに変える場合は --build-arg CUDA_ARCH=89;120 のように ; 区切りで複数指定可能。
ARG CUDA_ARCH=120
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-cli llama-mtmd-cli llama-quantize

# ===== Runtime stage =====
# devel(ビルドツール一式)を含まないランタイム専用イメージにすることで、
# USB転送するイメージサイズを大幅に削減する。
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# llama.cpp のバイナリと共有ライブラリ(libggml*.so, libllama.so 等)は
# すべて build/bin にまとめて出力される（RPATHが $ORIGIN 設定のため）。
COPY --from=build /build/build/bin/ /app/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV LD_LIBRARY_PATH=/app
ENV PATH=/app:${PATH}

# モデルはイメージに焼き込まず、実行時にホストからマウントする。
VOLUME ["/app/models"]
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
