# llama-cpp-env

`unsloth/Qwen3.8-27B-GGUF` のような GGUF モデルを、完全オフラインPC上で
Docker + VS Code devcontainer からロードして使うための環境です。

GitHub Actions（GPUなし）で CUDA 対応の `llama.cpp`（`llama-server`）をビルドし、
出力イメージを USB 等でオフラインPCに移して `docker load` で読み込み、
devcontainer からモデルをマウントして起動します。

以前作成した [`Build_Env_of_loading_GGUF`](https://github.com/pitaya0835/Build_Env_of_loading_GGUF)
との主な違いは以下のとおりです。

| | Build_Env_of_loading_GGUF | このリポジトリ |
|---|---|---|
| 実行エンジン | `llama-cpp-python[server]` (pip) | `llama.cpp` 本家を直接ビルド (`llama-server`) |
| ベースイメージ | `pytorch/pytorch`（PyTorch同梱で肥大化） | `nvidia/cuda` の multi-stage build（ビルドツール類を最終イメージに残さない） |
| 新モデル対応 | pip版が同梱する vendored llama.cpp のバージョンに依存し、最新アーキ（Qwen3系など）への追従が遅れがち | 本家 master を直接ビルドするため最新モデルへの対応が早い |
| 転送用イメージ | 非圧縮 tar | gzip 圧縮 tar（USB転送・保存容量に有利） |
| CUDA / GPUアーキテクチャ | CUDA 12.1 / arch 89 固定 | CUDA 12.8 / arch 120（RTX 5070 Ti・5060 TiのBlackwellにネイティブ対応）。ビルド引数 (`CUDA_ARCH`) / workflow_dispatch 入力で切り替え可能 |
| UI/API | 自作 `chat.py`（対話CLIのみ） | `llama-server` 標準搭載の OpenAI互換API + Web UI |

## 構成

```
Dockerfile                     # multi-stage build（build stage: nvidia/cuda devel, runtime stage: nvidia/cuda runtime）
docker/entrypoint.sh           # コンテナ起動時に llama-server を起動するスクリプト
.devcontainer/devcontainer.json
.github/workflows/build.yml    # GitHub ActionsでビルドしArtifactとしてtar.gzを出力
```

## 1. GitHub Actions でビルド

`main` ブランチに push するか、Actions タブから `workflow_dispatch` で手動実行してください。
手動実行時は以下を上書きできます。

- `cuda_arch`: 対象GPUの compute capability（既定値 `120` = RTX 5070 Ti / RTX 5060 Ti のBlackwell世代にネイティブ対応）。
  複数GPU環境を想定する場合は `89;120` のように `;` 区切りで複数指定可能（ビルド時間・イメージサイズは伸びます）。
- `llama_cpp_ref`: ビルドする llama.cpp のブランチ・タグ・コミットハッシュ（既定 `master`）。

ビルドが終わると Artifact `qwen-gguf-llamacpp-image` に `qwen-gguf-llamacpp.tar.gz` が生成されるので、
ダウンロードして USB 等でオフラインPCに移してください。

> **注記（GPUアーキテクチャについて）**: ベースイメージを `nvidia/cuda:12.8.1-devel/runtime-ubuntu22.04` へ、
> `CUDA_ARCH` を `120`（Blackwell世代のネイティブ compute capability）へ切り替え済みです。
> 実行側（オフラインPC）には NVIDIA Driver 570.x 以降が必要ですが、確認済みの 610.88 なら問題ありません。
>
> トレードオフとして、`120` 単独でビルドしたバイナリは Ada世代以前（例: RTX 40系, compute capability 89）の
> GPU では動作しません。将来的にBlackwell以外のGPUを組み合わせる予定がある場合は、
> `CUDA_ARCH=89;120` のように複数アーキ指定でビルドしてください（ビルド時間とイメージサイズが増えます）。
> 今回は「その時が来たら再ビルドすれば良い」との判断で `120` 単独にしています。

## 2. オフラインPCでの読み込み

`docker load` はgzip圧縮されたtarをそのまま自動認識するため、`gunzip`等での手動展開は不要です
（Windowsには標準で`gunzip`コマンドが無いため、`docker load -i`を直接使ってください）。

```powershell
docker load -i qwen-gguf-llamacpp.tar.gz
```

Linux/macOSでパイプで読み込みたい場合は次の書き方でも同様に動作します。

```bash
gunzip -c qwen-gguf-llamacpp.tar.gz | docker load
```

## 3. GGUFモデルの用意

`unsloth/Qwen3.8-27B-GGUF` はネイティブVision(画像・動画)対応モデルです。画像を扱うには、
言語モデル本体の量子化GGUFに加えて、**同じHFリポジトリ内にある `mmproj-*.gguf`（マルチモーダル
プロジェクタ、ビジョンエンコーダ）を別途ダウンロードする必要があります**。ビジョンエンコーダは
言語モデルGGUFの中には含まれていません。

オンライン環境で以下の2ファイルをダウンロードし、USBでオフラインPCの同じディレクトリ
（例 `C:\llm_models`）に置いてください。

- 言語モデル本体: 使用したい量子化レベルの `*.gguf`（例: `Qwen3.8-27B-Q4_K_M.gguf`）
- マルチモーダルプロジェクタ: `mmproj-F16.gguf`（通常1GB弱。量子化版が複数ある場合はF16を推奨）

- ファイル名は自由ですが、`devcontainer.json` の `MODEL_PATH`/`MMPROJ_PATH` をそのファイル名に合わせて書き換えるか、
  それぞれ `model.gguf` にリネームしてください（既定値は `/app/models/model.gguf`）。
- **VRAM見積もりの確認を推奨**: 27B級モデルは量子化レベルによってはロード時点で十数GB超のVRAMを消費し、
  `mmproj`もGPUにオフロードされるとさらに数百MB〜1GB程度上乗せされます。
  搭載VRAM（例: RTX 5070 Ti / 5060 Tiは16GB）に対してコンテキスト長分の余裕も必要になるため、
  収まらない場合は `N_CTX` を下げる、より低ビットの量子化（IQ4系など）を使う、
  または `NO_MMPROJ_OFFLOAD=1` で mmproj をCPU側に置く、のいずれかを検討してください。

## 4. devcontainer で起動

1. `.devcontainer/devcontainer.json` の `mounts` の `source` を、オフラインPCの実際のモデル格納先パスに書き換えてください。
2. VS Code で「Reopen in Container」。
3. コンテナ内ターミナルから手動でサーバーを起動します（devcontainerはENTRYPOINTの自動起動を上書きしているため）。

   ```bash
   /app/entrypoint.sh
   ```

4. ブラウザで `http://localhost:8080` を開くと llama-server 標準のチャットUIが使えます。
   OpenAI互換APIとしても `http://localhost:8080/v1/chat/completions` 等で利用できます。

`docker run` で直接使う場合はENTRYPOINTがそのまま自動起動するので、例えば以下のように実行できます
（`MMPROJ_PATH` を指定すると画像・動画入力が有効になります。未指定ならテキスト専用起動です）。

```bash
docker run --rm --gpus all \
  -v C:\llm_models:/app/models \
  -e MODEL_PATH=/app/models/Qwen3.8-27B-Q4_K_M.gguf \
  -e MMPROJ_PATH=/app/models/mmproj-F16.gguf \
  -p 8080:8080 \
  qwen-gguf-llamacpp:latest
```

## 5. 画像入力の使い方

`MMPROJ_PATH` を設定して起動すると、`llama-server` の Web UI (`http://localhost:8080`) から
画像をドラッグ＆ドロップしてそのままチャットできます。

OpenAI互換APIから使う場合は、`image_url` コンテンツパートに Base64 データURIを渡します。

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "この画像には何が写っていますか？"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<Base64文字列>"}}
        ]
      }
    ]
  }'
```

コンテナ内でAPIを介さず直接テストしたい場合は、同梱の `llama-mtmd-cli` も使えます。

```bash
/app/llama-mtmd-cli \
  --model /app/models/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /app/models/mmproj-F16.gguf \
  --image /app/models/sample.jpg \
  -p "この画像を説明してください"
```

Difyから画像付きで使う場合は、モデルプロバイダー設定の「Vision support」を有効にしてください。
有効にしないとDify側が画像添付欄自体を表示しません。

## トラブルシューティング

- `nvidia-smi` がコンテナ内で失敗する: オフラインPC側に NVIDIA Driver と `nvidia-container-toolkit`
  がインストールされ、Docker Desktop / Docker Engine の GPU サポートが有効になっているか確認してください。
- `GGUFモデルが見つかりません` と出る: `mounts` の `source` パス、`MODEL_PATH` のファイル名が一致しているか確認してください。
- VRAM不足で起動に失敗する: `N_GPU_LAYERS` を `-1`（全レイヤーGPU）から具体的な数値に下げ、
  一部レイヤーをCPUにオフロードしてください。
