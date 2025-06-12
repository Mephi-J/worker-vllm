FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

# 必要パッケージをインストール
RUN apt-get update -y \
    && apt-get install -y python3-pip

# CUDA の互換ライブラリを登録
RUN ldconfig /usr/local/cuda-12.1/compat/

# Python依存パッケージをインストール
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --break-system-packages --upgrade -r /requirements.txt

# vLLM と FlashInfer をインストール
RUN python3 -m pip install --break-system-packages vllm==0.9.0.1 && \
    python3 -m pip install --break-system-packages flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# モデル読み込み関連の ARG と ENV 設定
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONPATH="/:/vllm-workspace"

# モデルダウンロード用スクリプトと src をコピー
COPY src /src

# HuggingFace トークンが存在すればモデルをダウンロード
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# サーバレス実行時の起動コマンド
CMD ["python3", "/src/handler.py"]
