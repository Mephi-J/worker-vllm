FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 環境のアップデートと基本ツールのインストール
RUN apt-get update -y \
    && apt-get install -y python3-pip

# CUDA互換ライブラリのリンク設定
RUN ldconfig /usr/local/cuda-12.1/compat/

# Python依存パッケージのインストール
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# vLLM と FlashInfer のインストール
RUN python3 -m pip install vllm==0.9.0.1
#    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# モデル情報・キャッシュ関連の設定
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG HF_TOKEN=""

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
    HF_TOKEN=${HF_TOKEN}

ENV PYTHONPATH="/:/vllm-workspace"

# モデルダウンロードスクリプトの配置と実行
COPY src /src
RUN if [ -n "$MODEL_NAME" ]; then \
        python3 /src/download_model.py; \
    fi

# ハンドラーを実行
CMD ["python3", "/src/handler.py"]
