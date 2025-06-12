FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 基本セットアップ
RUN apt-get update -y && apt-get install -y python3-pip git && \
    ldconfig /usr/local/cuda-12.1/compat/

# requirements.txt をコピー
COPY builder/requirements.txt /requirements.txt

# pipのバージョン更新
RUN python3 -m pip install --upgrade pip

# CUDAに依存しないパッケージを先にインストール
RUN python3 -m pip install \
    ray \
    pandas \
    pyarrow \
    runpod~=1.7.7 \
    huggingface-hub \
    packaging \
    typing-extensions>=4.8.0 \
    pydantic \
    pydantic-settings \
    hf-transfer \
    optimum>=1.12.0

# CUDA依存のあるパッケージを順にインストール
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install transformers==4.52.3
RUN python3 -m pip install bitsandbytes>=0.45.0
RUN python3 -m pip install autoawq

# vLLM・flashinfer
RUN python3 -m pip install vllm==0.9.0.1
RUN python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

# モデル設定用の環境変数
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

# モデル読み込みスクリプトを配置
COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# ハンドラーを起動
CMD ["python3", "/src/handler.py"]
