#!/usr/bin/env bash
# Provision the LMTrade GPU box on Vast.ai and bring the bot up.
# Prereqs: VAST_API_KEY set, `vastai` CLI installed (pip install vastai).
# This is a reference recipe — review before running against a paid account.
set -euo pipefail

GPU="${GPU:-RTX_3090}"
DISK="${DISK:-30}"
IMAGE="${IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
SLM_MODEL="${LMTRADE_SLM_MODEL:-qwen2.5:1.5b}"

echo "==> Finding cheapest verified ${GPU} offer..."
OFFER=$(vastai search offers "reliability>0.98 verified=true rentable=true gpu_name=${GPU}" \
        --raw --order dph_total | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["id"])')
echo "    picked offer ${OFFER}"

echo "==> Creating instance..."
vastai create instance "${OFFER}" --image "${IMAGE}" --disk "${DISK}" \
  --onstart-cmd "bash -c '
    set -e
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve & sleep 5
    ollama pull ${SLM_MODEL}
    apt-get update && apt-get install -y git
    git clone https://github.com/269652/LMTrade.git /opt/lmtrade || true
    cd /opt/lmtrade && pip install -e .
    # Provide .env out-of-band (do NOT bake secrets into the image).
    nohup lmtrade run > /var/log/lmtrade-run.log 2>&1 &
    nohup lmtrade web --host 0.0.0.0 --port 8000 > /var/log/lmtrade-web.log 2>&1 &
  '"

echo "==> Instance requested. Track it with: vastai show instances"
echo "    Dashboard will be on the instance's mapped port 8000."
echo "    Remember: copy your .env (API keys) to /opt/lmtrade/.env on the box."
