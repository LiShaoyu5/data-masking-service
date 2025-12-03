#!/bin/bash

# 检查是否需要 --onefile 选项
ONEFILE_OPTION=""
if [ "$1" = "onefile" ]; then
    ONEFILE_OPTION="--onefile"
fi

python -m nuitka \
    --show-progress \
    --standalone \
    $ONEFILE_OPTION \
    --follow-imports \
    --nofollow-import-to=streamlit \
    --nofollow-import-to=streamlit.runtime \
    --include-module=service_logging \
    --include-package=hanlp \
    --include-package=uvicorn \
    --include-package=pyunit_address \
    --include-distribution-metadata=matplotlib \
    --include-distribution-metadata=hanlp \
    --include-distribution-metadata=uvicorn \
    --include-distribution-metadata=pydantic \
    --include-distribution-metadata=fastapi \
    --include-distribution-metadata=cryptography \
    --include-distribution-metadata=onnx \
    --include-data-dir=/mnt/hdd/cache/conda/envs/service/lib/python3.11/site-packages/dateutil/zoneinfo=dateutil/zoneinfo \
    --output-dir=/mnt/hdd/data/data_masking_build/1029 \
    service.py
