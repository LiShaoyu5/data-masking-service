#!/bin/bash

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
    --include-module=service \
    --include-module=worker \
    --include-module=_yaml \
    --include-package=hanlp \
    --include-package=uvicorn \
    --include-package=pyunit_address \
    --include-package=redis \
    --include-package=rq \
    --include-package=loguru \
    --include-package=yaml \
    --include-package=rq \
    --include-package=redis \
    --include-distribution-metadata=matplotlib \
    --include-distribution-metadata=hanlp \
    --include-distribution-metadata=uvicorn \
    --include-distribution-metadata=pydantic \
    --include-distribution-metadata=fastapi \
    --include-distribution-metadata=cryptography \
    --include-distribution-metadata=onnx \
    --include-distribution-metadata=redis \
    --include-distribution-metadata=rq \
    --include-distribution-metadata=loguru \
    --include-distribution-metadata=pyyaml \
    --include-data-dir=/mnt/hdd/cache/conda/envs/service/lib/python3.11/site-packages/dateutil/zoneinfo=dateutil/zoneinfo \
    --output-dir=/mnt/hdd/data/data_masking_build/1201 \
    main.py