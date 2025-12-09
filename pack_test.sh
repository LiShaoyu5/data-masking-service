#!/bin/bash

SECONDS=0
# 从命令行参数读输出文件夹名词
BASE_DIR="/mnt/hdd/data/data_masking_build/"
OUTPUT_DIR="${BASE_DIR}${1}"
echo "Output directory: $OUTPUT_DIR"

python -m nuitka \
    --show-progress \
    --standalone \
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
    --include-distribution-metadata=redis \
    --include-distribution-metadata=rq \
    --include-distribution-metadata=loguru \
    --include-distribution-metadata=pyyaml \
    --nofollow-import-to=triton \
    --noinclude-data-files="nvidia/nccl/*" \
    --noinclude-data-files="nvidia/nvshmem/*" \
    --noinclude-data-files="nvidia/cusparselt/*" \
    --noinclude-data-files="nvidia/cufft/*" \
    --noinclude-data-files="nvidia/cuda_cupti/*" \
    --noinclude-data-files="nvidia/cufile/*" \
    --noinclude-data-files="nvidia/cusparse/*" \
    --noinclude-data-files="nvidia/cusolver/*" \
    --include-data-dir=/mnt/hdd/cache/conda/envs/service/lib/python3.11/site-packages/dateutil/zoneinfo=dateutil/zoneinfo \
    --output-dir=$OUTPUT_DIR \
    main.py

echo "Time taken: $SECONDS seconds"
