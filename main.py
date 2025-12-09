#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一启动入口
"""

import sys
from datetime import datetime, date
from loguru import logger
import yaml


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if len(sys.argv) < 2:
        print("Usage: python main.py [service|worker]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "service":
        # 启动 FastAPI 服务
        from service import app
        import uvicorn
        from typing import Any

        current_date = datetime.now().date()
        trial_end_date = date(2026, 1, 31)
        if current_date > trial_end_date:
            logger.warning("试用期已过，退出程序。")
            sys.exit(1)

        # 解决uvicorn创建子进程超时无限循环的问题
        original_uvicorn_is_alive = uvicorn.supervisors.multiprocess.Process.is_alive

        # def patched_is_alive_legacy(self: Any) -> bool:
        #     timeout=120
        #     return original_uvicorn_is_alive(self, timeout)

        def patched_is_alive(self: Any, timeout=120) -> bool:
            return original_uvicorn_is_alive(self, timeout)

        uvicorn.supervisors.multiprocess.Process.is_alive = patched_is_alive

        server_config = config["server"]
        logger.info(
            f"Starting server with config: host={server_config['host']}, port={server_config['port']}, workers={server_config['workers']}"
        )

        # 使用当前文件的app实例
        uvicorn.run(
            "service:app",
            host=server_config["host"],
            port=server_config["port"],
            workers=server_config["workers"],
        )

    elif mode == "worker":
        # 启动 Worker
        import worker

        worker.main()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: service, worker")
        sys.exit(1)


if __name__ == "__main__":
    main()
