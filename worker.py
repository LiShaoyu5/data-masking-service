#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ Worker 启动脚本
用于处理表格脱敏任务队列

运行方式：
    python worker.py

或者使用多个worker（推荐）：
    python worker.py &
    python worker.py &
    python worker.py &
"""

import sys
import signal
from pathlib import Path

from redis import Redis
from rq import Worker, Queue
from loguru import logger
import yaml
import socket
import os

# 配置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.remove()  # 移除默认处理器
logger.add(
    str(log_dir / "worker.log"),
    rotation="500 MB",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {process} | {message}",
    level="INFO",
    enqueue=True,
    catch=True,
)

logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="INFO",
)

logger.info("Worker logger initialized successfully")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def signal_handler(signum, frame):
    """处理终止信号"""
    logger.warning(f"Received signal {signum}, shutting down worker gracefully...")
    sys.exit(0)


def main():
    """主函数：启动 RQ Worker"""

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 连接到 Redis
    redis_host = config["server"]["redis_host"]
    redis_port = config["server"]["redis_port"]
    redis_db = config["server"]["redis_db"]

    logger.info(f"Connecting to Redis at {redis_host}:{redis_port} (db={redis_db})...")

    try:
        redis_conn = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # 测试连接
        redis_conn.ping()
        logger.success(f"Successfully connected to Redis at {redis_host}:{redis_port}")

    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error("Please ensure Redis is running and accessible")
        sys.exit(1)

    # 创建队列
    queue_name = "table_masking"
    queue = Queue(queue_name, connection=redis_conn)

    logger.info(f"Worker will process tasks from queue: '{queue_name}'")
    logger.info(f"Current queue length: {len(queue)}")

    # 配置 Worker
    worker_name = f"worker-{socket.gethostname()}-{os.getpid()}"

    try:
        worker = Worker(
            [queue],
            connection=redis_conn,
            name=worker_name,
        )

        logger.success(f"Worker '{worker_name}' started successfully")
        logger.info("Worker is ready to process tasks. Press Ctrl+C to stop.")
        logger.info("=" * 60)

        # 开始工作
        worker.work(
            with_scheduler=True,  # 启用调度器支持
            logging_level="INFO",
        )

    except KeyboardInterrupt:
        logger.warning("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker encountered an error: {e}")
        raise
    finally:
        logger.info("Worker shutdown complete")


if __name__ == "__main__":
    # try:
    #     # 尝试导入以验证模块可用
    #     from service import process_table_masking_task

    #     logger.info(
    #         "Successfully imported task processing function from service module"
    #     )
    # except ImportError as e:
    #     logger.error(f"Failed to import service module: {e}")
    #     logger.error("Please ensure worker.py is in the same directory as service.py")
    #     sys.exit(1)

    main()
