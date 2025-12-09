"""
Task processor module for RQ workers
This module contains the task processing functions that will be executed by RQ workers
"""

import io
import json
import time
from pathlib import Path
from loguru import logger
import pandas as pd
import polars as pl

from worker_models import get_table_masking_worker

# Ensure output directory exists
TASK_RESULTS_DIR = Path("static")
TASK_RESULTS_DIR.mkdir(exist_ok=True)


def process_table_masking_task(
    file_content: bytes,
    filename: str,
    columns_dict: dict,
    request_id: str,
    file_extension: str,
):
    """
    Polars 优化版本

    Args:
        file_content: 文件内容（字节）
        filename: 原始文件名
        columns_dict: 列名和处理方式的字典
        request_id: 请求ID
        file_extension: 文件扩展名（.csv 或 .xlsx）

    Returns:
        dict: 包含处理结果的字典
    """
    try:
        start_time = time.time()

        logger.bind(type="TASK", request_id=request_id).info(
            f"Task started - Processing file: {filename} with columns: {columns_dict}"
        )

        # Get the pre-initialized table masking worker
        table_masking_worker = get_table_masking_worker()
        anonymizer = table_masking_worker.anonymizer

        # 读取文件
        if file_extension == ".csv":
            # 尝试不同的编码
            supported_encodings = ["utf-8", "gbk", "gb18030"]
            df_pl = None
            for encoding in supported_encodings:
                try:
                    df_pl = pl.read_csv(
                        io.BytesIO(file_content),
                        encoding=encoding,
                        null_values=["", "NULL", "null", "None"],
                        infer_schema_length=10000,  # 推断模式
                    )
                    logger.bind(type="TASK", request_id=request_id).info(
                        f"Successfully decoded CSV with encoding: {encoding}"
                    )
                    break
                except Exception:
                    continue
            if df_pl is None:
                raise ValueError("文件解码失败，尝试了所有支持的编码")
        else:  # .xlsx
            # 对于 Excel 文件，先用 pandas 读取，然后转换为 Polars
            df_pandas = pd.read_excel(io.BytesIO(file_content), keep_default_na=False)
            df_pl = pl.from_pandas(df_pandas)

        logger.bind(type="TASK", request_id=request_id).info(
            f"File loaded - Rows: {len(df_pl)}, Columns: {len(df_pl.columns)}"
        )

        # 检测列名是否存在
        missing_columns = [
            col for col in columns_dict.keys() if col not in df_pl.columns
        ]
        if missing_columns:
            raise ValueError(f"列名不存在: {missing_columns}")

        # 逐列处理
        processing_results = {}
        total_texts_processed = 0
        total_entities_found = 0
        column_stats = {}
        all_masking_operations = []

        for idx, (column, method) in enumerate(columns_dict.items(), 1):
            column_start_time = time.time()

            logger.bind(type="TASK", request_id=request_id).info(
                f"Processing column {idx}/{len(columns_dict)}: {column} with method: {method}"
            )

            # 获取列数据（转换为 list 用于处理）
            column_data = df_pl[column].to_list()
            non_empty_data = [
                text for text in column_data if text is not None and str(text).strip()
            ]
            total_texts_processed += len(non_empty_data)

            if method == "text":
                # 对于文本处理，仍然使用原有的 func_f 方法，但分块处理以节省内存
                chunk_size = 100  # 每次处理100个文本
                all_processed_texts = []
                all_entities = []
                all_chunk_masking_ops = []

                for i in range(0, len(non_empty_data), chunk_size):
                    chunk = non_empty_data[i : i + chunk_size]

                    # 处理当前块
                    chunk_results = table_masking_worker.func_f(
                        chunk, request_id=request_id
                    )

                    # 收集结果
                    if "text" in chunk_results:
                        all_processed_texts.extend(chunk_results["text"])

                    if "entities" in chunk_results:
                        all_entities.extend(chunk_results["entities"])

                    if "masking_operations" in chunk_results:
                        for op in chunk_results["masking_operations"]:
                            op_with_context = {
                                "column": column,
                                "original_text": op["original_text"],
                                "entity_type": op["entity_type"],
                                "replaced_text": op["replaced_text"],
                                "applied_method": op.get("applied_method", "text"),
                                "original_text_index": op.get("original_text_index", -1)
                                + i,
                            }
                            all_chunk_masking_ops.append(op_with_context)

                # 将处理结果映射回 Polars DataFrame
                processed_count = 0
                processed_column_data = []

                for text in column_data:
                    if text is not None and str(text).strip():
                        if processed_count < len(all_processed_texts):
                            processed_column_data.append(
                                all_processed_texts[processed_count]
                            )
                            processed_count += 1
                        else:
                            processed_column_data.append(text)
                    else:
                        processed_column_data.append(text)

                # 更新列数据
                df_pl = df_pl.with_columns(
                    [pl.Series(processed_column_data).alias(column)]
                )

                # 统计结果
                column_entities = sum(len(entities) for entities in all_entities)
                total_entities_found += column_entities

                column_stats[column] = {
                    "method": "text",
                    "texts_processed": len(non_empty_data),
                    "entities_found": column_entities,
                    "avg_entities_per_text": column_entities / len(non_empty_data)
                    if non_empty_data
                    else 0,
                    "processing_time": time.time() - column_start_time,
                }

                processing_results[column] = {
                    "method": "text",
                    "processed_count": len(all_processed_texts),
                    "entities_found": column_entities,
                }

                all_masking_operations.extend(all_chunk_masking_ops)

            else:
                # apply_method_series 输入从list换为series
                processed_data = anonymizer.apply_method_series(column_data, method)

                # 将处理结果更新到 Polars DataFrame
                df_pl = df_pl.with_columns([pl.Series(processed_data).alias(column)])

                # 记录处理操作
                column_masking_ops = []
                for idx_val, (original, processed) in enumerate(
                    zip(column_data, processed_data)
                ):
                    if (
                        original != processed
                        and original is not None
                        and str(original).strip()
                    ):
                        column_masking_ops.append(
                            {
                                "column": column,
                                "original_text": str(original),
                                "entity_type": method.upper(),
                                "replaced_text": str(processed),
                                "applied_method": method,
                                "original_text_index": idx_val,
                            }
                        )

                all_masking_operations.extend(column_masking_ops)

                column_stats[column] = {
                    "method": method,
                    "texts_processed": len(non_empty_data),
                    "transformations": len(column_masking_ops),
                    "avg_transformations_per_text": len(column_masking_ops)
                    / len(non_empty_data)
                    if non_empty_data
                    else 0,
                    "processing_time": time.time() - column_start_time,
                }

                processing_results[column] = {
                    "method": method,
                    "processed_count": len(processed_data),
                    "transformations_count": len(column_masking_ops),
                }

            logger.bind(type="TASK", request_id=request_id).info(
                f"Column {column} processed in {time.time() - column_start_time:.2f}s"
            )

        # 保存处理后的文件
        output_filename = f"{request_id}_{filename}"
        output_file = TASK_RESULTS_DIR / output_filename

        # 将 Polars DataFrame 保存为文件
        if file_extension == ".csv":
            df_pl.write_csv(output_file)
        else:  # .xlsx
            # 对于 Excel，转换为 pandas 后保存
            df_pandas_final = df_pl.to_pandas()
            df_pandas_final.to_excel(output_file, index=False, engine="openpyxl")

        processing_time = time.time() - start_time

        # 构建汇总结果
        aggregation_summary = {
            "file_name": filename,
            "total_rows": len(df_pl),
            "total_columns": len(columns_dict),
            "total_texts_processed": total_texts_processed,
            "total_entities_found": total_entities_found,
            "total_transformations": len(all_masking_operations),
            "processing_time": processing_time,
            "avg_processing_time_per_text": processing_time / total_texts_processed
            if total_texts_processed > 0
            else 0,
            "column_stats": column_stats,
        }

        logger.bind(type="TASK", request_id=request_id).success(
            f"Task completed - File: {filename}, Time: {processing_time:.2f}s, "
            f"Rows: {len(df_pl)}, Entities: {total_entities_found}"
        )

        # 返回结果
        return {
            "success": True,
            "file_path": str(output_file),
            "filename": output_filename,
            "original_filename": filename,
            "processed_columns": list(columns_dict.keys()),
            "column_methods": columns_dict,
            "total_rows": len(df_pl),
            "aggregation_results": aggregation_summary,
            "request_id": request_id,
        }

    except Exception as e:
        logger.bind(type="TASK", request_id=request_id).error(
            f"Task failed - Error: {str(e)}"
        )
        raise