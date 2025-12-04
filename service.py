import base64
import hashlib
import io
import json
import math
import os
import random
import re
import secrets
import shutil
import sys
import tempfile
import time
import unicodedata
import uuid
from copy import copy
from datetime import date, datetime
from pathlib import Path
from typing import List, Union

import hanlp
import pandas as pd
import polars as pl
import yaml
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from cryptography.fernet import Fernet
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from func_timeout import func_set_timeout
from gmssl.sm4 import SM4_DECRYPT, SM4_ENCRYPT, CryptSM4
from loguru import logger
from pyunit_address import Address, find_address
from transformers import pipeline
from fastapi import BackgroundTasks

from rq import Queue
from redis import Redis
from fastapi.responses import FileResponse
import csv

# Configure loguru logging

# 确保日志目录存在
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 使用绝对路径
log_file = log_dir / "data-masking-service.log"

logger.remove()  # Remove default handler
logger.add(
    str(log_file),
    rotation="500 MB",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="INFO",
    enqueue=True,
    catch=True,
)

# 添加一个控制台输出以便调试
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
    level="INFO",
)

# 测试日志是否正常工作
logger.info("Logger initialized successfully")

# Load configuration and initialize services
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化 Redis 和 RQ
redis_conn = Redis(
    host=config["server"]["redis_host"],
    port=config["server"]["redis_port"],
    db=config["server"]["redis_db"],
    decode_responses=False,
)
task_queue = Queue("table_masking", connection=redis_conn)
logger.info("Redis and RQ initialized successfully")

# 确保输出目录存在
TASK_RESULTS_DIR = Path("static")
TASK_RESULTS_DIR.mkdir(exist_ok=True)


class CryptoFileManager:
    def __init__(self, key: str = "x57Lrc7M9K0xPU2kJ5zB9c2Pg7vav_OibEwKi3euD8g="):
        self.key = key.encode() if isinstance(key, str) else key
        self.fernet = Fernet(self.key)

    def encrypt_file(self, input_path: str, output_path: str):
        with open(input_path, "rb") as f:
            data = f.read()
        encrypted = self.fernet.encrypt(data)
        with open(output_path, "wb") as f:
            f.write(encrypted)

    def decrypt_file(self, input_path: str, suffix: str = "") -> str:
        with open(input_path, "rb") as f:
            enc_data = f.read()
        data = self.fernet.decrypt(enc_data)

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return tmp_path

    def encrypt_folder(self, folder_path: str, output_path: str):
        tmp_zip = shutil.make_archive(tempfile.mktemp(), "zip", folder_path)
        self.encrypt_file(tmp_zip, output_path)
        os.remove(tmp_zip)

    def decrypt_folder(self, input_path: str) -> str:
        tmp_dir = tempfile.mkdtemp()
        decrypted_zip = os.path.join(tmp_dir, "model.zip")

        with open(input_path, "rb") as f:
            enc_data = f.read()
        data = self.fernet.decrypt(enc_data)
        with open(decrypted_zip, "wb") as f:
            f.write(data)

        shutil.unpack_archive(decrypted_zip, tmp_dir)
        os.remove(decrypted_zip)
        return tmp_dir


class TextMasking:
    def __init__(
        self,
        model_path: str,
        rules_file_path: str,
        medicine_path: str,
        name_path: str,
        extra_rules_file_path: str = None,
        crypto_manager: CryptoFileManager = None,
        alg: dict = None,
    ):
        """
        加载模型和正则规则
        """
        default_rules_path = crypto_manager.decrypt_file(rules_file_path)
        medicine_list = pd.read_csv(medicine_path)
        self.medicine_df = []
        self.medicine_df.extend(medicine_list["通用名称"].dropna().tolist())
        model_path = crypto_manager.decrypt_folder(model_path)
        self.rules_df = pd.read_csv(default_rules_path)
        # with open(name_path, "r", encoding="utf-8") as f:
        #     self.name_list = [line.strip() for line in f if line.strip()]
        if extra_rules_file_path:
            # 用户自定义规则，无加密
            extra_rules_df = pd.read_csv(extra_rules_file_path)
            self.rules_df = pd.concat(
                [self.rules_df, extra_rules_df], ignore_index=True
            )
        self.model = hanlp.load(model_path)
        os.remove(default_rules_path)
        shutil.rmtree(model_path)

        self.p = pipeline(
            "ner",
            model="./roberta-base-finetuned-cluener2020-chinese",
            tokenizer="./roberta-base-finetuned-cluener2020-chinese",
            aggregation_strategy="average",
            device="cuda:0",
        )

        self.alg = alg
        self.anonymizer = Anonymizer(key="mysecretkey123")

    def _func_overlap(self, e):
        if not e:
            return []
        resolved_e = [e[0]]
        for current_entity in e[1:]:
            last_entity = resolved_e[-1]
            # 如果出现重叠
            if current_entity[2] < last_entity[3]:
                winner = None
                # 1. 优先更长的
                len_last = last_entity[3] - last_entity[2]
                len_current = current_entity[3] - current_entity[2]
                if len_current > len_last:
                    winner = current_entity
                elif len_last > len_current:
                    winner = last_entity
                else:
                    # 长度相等
                    # 2. 优先 "CARD" 或 "NUMBER"
                    last_is_special = (
                        "CARD" in last_entity[0] or "NUMBER" in last_entity[0]
                    )
                    current_is_special = (
                        "CARD" in current_entity[0] or "NUMBER" in current_entity[0]
                    )
                    if current_is_special and not last_is_special:
                        winner = current_entity
                    elif last_is_special and not current_is_special:
                        winner = last_entity
                    else:
                        # 3. 优先 "PERSON"
                        last_is_person = last_entity[0] == "PERSON"
                        current_is_person = current_entity[0] == "PERSON"
                        if current_is_person and not last_is_person:
                            winner = current_entity
                        else:
                            # 默认保留先出现的
                            winner = last_entity
                resolved_e[-1] = winner
            else:  # 不重叠
                resolved_e.append(current_entity)
        for entity in resolved_e:
            entity[0] = entity[0].upper()
        return resolved_e

    def _func_st(self, text, target_length=3000, max_length=5000):
        if len(text) <= max_length:
            return [text]

        punctuation_patterns = [
            r"[。！？]",
            r"[；：]",
            r"[，、]",
            r'[）】」』"]',
            r"[\n\r]",
            r"[ \t]",
        ]

        chunks = []
        remaining_text = text

        while len(remaining_text) > target_length:
            if len(remaining_text) <= max_length:
                chunks.append(remaining_text)
                break

            best_split_pos = target_length
            search_start = max(target_length - 1000, target_length // 2)
            search_end = min(target_length + 1000, len(remaining_text))
            search_text = remaining_text[search_start:search_end]

            found_split = False
            for pattern in punctuation_patterns:
                matches = list(re.finditer(pattern, search_text))
                if matches:
                    # 找到离目标长度最近的匹配
                    target_in_search = target_length - search_start
                    best_match = min(
                        matches, key=lambda m: abs(m.end() - target_in_search)
                    )
                    best_split_pos = search_start + best_match.end()
                    found_split = True
                    break

            # 如果没有找到合适的标点符号，就在目标位置分割
            if not found_split:
                best_split_pos = target_length

            # 分割文本
            chunk = remaining_text[:best_split_pos].rstrip()
            if chunk:  # 确保chunk不为空
                chunks.append(chunk)

            remaining_text = remaining_text[best_split_pos:].lstrip()

        # 添加剩余文本
        if remaining_text:
            chunks.append(remaining_text)

        return chunks

    def _func_0(self, text, e, request_id=None):
        """
        脱敏操作
        """
        new_text = text
        masking_operations = []
        for i, p in enumerate(sorted(e, key=lambda e: -e[-2])):
            original_text = text[p[-2] : p[-1]]
            # id = f"<{p[0]}>".upper()
            if p[0].lower() not in self.alg.keys():
                continue
            id = self.anonymizer.apply_method(p[1], self.alg[(p[0]).lower()])
            new_text = new_text[: p[-2]] + str(id) + new_text[p[-1] :]

            # 记录脱敏操作
            masking_operations.append(
                {
                    "original_text": original_text,
                    "entity_type": p[0],
                    "replaced_text": id,
                    "applied_method": "text",  # 标识为文本脱敏方法
                }
            )

        # 记录脱敏操作日志
        if masking_operations:
            if request_id:
                logger.bind(type="MASKING", request_id=request_id).info(
                    f"Masking operations: {json.dumps(masking_operations, ensure_ascii=False)}"
                )
            else:
                logger.bind(type="MASKING").info(
                    f"Masking operations: {json.dumps(masking_operations, ensure_ascii=False)}"
                )

        return new_text, masking_operations

    def _func_4(self, text, l1: list):
        l2 = [
            quard
            for quard in l1
            if not (quard[0] in ["ORGANIZATION"] and len(quard[1]) == 1)
        ]
        if len(l2) == 0:
            return l2
        name_list = [quard[0] for quard in l2]

        if False:
            if (set(["PERSON", "LOCATION", "ORGANIZATION", "TIME", "DATE"])) == 0:
                return l2

        while True:
            l3 = self._func_5(text, l2)
            if l3 == l2:
                break
            l2 = l3

        def clean(text, start, window=2):
            context = text[max(start - window, 0) : start]
            return re.sub(r"[\s_：]+", "", context)

        filtered_list = []
        for quard in l2:
            ctx = clean(text, quard[2], window=10)
            if quard[0] == "地址":
                if not any(kw in ctx[-3:] for kw in ["地址", "地点"]):
                    continue
            elif quard[0] == "ORGANIZATION":
                if not any(kw in ctx[-3:] for kw in ["账户", "名"]):
                    continue
            filtered_list.append(quard)

        if len(l2) == 0:
            return l2

        return l2

    def _func_5(self, text, l1: list):
        l2 = []
        i = 0
        length = len(l1)
        while i < length:
            quard = l1[i]
            if quard[0] in {"LOCATION", "ORGANIZATION", "PERSON"}:
                p2 = quard[2]
                new_text = quard[1]
                p3 = quard[3]
                p0 = [quard[0]]
                i += 1

                while (
                    i < length
                    and l1[i][0] in {"LOCATION", "ORGANIZATION", "PERSON"}
                    and ((l1[i][2] - p3) <= 0)
                ):
                    p0.append(l1[i][0])
                    p3 = l1[i][3]
                    new_text += l1[i][1]
                    i += 1
                final_label = "PERSON" if "PERSON" in p0 else "LOCATION"
                l2.append([final_label, new_text, p2, p3])

                # 检查 LOCATION 后续的特定模式（紧跟着的5个字符）
                if p3 < len(text):
                    next_five_chars = text[p3 : p3 + 5]
                    # match = re.match(
                    # r'^[A-Za-z0-9]+(号|栋|楼|座|层|单元)', next_five_chars)
                    match = re.match(
                        r"^[\dA-Za-z一二三四五六七八九十百千万]+(号|栋|楼|座|层|单元)",
                        next_five_chars,
                    )
                    if match:
                        l2[-1][1] += match.group(0)
                        l2[-1][3] += len(match.group(0))

            elif quard[0] in {"TIME", "日期"}:
                p2 = quard[2]
                new_text = quard[1]
                p3 = quard[3]
                i += 1
                while i < length and l1[i][0] in {"TIME", "日期"} and p3 == l1[i][2]:
                    p3 = l1[i][3]
                    new_text += l1[i][1]
                    i += 1
                l2.append([quard[0], new_text, p2, p3])

            else:
                l2.append(quard)
                i += 1
        return l2

    def _func_6(self, text, l1: list, finds):
        l2 = [
            quard
            for quard in l1
            if not (quard[0] in ["LOCATION", "ORGANIZATION"] and len(quard[1]) == 1)
        ]
        if len(l2) == 0:
            return l2
        name_list = [quard[0] for quard in l2]

        find_list = []

        for find in finds:
            start_pos = text.find(find)
            if start_pos == -1:
                continue
            end_pos = start_pos + len(find)

            find_list.append(["LOCATION", find, start_pos, end_pos])

        if False:
            if (set(["PERSON", "LOCATION", "ORGANIZATION", "TIME", "DATE"])) == 0:
                return l2

        while True:
            l3 = self._clean_2(text, l2, find_list)
            if l3 == l2:
                break
            l2 = l3

        def clean(text, start, window=2):
            context = text[max(start - window, 0) : start]
            return re.sub(r"[\s_：]+", "", context)

        filtered_list = []
        for quard in l2:
            ctx = clean(text, quard[2], window=10)
            if quard[0] == "地址":
                if not any(kw in ctx[-3:] for kw in ["地址", "地点"]):
                    continue
            elif quard[0] == "ORGANIZATION":
                if not any(kw in ctx[-3:] for kw in ["账户", "名"]):
                    continue
            filtered_list.append(quard)

        if len(l2) == 0:
            return l2

        return l2

    def _clean_2(self, text, l1: list, find_list: list):
        l2 = []
        i = 0
        length = len(l1)
        while i < length:
            quard = l1[i]
            if quard[0] in {"LOCATION", "ORGANIZATION"}:
                p2 = quard[2]
                new_text = quard[1]
                p3 = quard[3]
                p0 = {quard[0]}
                i += 1

                while (
                    i < length
                    and l1[i][0] in {"LOCATION", "ORGANIZATION"}
                    and ((l1[i][2] - p3) <= 1)
                ):
                    p0.add(l1[i][0])
                    p3 = l1[i][3]
                    new_text += l1[i][1]
                    i += 1
                final_label = "LOCATION" if p0 == {"LOCATION"} else "ORGANIZATION"
                l2.append([final_label, new_text, p2, p3])

                # 检查 LOCATION 后续的特定模式（紧跟着的5个字符）
                if p3 < len(text):
                    next_five_chars = text[p3 : p3 + 5]
                    match = re.match(
                        r"^[\dA-Za-z一二三四五六七八九十百千万]+(号|栋|楼|座|层|单元)",
                        next_five_chars,
                    )
                    if match:
                        l2[-1][1] += match.group(0)
                        l2[-1][3] += len(match.group(0))

            elif quard[0] in {"TIME", "日期"}:
                p2 = quard[2]
                new_text = quard[1]
                p3 = quard[3]
                i += 1
                while i < length and l1[i][0] in {"TIME", "日期"} and p3 == l1[i][2]:
                    p3 = l1[i][3]
                    new_text += l1[i][1]
                    i += 1
                l2.append([quard[0], new_text, p2, p3])

            else:
                l2.append(quard)
                i += 1

        for find in find_list:
            for sensi in l2:
                if sensi[2] >= find[2] and sensi[3] <= find[3]:
                    l2.remove(sensi)
                    l2.append(find)
                elif sensi[2] > find[3]:
                    break
        l2 = sorted(l2, key=lambda x: x[2])

        return l2

    def _valid_1(self, code):
        city = {
            "11": "北京",
            "12": "天津",
            "13": "河北",
            "14": "山西",
            "15": "内蒙古",
            "21": "辽宁",
            "22": "吉林",
            "23": "黑龙江",
            "31": "上海",
            "32": "江苏",
            "33": "浙江",
            "34": "安徽",
            "35": "福建",
            "36": "江西",
            "37": "山东",
            "41": "河南",
            "42": "湖北",
            "43": "湖南",
            "44": "广东",
            "45": "广西",
            "46": "海南",
            "50": "重庆",
            "51": "四川",
            "52": "贵州",
            "53": "云南",
            "54": "西藏",
            "61": "陕西",
            "62": "甘肃",
            "63": "青海",
            "64": "宁夏",
            "65": "新疆",
            "71": "台湾",
            "81": "香港",
            "82": "澳门",
            "91": "国外",
        }

        if code[0:2] not in city:
            return False

        if len(code) == 18:
            factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
            parity = ["1", "0", "X", "9", "8", "7", "6", "5", "4", "3", "2"]
            code_list = list(code)
            sum = 0
            for i in range(17):
                sum += int(code_list[i]) * factors[i]
            last = parity[sum % 11]
            if last != code_list[17]:
                # print("校验位错误")
                return False

        return True

    def _valid_2(self, code):
        city = {
            "11": "北京",
            "12": "天津",
            "13": "河北",
            "14": "山西",
            "15": "内蒙古",
            "21": "辽宁",
            "22": "吉林",
            "23": "黑龙江",
            "31": "上海",
            "32": "江苏",
            "33": "浙江",
            "34": "安徽",
            "35": "福建",
            "36": "江西",
            "37": "山东",
            "41": "河南",
            "42": "湖北",
            "43": "湖南",
            "44": "广东",
            "45": "广西",
            "46": "海南",
            "50": "重庆",
            "51": "四川",
            "52": "贵州",
            "53": "云南",
            "54": "西藏",
            "61": "陕西",
            "62": "甘肃",
            "63": "青海",
            "64": "宁夏",
            "65": "新疆",
            "71": "台湾",
            "81": "香港",
            "82": "澳门",
            "91": "国外",
        }

        if code[0:2] not in city:
            return False

        return True

    def _is_fa(self, s1, e1, s2, e2):
        return not (e1 <= s2 or e2 <= s1)

    def _func_o(self, text, data):
        for item in data:
            item_idx = None

            # 过长退出
            if False and len(text) > 1000:
                return text

            for idx, reg in enumerate(text):
                if self._is_fa(reg[2], reg[3], item[2], item[3]):
                    item_idx = idx
                    break

            if item_idx is not None:
                reg = text[item_idx]
                if (item[0] == "PERSON") & (item[2] < reg[2] or item[3] > reg[3]):
                    text[item_idx] = item
            else:
                text.append(item)
                while False:
                    break

        return text

    def _func_h(self, text: str, rule_df: pd.DataFrame):
        d1 = []
        result = []
        for i in range(len(rule_df)):
            pattern = rule_df.loc[i, "rule"]
            label = int(rule_df.loc[i, "group_label"])
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = rule_df.loc[i, "name"]
                found_text = match.group(label)
                if name == "其他证号":
                    name = match.group(1)
                if name == "ID_Number" and not self._valid_1(found_text):
                    continue
                elif name == "ID_Number_2" and not self._valid_2(found_text):
                    continue
                elif ("CARD" in name) or ("CREDIT" in name):
                    name = "BANK_CARD"
                start = match.start(label)
                end = match.end(label)
                overlap = False
                for t in range(len(d1)):
                    if self._is_fa(d1[t][2], d1[t][3], start, end):
                        overlap = True
                        break
                if not overlap:
                    d1.append([name, found_text, start, end])
        result.sort(key=lambda x: x[0])
        return d1

    def _func_b(self, q, segments):
        after = [
            "颗粒",
            "口溶颗粒",
            "凝胶",
            "糖浆",
            "蛋白粉",
            "注射液",
            "疫苗",
            "激素",
            "干扰素",
            "喷雾",
            "蛋白",
            "胶丸",
            "肠溶片",
            "分散片",
            "泡腾片",
            "贴片",
            "咀嚼片",
            "钙片",
            "含片",
            "薄膜衣片",
            "糖衣片",
            "舌下片",
            "控释片",
            "口溶片",
            "分层片",
            "口腔崩解片",
            "透皮贴片",
            "合剂",
            "制剂",
            "冲剂",
            "擦剂",
            "搽剂",
            "贴剂",
            "喷雾剂",
            "气雾剂",
            "洗剂",
            "散剂",
            "鼻喷剂",
            "粉雾剂",
            "滴剂",
            "酊剂",
            "涂膜剂",
            "糊剂",
            "缓释片",
            "营养剂",
            "干混悬剂",
            "细粉剂",
            "糊剂",
            "干粉吸入剂",
            "雾化吸入剂",
            "吸入溶液",
            "胶囊",
            "缓释胶囊",
            "软胶囊",
            "肠溶胶囊",
            "冻干胶囊",
            "口服液",
            "混悬液",
            "滴眼液",
            "溶液",
            "滴鼻液",
            "洗液",
            "含漱液",
            "滴耳液",
            "药酒",
            "乳膏",
            "软膏",
            "眼膏",
            "眼药",
            "贴膏",
            "浸膏",
            "药膏",
            "膏药",
            "滴丸",
            "鱼肝油",
            "鱼油",
            "袋泡茶",
            "散",
            "油",
            "丸",
            "病毒",
            "寄生虫",
            "细胞",
            "菌",
            "细菌",
            "真菌",
            "杆菌",
            "支原体",
            "衣原体",
            "螺旋体",
            "立克次体",
            "病",
            "症",
            "征",
            "综合征",
            "症候群",
            "障碍",
            "畸形",
            "缺陷",
            "癌",
            "瘤",
            "肉瘤",
            "淋巴瘤",
            "炎",
            "结节",
            "囊肿",
            "彩超",
            "超声",
            "化疗",
            "放疗",
            "治疗",
            "CT",
            "MRI",
            "染色",
            "测验",
            "测试",
            "检查",
            "检测",
            "效应",
            "方法",
            "试验",
            "单抗",
            "血清",
            "因子",
            "微球",
            "肾上腺素",
            "宝塔糖",
            "毒素",
            "提取物",
            "提取物",
            "冻干粉",
            "电解质",
            "免疫核糖核酸",
        ]
        before = [
            "注射用",
            "静脉注射",
            "皮下注射",
            "肌肉注射",
            "静脉滴注",
            "动脉注射",
            "鞘内注射",
            "关节腔注射",
            "皮内注射",
            "腹腔注射",
            "口服",
            "舌下含服",
            "滴眼用",
            "滴耳用",
            "滴鼻用",
            "鼻腔吸入",
            "吸入",
            "外用涂抹",
            "透皮贴附",
            "阴道给药",
            "直肠给药",
        ]
        # before_pattern = re.compile("|".join(map(re.escape, before)))
        # after_pattern = re.compile("|".join(map(re.escape, after)))
        after_pattern = r"[\u4e00-\u9fff\w]+(" + "|".join(map(re.escape, after)) + r")"
        before_pattern = "(" + "|".join(map(re.escape, before)) + r")[\u4e00-\u9fff\w]*"
        pattern = re.compile(f"{after_pattern}|{before_pattern}")

        e = q[1]

        for seg in segments:
            if e in seg:
                if pattern.search(seg):
                    return True
                # return False
                idx = seg.index(e)
                b_e = seg[idx - 1 : len(e) + idx] if idx > 1 else " "
                a_e = (
                    seg[idx : len(e) + idx + 1] if len(e) + idx < len(seg) - 1 else " "
                )
                if any((b_e or a_e) in item for item in self.medicine_df) or (
                    e in self.medicine_df
                ):
                    return True
                else:
                    return False
        return False

    def _seg2pos(self, sent_tok_list):
        pos_list = [0]
        for i in range(len(sent_tok_list)):
            pos_list.append(len(sent_tok_list[i]) + pos_list[i])
        return pos_list

    def _func_r(self, text_list):
        r1 = self.model(text_list, tasks="ner/pku")
        s1 = [self._seg2pos(x) for x in r1["tok/fine"]]
        s2 = r1["ner/pku"]

        mask = {"PERSON"}

        s3 = [
            [[q2[0], "PERSON", q2[2], q2[3]] for q2 in q1 if q2[1] == "nr"] for q1 in s2
        ]

        return s3, s1

    def _merge(self, text_list, r1, r1_p):
        # 目前使用uer的地址识别
        result = []
        for i in r1:
            if "LOCATION" not in i[0]:
                result.append(i)
        for i in r1_p:
            result.append(i)
        result = sorted(result, key=lambda x: x[2])
        return result

    def _func_1(self, text, is_type=False, request_id=None):
        start_time = time.time()
        original_text = text[0]

        text_list = [re.sub(r"\s+|\|", "，", t) for t in text]
        segments = re.split(r"[，。；,.;/\|]|//", text[0])
        # text_list[0]=mask_all_exact(text_list[0],data)
        s1, s2 = self._func_r(text_list)
        # print(s1)

        s3 = self._func_h(text[0], self.rules_df)
        s4 = copy(s3)

        finds = []
        if is_type == True:
            address = Address(is_max_address=True)
            finds = find_address(address, text[0])

        i = 0
        for q in s1[i]:
            is_m = False
            idx_1 = s2[i][q[2]]
            idx_2 = s2[i][q[3]]
            e = [q[1], q[0], idx_1, idx_2]
            if (q[1] == "PERSON") & (
                (len(q[0]) == 1)
                or q[0] in ["孟德尔", "高尔基"]
                or q[0].endswith("多普勒")
            ):
                continue
            pattern = r"(患者|病人|病患|爱人|妻子|丈夫|夫人|先生|子女|儿子|女儿|哥哥|姐姐|弟弟|妹妹|\d+(?:床|病床))"
            if text[0][idx_1 - 1] in [":", "：", "是", "为"]:
                match = re.search(pattern, text[0][idx_1 - 3 : idx_1 - 1])
            else:
                match = re.search(pattern, text[0][idx_1 - 2 : idx_1])

            if not match:
                is_m = self._func_b(e, segments)
                if is_m:
                    continue
            s4 = self._func_o(s4, [e])
        if len(s4) != 0:
            s4.sort(key=lambda x: x[2])

        if len(finds) == 0:
            r1 = self._func_4(text[0], s4)
        else:
            r1 = self._func_6(text[0], s4, finds)

        p_result_raw = self.p(text_list)
        # print(p_result_raw)
        # pprint(p_result_raw)
        r1_p = []
        for i in p_result_raw:
            item = []
            for j in i:
                if (
                    len(j["word"].replace(" ", "")) > 2
                    and j["entity_group"] == "address"
                ):
                    item.append(
                        ["LOCATION", j["word"].replace(" ", ""), j["start"], j["end"]]
                    )
            r1_p.append(item)

        # 合并结果。后续艺伟模型如果更新了地址，直接把这步和self.p相关的去掉就行
        r1 = self._merge(text_list, r1, r1_p[0])
        # 处理重叠实体，目前优先顺序：长度→正则（手机/卡号等）→人名
        r1 = self._func_overlap(r1)
        # 处理原文本
        r2, masking_ops = self._func_0(text[0], r1, request_id)

        # 记录整体脱敏过程日志
        processing_time = time.time() - start_time
        if request_id:
            logger.bind(type="MASKING", request_id=request_id).info(
                f"Text masking completed - Original length: {len(original_text)}, Processed time: {processing_time:.4f}s, Entities found: {len(r1)}"
            )
        else:
            logger.bind(type="MASKING").info(
                f"Text masking completed - Original length: {len(original_text)}, Processed time: {processing_time:.4f}s, Entities found: {len(r1)}"
            )

        return r2, r1, masking_ops

    @func_set_timeout(3000)
    def func_f(self, text_list_request, is_type=False, request_id=None):
        r2_list, r1_list = [], []
        total_entities = 0
        total_processing_time = 0
        all_masking_operations = []  # 收集所有文本的脱敏操作

        for text in text_list_request:
            if len(text) > 5000:
                # 分割文本
                text_chunks = self._func_st(text)

                all_r2 = []
                all_r1 = []

                # 记录当前在原文中的偏移量
                current_offset = 0

                # 逐段处理
                text_masking_ops = []  # 收集当前文本所有chunk的脱敏操作
                for chunk in text_chunks:
                    chunk_r2, chunk_r1, chunk_masking_ops = self._func_1(
                        [chunk], is_type=is_type, request_id=request_id
                    )
                    all_r2.append(chunk_r2)

                    # 调整实体索引，加上当前偏移量
                    for entity in chunk_r1:
                        adjusted_entity = [
                            entity[0].upper(),  # 类别
                            entity[1],  # 内容
                            entity[2] + current_offset,  # 起始index + 偏移量
                            entity[3] + current_offset,  # 结束index + 偏移量
                        ]
                        all_r1.append(adjusted_entity)

                    # 调整脱敏操作的位置信息
                    for op in chunk_masking_ops:
                        adjusted_op = {
                            "original_text": op["original_text"],
                            "entity_type": op["entity_type"].upper(),
                            "replaced_text": op["replaced_text"].upper(),
                            "original_text_index": text.index(op["original_text"])
                            if op["original_text"] in text
                            else -1,
                            # "source_text": text,  # 添加原始文本上下文
                        }
                        text_masking_ops.append(adjusted_op)

                    # 更新偏移量：加上当前chunk的原始长度
                    current_offset += len(chunk)

                # 合并结果
                r2 = "".join(all_r2)  # 所有r2直接拼接
                r1 = all_r1  # 调整后的实体列表
                masking_ops = text_masking_ops  # 当前文本的脱敏操作
            else:
                # 原始处理方式
                r2, r1, masking_ops = self._func_1(
                    [text], is_type=is_type, request_id=request_id
                )
                # 为脱敏操作添加原始文本上下文
                for op in masking_ops:
                    # op["source_text"] = text
                    op["original_text_index"] = (
                        text.index(op["original_text"])
                        if op["original_text"] in text
                        else -1
                    )

            r2_list.append(r2)
            r1_list.append(r1)
            total_entities += len(r1)
            all_masking_operations.extend(masking_ops)  # 收集所有脱敏操作

        # 记录批量处理结果
        if request_id:
            logger.bind(type="MASKING", request_id=request_id).info(
                f"Batch processing completed - Total texts: {len(text_list_request)}, Total entities: {total_entities}"
            )

        return {
            "text": r2_list,
            "entities": r1_list,
            "masking_operations": all_masking_operations,
        }


class Anonymizer:
    def __init__(self, key: str = "20251031"):
        self.key = key
        # AES 和 SM4 的 key 和 IV 初始化
        self._aes_key = hashlib.sha256(key.encode()).digest()
        self._aes_iv = b"abcdef9876543210"
        self._sm4_key = hashlib.sha256(key.encode()).digest()[:16]
        self._sm4_iv = b"abcdef9876543210"

        # df = pd.read_csv('/home/richorange/yiwei/jiaofu_masking/jiaofu_mix/city.csv')
        df = pd.read_csv(config["paths"]["city_file"])
        provinces = df.iloc[:, 0].dropna().unique().tolist()
        cities = df.iloc[:, 1].dropna().unique().tolist()

        self.province_names = [
            p.replace("省", "")
            .replace("市", "")
            .replace("自治区", "")
            .replace("特别行政区", "")
            for p in provinces
        ]
        self.city_names = [c.replace("市", "") for c in cities]

        province_pattern = "|".join(
            sorted(map(re.escape, self.province_names), key=len, reverse=True)
        )
        city_pattern = "|".join(
            sorted(map(re.escape, self.city_names), key=len, reverse=True)
        )

        self.pattern = re.compile(
            rf"(?:(?P<province>{province_pattern})(?P<province_suffix>省|市|自治区|特别行政区)?)?"
            rf"(?:(?P<city>{city_pattern})(?P<city_suffix>市)?)?"
        )

    @staticmethod
    def m(e, mc="*", mp=None, rev=False):
        r = None
        if mp:
            s, t = mp
            if s < 0:
                s, t = len(e) + s, len(e) + t
            if rev:
                r = mc * s + e[s:t] + mc * (len(e) - t)
            else:
                r = e[:s] + mc * (t - s) + e[t:]
        else:
            r = mc * len(e)
        return r

    @staticmethod
    def d(s):
        p = r"^\d{4}-\d{2}-\d{2}$"
        if re.match(p, s):
            try:
                datetime.strptime(s, "%Y-%m-%d")
                return True
            except:
                return False
        return False

    @staticmethod
    def rd(y):
        sy, ey = datetime.now().year - y, datetime.now().year
        yr = random.randint(sy, ey)
        mo = random.randint(1, 12)
        dm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if mo == 2 and (yr % 4 == 0 and yr % 100 != 0 or yr % 400 == 0):
            dm[1] = 29
        day = random.randint(1, dm[mo - 1])
        return datetime(yr, mo, day).strftime("%Y-%m-%d")

    @staticmethod
    def rc():
        cs = list(
            "一乙二十丁厂七卜人入八九几儿了力乃刀又三于干亏士工土才寸下大丈与万上小口巾山千乞川亿个勺久凡及夕丸么广亡门义之尸弓己已子卫也女飞刃习叉马乡丰王井开夫天无元专云扎艺木五支厅不太犬区历尤友匹车巨牙屯比互切瓦止少日中冈贝内水见午牛手毛气升长仁什片仆化仇币仍仅斤爪反介父从今凶分乏公仓月氏勿欠风丹匀乌凤勾文六方火为斗忆订计户认心尺引丑巴孔队办以允予劝双书幻玉刊示末未击打巧正扑扒功扔去甘世古节本术可丙左厉右石布龙平灭轧东卡北占业旧帅归且旦目叶甲申叮电号田由史只央兄叼叫另叨叹四生失禾丘付仗代仙们仪白仔他斥瓜乎丛令用甩印乐句匆册犯外处冬鸟务包饥主市立闪兰半汁汇头汉宁穴它讨写让礼训必议讯记永司尼民出辽奶奴加召皮边发孕圣对台矛纠母幼丝式刑动扛寺吉扣考托老执巩圾扩扫地扬场耳共芒亚芝朽朴机权过臣再协西压厌在有百存而页匠夸夺灰达列死成夹轨邪划迈毕至此贞师尘尖劣光当早吐吓虫曲团同吊吃因吸吗屿帆岁回岂刚则肉网年朱先丢舌竹迁乔伟传乒乓休伍伏优伐延件任伤价份华仰仿伙伪自血向似后行舟全会杀合兆企众爷伞创肌朵杂危旬旨负各名多争色壮冲冰庄庆亦刘齐交次衣产决充妄闭问闯羊并关米灯州汗污江池汤忙兴宇守宅字安讲军许论农讽设访寻那迅尽导异孙阵阳收阶阴防奸如妇好她妈戏羽观欢买红纤级约纪驰巡寿弄麦形进戒吞远违运扶抚坛技坏扰拒找批扯址走抄坝贡攻赤折抓扮抢孝均抛投坟抗坑坊抖护壳志扭块声把报却劫芽花芹芬苍芳严芦劳克苏杆杠杜材村杏极李杨求更束豆两丽医辰励否还歼来连步坚旱盯呈时吴助县里呆园旷围呀吨足邮男困吵串员听吩吹呜吧吼别岗帐财针钉告我乱利秃秀私每兵估体何但伸作伯伶佣低你住位伴身皂佛近彻役返余希坐谷妥含邻岔肝肚肠龟免狂犹角删条卵岛迎饭饮系言冻状亩况床库疗应冷这序辛弃冶忘闲间闷判灶灿弟汪沙汽沃泛沟没沈沉怀忧快完宋宏牢究穷灾良证启评补初社识诉诊词译君灵即层尿尾迟局改张忌际陆阿陈阻附妙妖妨努忍劲鸡驱纯纱纳纲驳纵纷纸纹纺驴纽奉玩环武青责现表规抹拢拔拣担坦押抽拐拖拍者顶拆拥抵拘势抱垃拉拦拌幸招坡披拨择抬其取苦若茂苹苗英范直茄茎茅林枝杯柜析板松枪构杰述枕丧或画卧事刺枣雨卖矿码厕奔奇奋态欧垄妻轰顷转斩轮软到非叔齿些虎虏肾贤尚旺具果"
        )
        return random.choice(cs)

    @staticmethod
    def rn(d):
        mn, mx = 10 ** (d - 1), 10**d - 1
        return secrets.randbelow(mx - mn + 1) + mn

    def rm(self, data):
        sm, nm, res = {}, {}, []
        if not isinstance(data, list):
            data = [data]
        for x in data:
            mval = None
            if isinstance(x, str):
                if x not in sm:
                    if self.d(x):
                        mval = self.rd(10)
                    elif x == "无":
                        mval = "无"
                    else:
                        tmp = ""
                        for c in x:
                            if "CJK" in unicodedata.name(c):
                                tmp += self.rc()
                            elif unicodedata.category(c) == "Lu":
                                tmp += chr(random.randint(65, 90))
                            elif unicodedata.category(c) == "Ll":
                                tmp += chr(random.randint(97, 122))
                            elif unicodedata.category(c) == "Nd":
                                tmp += str(random.randint(0, 9))
                            else:
                                tmp += c
                        mval = tmp
                    sm[x] = mval
                else:
                    mval = sm[x]
            elif isinstance(x, int):
                if x in nm:
                    mval = nm[x]
                else:
                    if x >= 0:
                        mval = self.rn(len(str(x)))
                    else:
                        mval = 0 - self.rn(len(str(abs(x))))
                    nm[x] = mval
            elif isinstance(x, float):
                s = str(x).replace(",", "")
                dot = s.find(".")
                is_neg = x < 0
                af = s[1:] if is_neg else s
                if dot == -1:
                    dot = af.find(".")
                ip, dp = af[:dot], af[dot + 1 :]
                si = self.rn(len(ip))
                sd = self.rn(len(dp))
                sdata = f"{si}.{sd}"
                mval = float(sdata) if not is_neg else 0 - float(sdata)
            res.append(mval)
        return res[0] if not isinstance(data, list) else res

    @staticmethod
    def _empty_anonymize(entity):
        return ""

    def hash_anonymize(self, entity: str) -> str:
        """SHA-256 哈希脱敏"""
        return hashlib.sha256(entity.encode()).hexdigest()

    def address_anonymize(self, entity: str) -> str:
        match = self.pattern.search(entity)
        province = match.group("province") or ""
        province_suffix = match.group("province_suffix") or ""
        city = match.group("city") or ""
        city_suffix = match.group("city_suffix") or ""

        keep_prefix = province + province_suffix + city + city_suffix
        keep_len = len(keep_prefix)

        masked_len = len(entity) - keep_len
        masked_part = "*" * masked_len if masked_len > 0 else ""

        return keep_prefix + masked_part

    def truncate_anonymize(
        self, entity: str, keep_start: int = 0, keep_end: int = -1
    ) -> str:
        if keep_end is not None:
            return entity[keep_start:keep_end]
        else:
            return entity[keep_start:]

    def round_anonymize(self, entity, precision: int = -2):
        s_item = str(entity).replace(",", "")
        try:
            num = float(s_item)
            if precision >= 0:
                return round(num, precision)
            else:
                power_of_ten = 10 ** abs(precision)
                rounded_num = math.floor(num / power_of_ten) * power_of_ten
                return (
                    int(rounded_num) if rounded_num == int(rounded_num) else rounded_num
                )
        except ValueError:
            return entity

    def bucket_anonymize(self, entity, buckets: list = None) -> str:
        if buckets is None:
            buckets = [0, 100000, 200000, 300000, 400000]

        entity_str = str(entity).replace(",", "")
        try:
            data = float(entity_str)
        except ValueError:
            return entity_str

        for i in range(len(buckets)):
            if data <= buckets[i]:
                lower_bound = buckets[i - 1] if i > 0 else "-∞"
                return f"{lower_bound}-{buckets[i]}"

        return f"{buckets[-1]}+"

    def aes_encrypt(self, plaintext: str) -> str:
        cipher = AES.new(self._aes_key, AES.MODE_CBC, self._aes_iv)
        ciphertext = cipher.encrypt(pad(plaintext.encode("utf-8"), AES.block_size))
        return "8e4f1" + base64.b64encode(ciphertext).decode("utf-8") + "202cf"

    def aes_decrypt(self, ciphertext_b64: str) -> str:
        cipher = AES.new(self._aes_key, AES.MODE_CBC, self._aes_iv)
        plaintext = unpad(
            cipher.decrypt(base64.b64decode(ciphertext_b64)), AES.block_size
        )
        return plaintext.decode("utf-8")

    def sm4_encrypt(self, plaintext: str) -> str:
        crypt_sm4 = CryptSM4()
        crypt_sm4.set_key(self._sm4_key, SM4_ENCRYPT)
        ciphertext = crypt_sm4.crypt_cbc(self._sm4_iv, plaintext.encode("utf-8"))
        return "8e4f1" + base64.b64encode(ciphertext).decode("utf-8") + "202cf"

    def sm4_decrypt(self, ciphertext_b64: str) -> str:
        crypt_sm4 = CryptSM4()
        crypt_sm4.set_key(self._sm4_key, SM4_DECRYPT)
        plaintext = crypt_sm4.crypt_cbc(self._sm4_iv, base64.b64decode(ciphertext_b64))
        return plaintext.decode("utf-8")

    def apply_method(self, entity, method: str):
        is_list = isinstance(entity, list)
        data = entity if is_list else [entity]

        if method == "random":
            return self.rm(data)

        results = []
        for item in data:
            item_str = str(item)

            if method == "null":
                result = self._empty_anonymize(item)
            elif method == "hash":
                result = self.hash_anonymize(item_str)
            elif method == "location":
                result = self.address_anonymize(item_str)
            elif method == "truncate":
                result = self.truncate_anonymize(item_str)
            elif method == "round":
                result = self.round_anonymize(item, -2)
            elif method == "bucket":
                result = self.bucket_anonymize(item)
            elif method == "enc":
                result = self.aes_encrypt(item_str)
            elif method == "sm4":
                result = self.sm4_encrypt(item_str)
            else:
                result = item

            results.append(result)

        return results if is_list else results[0]

    def apply_method_series(self, series_data, method: str):
        """
        Vectorized method to apply anonymization to a pandas Series or similar iterable.
        Returns a list of processed values.
        """
        import numpy as np

        if method == "null":
            return [
                self._empty_anonymize(item)
                if pd.notna(item) and str(item).strip()
                else item
                for item in series_data
            ]

        results = []
        for item in series_data:
            if pd.isna(item) or not str(item).strip():
                results.append(item)
                continue

            item_str = str(item)

            if method == "hash":
                result = self.hash_anonymize(item_str)
            elif method == "location":
                result = self.address_anonymize(item_str)
            elif method == "truncate":
                result = self.truncate_anonymize(item_str)
            elif method == "round":
                result = self.round_anonymize(item, -2)
            elif method == "bucket":
                result = self.bucket_anonymize(item)
            elif method == "enc":
                result = self.aes_encrypt(item_str)
            elif method == "sm4":
                result = self.sm4_encrypt(item_str)
            elif method == "random":
                result = self.rm([item])[0]
            else:
                result = item

            results.append(result)

        return results

    def decrypt_mask(self, text: str) -> str:
        pattern = re.compile(r"8e4f1([A-Za-z0-9+/]+=*)202cf")
        matches = pattern.findall(text)

        decrypted_text = text
        for b64 in matches:
            try:
                plain_data = self.aes_decrypt(b64)
                decrypted_text = decrypted_text.replace(f"8e4f1{b64}202cf", plain_data)
            except Exception as e:
                print(f"解密失败: {e}")
                continue

        return decrypted_text

    def decrypt_sm4_mask(self, text: str) -> str:
        pattern = re.compile(r"8e4f1([A-Za-z0-9+/]+=*)202cf")
        matches = pattern.findall(text)

        decrypted_text = text
        for b64 in matches:
            try:
                plain_data = self.sm4_decrypt(b64)
                decrypted_text = decrypted_text.replace(f"8e4f1{b64}202cf", plain_data)
            except Exception as e:
                print(f"解密失败: {e}")
                continue

        return decrypted_text


crypto_manager = CryptoFileManager()
text_masking_worker = TextMasking(
    model_path=config["paths"]["model"],
    rules_file_path=config["paths"]["rules_file"],
    medicine_path=config["paths"]["medicine_file"],
    name_path=config["paths"]["name_file"],
    extra_rules_file_path=config["paths"]["extra_rules_file"],
    crypto_manager=crypto_manager,
    alg=config["alg"],
)
table_masking_worker = TextMasking(
    model_path=config["paths"]["model"],
    rules_file_path=config["paths"]["rules_file"],
    medicine_path=config["paths"]["medicine_file"],
    name_path=config["paths"]["name_file"],
    extra_rules_file_path=config["paths"]["extra_rules_file"],
    crypto_manager=crypto_manager,
    alg=config["alg"],
)

app = FastAPI(
    title="非结构化文本脱敏API",
    version="0.1.0",
)


# def process_table_masking_task(
#     file_content: bytes,
#     filename: str,
#     columns_dict: dict,
#     request_id: str,
#     file_extension: str,
# ):
#     """
#     独立的表格脱敏任务处理函数，可以被 RQ worker 调用

#     Args:
#         file_content: 文件内容（字节）
#         filename: 原始文件名
#         columns_dict: 列名和处理方式的字典
#         request_id: 请求ID
#         file_extension: 文件扩展名（.csv 或 .xlsx）

#     Returns:
#         dict: 包含处理结果的字典
#     """
#     try:
#         start_time = time.time()

#         logger.bind(type="TASK", request_id=request_id).info(
#             f"Task started - Processing file: {filename} with columns: {columns_dict}"
#         )

#         # 读取文件
#         if file_extension == ".csv":
#             supported_encodings = ["utf-8", "gbk", "gb18030"]
#             df = None
#             for encoding in supported_encodings:
#                 try:
#                     df = pd.read_csv(
#                         io.BytesIO(file_content),
#                         encoding=encoding,
#                         keep_default_na=False,
#                     )
#                     logger.bind(type="TASK", request_id=request_id).info(
#                         f"Successfully decoded CSV with encoding: {encoding}"
#                     )
#                     break
#                 except UnicodeDecodeError:
#                     continue
#             if df is None:
#                 raise ValueError("文件解码失败，尝试了所有支持的编码")
#         else:  # .xlsx
#             df = pd.read_excel(io.BytesIO(file_content), keep_default_na=False)

#         logger.bind(type="TASK", request_id=request_id).info(
#             f"File loaded - Rows: {len(df)}, Columns: {len(df.columns)}"
#         )

#         # 检测列名是否存在
#         missing_columns = [col for col in columns_dict.keys() if col not in df.columns]
#         if missing_columns:
#             raise ValueError(f"列名不存在: {missing_columns}")

#         # 逐列处理
#         processing_results = {}
#         total_texts_processed = 0
#         total_entities_found = 0
#         column_stats = {}
#         all_masking_operations = []
#         anonymizer = table_masking_worker.anonymizer

#         for idx, (column, method) in enumerate(columns_dict.items(), 1):
#             column_start_time = time.time()

#             logger.bind(type="TASK", request_id=request_id).info(
#                 f"Processing column {idx}/{len(columns_dict)}: {column} with method: {method}"
#             )

#             column_data = df[column].fillna("").astype(str).tolist()
#             non_empty_data = [text for text in column_data if text.strip()]
#             total_texts_processed += len(non_empty_data)

#             if method == "text":
#                 # 使用文本脱敏模型处理
#                 results = table_masking_worker.func_f(
#                     non_empty_data, request_id=request_id
#                 )

#                 processing_results[column] = results
#                 if "text" in results:
#                     # 将处理结果映射回原DataFrame
#                     non_empty_indices = [
#                         i for i, text in enumerate(column_data) if text.strip()
#                     ]
#                     for idx_pos, result_text in zip(non_empty_indices, results["text"]):
#                         df.at[idx_pos, column] = result_text

#                 # 收集脱敏操作信息
#                 if "masking_operations" in results and results["masking_operations"]:
#                     for op in results["masking_operations"]:
#                         op_with_context = {
#                             "column": column,
#                             "original_text": op["original_text"],
#                             "entity_type": op["entity_type"],
#                             "replaced_text": op["replaced_text"],
#                             "applied_method": op.get("applied_method", "text"),
#                             "original_text_index": op.get("original_text_index", -1),
#                         }
#                         all_masking_operations.append(op_with_context)

#                 # 统计实体数量
#                 column_entities = 0
#                 if "entities" in results:
#                     column_entities = sum(
#                         len(entities) for entities in results["entities"]
#                     )
#                     total_entities_found += column_entities

#                 column_stats[column] = {
#                     "method": "text",
#                     "texts_processed": len(non_empty_data),
#                     "entities_found": column_entities,
#                     "avg_entities_per_text": column_entities / len(non_empty_data)
#                     if non_empty_data
#                     else 0,
#                     "processing_time": time.time() - column_start_time,
#                 }
#             else:
#                 # 使用其他脱敏方法
#                 processed_data = anonymizer.apply_method(non_empty_data, method)

#                 # 将处理结果映射回原DataFrame
#                 non_empty_indices = [
#                     i for i, text in enumerate(column_data) if text.strip()
#                 ]
#                 for idx_pos, result in zip(non_empty_indices, processed_data):
#                     df.at[idx_pos, column] = result

#                 # 记录处理操作
#                 column_masking_ops = []
#                 for idx_val, (original, processed) in enumerate(
#                     zip(non_empty_data, processed_data)
#                 ):
#                     if original != processed:
#                         column_masking_ops.append(
#                             {
#                                 "column": column,
#                                 "original_text": original,
#                                 "entity_type": method.upper(),
#                                 "replaced_text": processed,
#                                 "applied_method": method,
#                                 "original_text_index": idx_val,
#                             }
#                         )

#                 all_masking_operations.extend(column_masking_ops)

#                 column_stats[column] = {
#                     "method": method,
#                     "texts_processed": len(non_empty_data),
#                     "transformations": len(column_masking_ops),
#                     "avg_transformations_per_text": len(column_masking_ops)
#                     / len(non_empty_data)
#                     if non_empty_data
#                     else 0,
#                     "processing_time": time.time() - column_start_time,
#                 }

#                 processing_results[column] = {
#                     "method": method,
#                     "processed_count": len(processed_data),
#                     "transformations_count": len(column_masking_ops),
#                 }

#             logger.bind(type="TASK", request_id=request_id).info(
#                 f"Column {column} processed in {time.time() - column_start_time:.2f}s"
#             )

#         # 保存处理后的文件
#         output_filename = f"{request_id}_{filename}"
#         output_file = TASK_RESULTS_DIR / output_filename

#         if file_extension == ".csv":
#             df.to_csv(output_file, index=False, encoding="utf-8")
#         else:  # .xlsx
#             df.to_excel(output_file, index=False, engine="openpyxl")

#         processing_time = time.time() - start_time

#         # 构建汇总结果
#         aggregation_summary = {
#             "file_name": filename,
#             "total_rows": len(df),
#             "total_columns": len(columns_dict),
#             "total_texts_processed": total_texts_processed,
#             "total_entities_found": total_entities_found,
#             "total_transformations": len(all_masking_operations),
#             "processing_time": processing_time,
#             "avg_processing_time_per_text": processing_time / total_texts_processed
#             if total_texts_processed > 0
#             else 0,
#             "column_stats": column_stats,
#         }

#         logger.bind(type="TASK", request_id=request_id).success(
#             f"Task completed - File: {filename}, Time: {processing_time:.2f}s, "
#             f"Rows: {len(df)}, Entities: {total_entities_found}"
#         )

#         # 返回结果
#         return {
#             "success": True,
#             "file_path": str(output_file),
#             "filename": output_filename,
#             "original_filename": filename,
#             "processed_columns": list(columns_dict.keys()),
#             "column_methods": columns_dict,
#             "total_rows": len(df),
#             "aggregation_results": aggregation_summary,
#             # "processing_results": processing_results,
#             # "masking_operations": all_masking_operations,
#             "request_id": request_id,
#         }

#     except Exception as e:
#         logger.bind(type="TASK", request_id=request_id).error(
#             f"Task failed - Error: {str(e)}"
#         )
#         raise


def process_table_masking_task(
    file_content: bytes,
    filename: str,
    columns_dict: dict,
    request_id: str,
    file_extension: str,
):
    """
    Polars 优化的版本

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

        # 获取 anonymizer 实例
        anonymizer = table_masking_worker.anonymizer

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


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求添加唯一ID"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # 使用loguru的contextualize功能将request_id传递给所有日志
    with logger.contextualize(request_id=request_id):
        response = await call_next(request)
        return response


@app.get("/")
async def root():
    return {"message": "非结构化文本脱敏API"}


@app.post("/text_masking", summary="文本脱敏")
async def text_masking(
    request: Request,
    text: Union[str, List[str]] = Form(..., description="需要脱敏的文本或文本列表"),
    run_name: str = Form("default_text", description="本次运行的名称"),
):
    """
    对输入的单个字符串或字符串列表进行脱敏处理。

    - **text**: 可以是单个字符串，也可以是多个字符串组成的列表。

    **调用示例:**
    ```python
    import requests

    response = requests.post("http://127.0.0.1:18819/text_masking", data={'text': ['李四的地址是北京市朝阳区', '王五的邮箱是wangwu@example.com']})
    print(response.json())
    ```
    ```shell
    curl -X POST "http://127.0.0.1:18819/text_masking" -F "text=李四的地址是北京市朝阳区" -F "text=王五的邮箱是wangwu@example.com"
    ```
    """

    current_date = datetime.now().date()
    trial_end_date = date(2026, 1, 31)
    if current_date > trial_end_date:
        logger.warning("试用期已过，退出程序。")
        sys.exit(1)

    request_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    request_id = request.state.request_id

    if not text:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Invalid input: empty text"
        )
        raise HTTPException(status_code=400, detail="输入文本不能为空。")

    try:
        text_list = [text] if isinstance(text, str) else text
        if not isinstance(text_list, list) or not all(
            isinstance(item, str) for item in text_list
        ):
            logger.bind(type="SERVICE", request_id=request_id).error(
                f"Request from {client_ip} - Invalid input format"
            )
            raise HTTPException(
                status_code=422, detail="输入格式不正确，请传入字符串或字符串列表。"
            )

        logger.bind(type="SERVICE", request_id=request_id).info(
            f"Request from {client_ip} - Processing {len(text_list)} texts for masking"
        )
        results = table_masking_worker.func_f(text_list, request_id=request_id)
        logger.bind(type="SERVICE", request_id=request_id).success(
            f"Request from {client_ip} - Successfully processed {len(text_list)} texts"
        )
        results["original_text"] = text_list
        return results
    except HTTPException as he:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - HTTP Exception: {he.detail}"
        )
        raise
    except Exception as e:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Internal error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/text_decryption", summary="文本解密")
async def text_decryption(
    request: Request,
    text: Union[str, List[str]] = Form(..., description="需要解密的文本或文本列表"),
    run_name: str = Form("default_decrypt", description="本次运行的名称"),
):
    """
    对输入的单个字符串或字符串列表进行解密处理。

    - **text**: 可以是单个字符串，也可以是多个字符串组成的列表。

    **调用示例:**
    ```python
    import requests

    response = requests.post("http://127.0.0.1:18819/text_decryption", data={'text': ['8e4f1c2VsdCBkYXRh202cf', 'another encrypted text']})
    print(response.json())
    ```
    ```shell
    curl -X POST "http://127.0.0.1:18819/text_decryption" -F "text=8e4f1c2VzdCBkYXRh202cf" -F "text=another encrypted text"
    ```
    """

    current_date = datetime.now().date()
    trial_end_date = date(2026, 1, 31)
    if current_date > trial_end_date:
        logger.warning("试用期已过，退出程序。")
        sys.exit(1)

    request_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    request_id = request.state.request_id

    if not text:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Invalid input: empty text"
        )
        raise HTTPException(status_code=400, detail="输入文本不能为空。")

    try:
        text_list = [text] if isinstance(text, str) else text
        if not isinstance(text_list, list) or not all(
            isinstance(item, str) for item in text_list
        ):
            logger.bind(type="SERVICE", request_id=request_id).error(
                f"Request from {client_ip} - Invalid input format"
            )
            raise HTTPException(
                status_code=422, detail="输入格式不正确，请传入字符串或字符串列表。"
            )

        logger.bind(type="SERVICE", request_id=request_id).info(
            f"Request from {client_ip} - Processing {len(text_list)} texts for decryption"
        )

        # 使用 Anonymizer 的 decrypt_mask 方法进行解密
        anonymizer = text_masking_worker.anonymizer
        decrypted_texts = []

        for input_text in text_list:
            try:
                decrypted_text = anonymizer.decrypt_mask(input_text)
                decrypted_texts.append(decrypted_text)
                logger.bind(type="SERVICE", request_id=request_id).info(
                    f"Successfully decrypted text of length {len(input_text)} to {len(decrypted_text)}"
                )
            except Exception as e:
                logger.bind(type="SERVICE", request_id=request_id).error(
                    f"Failed to decrypt text: {e}"
                )
                # 解密失败时返回原文
                decrypted_texts.append(input_text)

        logger.bind(type="SERVICE", request_id=request_id).success(
            f"Request from {client_ip} - Successfully processed {len(text_list)} texts for decryption"
        )

        return {
            "decrypted_text": decrypted_texts,
            "original_text": text_list,
            "processed_count": len(text_list),
        }

    except HTTPException as he:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - HTTP Exception: {he.detail}"
        )
        raise
    except Exception as e:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Internal error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/text_apply", summary="文本方法处理")
async def text_apply(
    request: Request,
    text: Union[str, List[str]] = Form(..., description="需要处理的文本或文本列表"),
    method: str = Form(
        ..., description="处理方法: 'enc'(加密), 'null'(置空), 'location'(地址脱敏)等"
    ),
    run_name: str = Form("default_apply", description="本次运行的名称"),
):
    """
    对输入的单个字符串或字符串列表使用指定方法进行直接处理。

    - **text**: 可以是单个字符串，也可以是多个字符串组成的列表。
    - **method**: 处理方法，支持以下选项：
      - "enc": AES加密
      - "null": 置空
      - "location": 地址脱敏
      - "hash": SHA-256哈希
      - "truncate": 截断
      - "round": 数值 rounding
      - "bucket": 分桶
      - "sm4": SM4加密

    **调用示例:**
    ```python
    import requests

    response = requests.post(
        "http://127.0.0.1:18819/text_apply",
        data={
            'text': ['张三的电话是13812345678', '李四的地址是北京市朝阳区'],
            'method': 'enc'
        }
    )
    print(response.json())
    ```
    ```shell
    curl -X POST "http://127.0.0.1:18819/text_apply" \
        -F "text=张三的电话是13812345678" \
        -F "text=李四的地址是北京市朝阳区" \
        -F "method=enc"
    ```
    """

    current_date = datetime.now().date()
    trial_end_date = date(2026, 1, 31)
    if current_date > trial_end_date:
        logger.warning("试用期已过，退出程序。")
        sys.exit(1)

    request_time = datetime.now()
    client_ip = request.client.host if request.client else "unknown"
    request_id = request.state.request_id

    # 验证处理方法
    valid_methods = [
        "enc",
        "null",
        "location",
        "hash",
        "truncate",
        "round",
        "bucket",
        "sm4",
    ]
    if method not in valid_methods:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Invalid processing method: {method}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"处理方法 '{method}' 无效，支持的方法: {', '.join(valid_methods)}",
        )

    if not text:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Invalid input: empty text"
        )
        raise HTTPException(status_code=400, detail="输入文本不能为空。")

    try:
        text_list = [text] if isinstance(text, str) else text
        if not isinstance(text_list, list) or not all(
            isinstance(item, str) for item in text_list
        ):
            logger.bind(type="SERVICE", request_id=request_id).error(
                f"Request from {client_ip} - Invalid input format"
            )
            raise HTTPException(
                status_code=422, detail="输入格式不正确，请传入字符串或字符串列表。"
            )

        logger.bind(type="SERVICE", request_id=request_id).info(
            f"Request from {client_ip} - Processing {len(text_list)} texts with method: {method}"
        )

        start_time = time.time()
        anonymizer = text_masking_worker.anonymizer

        # 使用指定方法处理文本
        processed_texts = []
        masking_operations = []

        for idx, input_text in enumerate(text_list):
            original_text = input_text
            processed_text = anonymizer.apply_method(original_text, method)
            processed_texts.append(processed_text)

            # 记录处理操作
            if original_text != processed_text:
                masking_operations.append(
                    {
                        "original_text": original_text,
                        "processed_text": processed_text,
                        "replaced_text": processed_text,  # 保持一致性，使用replaced_text
                        "entity_type": method.upper(),  # 标识实体类型
                        "applied_method": method,  # 添加应用的方法
                        "method": method.upper(),  # 保持向后兼容
                        "text_index": idx,
                        "original_length": len(original_text),
                        "processed_length": len(processed_text),
                    }
                )

        processing_time = time.time() - start_time

        # 记录处理结果日志
        logger.bind(type="PROCESSING", request_id=request_id).info(
            f"Text processing completed - Method: {method}, Texts processed: {len(text_list)}, "
            f"Transformations: {len(masking_operations)}, Processing time: {processing_time:.4f}s"
        )

        # 记录详细的处理操作
        if masking_operations:
            logger.bind(type="PROCESSING", request_id=request_id).info(
                f"Processing operations: {json.dumps(masking_operations, ensure_ascii=False)}"
            )

        logger.bind(type="SERVICE", request_id=request_id).success(
            f"Request from {client_ip} - Successfully processed {len(text_list)} texts with method: {method}"
        )

        return {
            "success": True,
            "processed_text": processed_texts,
            "original_text": text_list,
            "method": method,
            "processing_stats": {
                "total_texts": len(text_list),
                "transformations": len(masking_operations),
                "processing_time": processing_time,
                "avg_processing_time_per_text": processing_time / len(text_list)
                if text_list
                else 0,
            },
            "operations": masking_operations,
        }

    except HTTPException as he:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - HTTP Exception: {he.detail}"
        )
        raise
    except Exception as e:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Internal error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/table_masking", summary="表格脱敏")
async def submit_table_masking(
    request: Request,
    file: UploadFile = File(..., description="包含待脱敏数据的 CSV 或 XLSX 文件"),
    columns: str = Form(
        ...,
        description="待处理的列名及处理方式，JSON格式的字典，key为列名，value为处理方式('text', 'enc', 'null', 'location'等)",
    ),
    run_name: str = Form("default_table", description="本次运行的名称"),
):
    """
    提交表格脱敏任务到队列，立即返回任务ID。

    - **file**: 上传的文件对象，csv或xlsx，尽量UTF-8编码。
    - **columns**: 待处理的列名及处理方式，JSON格式的字典。
      - "text": 使用文本脱敏模型处理
      - "enc": 使用AES加密
      - "null": 置空
      - "location": 地址脱敏
      - 其他: 支持Anonymizer.apply_method中的其他方法

    返回任务ID，可通过 /table_masking/status/{task_id} 查询状态，
    通过 /table_masking/download/{task_id} 下载结果。

    **调用示例:**
    ```python
    import requests

    with open("test.csv", "rb") as f:
        response = requests.post(
            "http://127.0.0.1:18819/table_masking",
            files={"file": f},
            data={"columns": '{"病历": "text", "手机号": "enc", "地址": "location"}'},
        )
    task_id = response.json()["task_id"]
    print(f"任务已提交: {task_id}")
    ```
    """

    current_date = datetime.now().date()
    trial_end_date = date(2026, 1, 31)
    if current_date > trial_end_date:
        logger.warning("试用期已过，退出程序。")
        sys.exit(1)

    client_ip = request.client.host if request.client else "unknown"
    request_id = request.state.request_id

    # 验证文件
    if not file.filename:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - No filename provided"
        )
        raise HTTPException(status_code=400, detail="未提供文件名。")

    allowed_extensions = {".csv", ".xlsx"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Invalid file type: {file_extension}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: '{file_extension}'。请上传 {', '.join(allowed_extensions)} 文件。",
        )

    try:
        # 解析列名参数
        try:
            columns_dict: dict = json.loads(columns)
            if not isinstance(columns_dict, dict):
                raise ValueError("columns参数必须是字典格式")
        except (json.JSONDecodeError, ValueError) as e:
            logger.bind(type="SERVICE", request_id=request_id).error(
                f"Request from {client_ip} - Invalid columns format: {e}"
            )
            raise HTTPException(status_code=400, detail=f"列名参数格式错误: {e}")

        # 验证处理方法
        valid_methods = [
            "text",
            "enc",
            "null",
            "location",
            "hash",
            "truncate",
            "round",
            "bucket",
            "sm4",
        ]
        for col, method in columns_dict.items():
            if method not in valid_methods:
                logger.bind(type="SERVICE", request_id=request_id).error(
                    f"Request from {client_ip} - Invalid processing method: {method} for column: {col}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"列 '{col}' 的处理方式 '{method}' 无效，支持的方法: {', '.join(valid_methods)}",
                )

        # 读取文件内容
        file_content = await file.read()

        if len(file_content) == 0:
            logger.bind(type="SERVICE", request_id=request_id).error(
                f"Request from {client_ip} - Empty file"
            )
            raise HTTPException(status_code=400, detail="文件内容为空。")

        logger.bind(type="SERVICE", request_id=request_id).info(
            f"Request from {client_ip} - Submitting task for file: {file.filename}, "
            f"size: {len(file_content)} bytes, columns: {list(columns_dict.keys())}"
        )

        # 提交任务到 RQ 队列
        job = task_queue.enqueue(
            process_table_masking_task,
            file_content=file_content,
            filename=file.filename,
            columns_dict=columns_dict,
            request_id=request_id,
            file_extension=file_extension,
            job_timeout="30m",  # 30分钟超时
            result_ttl=86400,  # 结果保留24小时
            failure_ttl=86400,  # 失败信息保留24小时
        )

        logger.bind(type="SERVICE", request_id=request_id).success(
            f"Request from {client_ip} - Task submitted successfully, task_id: {job.id}"
        )

        return {
            "success": True,
            "task_id": job.id,
            "message": "任务已提交到队列",
            "filename": file.filename,
            "columns": list(columns_dict.keys()),
            "request_id": request_id,
            "estimated_time": "处理时间取决于文件大小和列数",
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.bind(type="SERVICE", request_id=request_id).error(
            f"Request from {client_ip} - Failed to submit task: {e}"
        )
        raise HTTPException(status_code=500, detail=f"提交任务失败: {e}")


# 完整的任务状态查询接口
@app.get("/table_masking/status/{task_id}", summary="查询任务状态")
async def get_task_status(
    task_id: str,
    request: Request = None,
):
    """
    查询表格脱敏任务的处理状态。

    - **task_id**: 任务ID（提交任务时返回）

    返回信息：
    - status: 任务状态
      - "queued": 排队中
      - "started": 处理中
      - "finished": 已完成
      - "failed": 失败
      - "deferred": 延迟
      - "scheduled": 已调度
    - progress: 处理进度（如果任务支持）
    - result: 处理结果（仅当任务完成时）
    - error: 错误信息（仅当任务失败时）

    **调用示例:**
    ```python
    import requests
    import time

    task_id = "your-task-id"

    while True:
        response = requests.get(f"http://127.0.0.1:18819/table_masking/status/{task_id}")
        data = response.json()
        print(f"状态: {data['status']}")

        if data['status'] == 'finished':
            print("任务完成！")
            print(f"处理结果: {data['result']}")
            break
        elif data['status'] == 'failed':
            print(f"任务失败: {data['error']}")
            break

        time.sleep(2)  # 每2秒查询一次
    ```
    """
    client_ip = request.client.host if request and request.client else "unknown"

    try:
        # 从队列中获取任务
        job = task_queue.fetch_job(task_id)

        if not job:
            logger.bind(type="SERVICE").warning(
                f"Request from {client_ip} - Task not found: {task_id}"
            )
            # list all jobs
            jobs = task_queue.get_jobs()
            logger.bind(type="SERVICE").warning(
                f"Request from {client_ip} - All jobs: {jobs}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"任务不存在: {task_id}。任务可能已过期或ID错误。",
            )

        # 获取任务状态
        status = job.get_status()

        # 构建响应
        response_data = {
            "task_id": task_id,
            "status": status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }

        # 添加进度信息（如果有）
        if hasattr(job, "meta") and job.meta:
            response_data["progress"] = job.meta.get("progress", 0)
            response_data["current_step"] = job.meta.get("current_step", "")

        # 如果任务完成，添加结果
        if job.is_finished:
            result = job.result
            response_data["result"] = {
                "success": result.get("success", False),
                "filename": result.get("filename"),
                "original_filename": result.get("original_filename"),
                "total_rows": result.get("total_rows"),
                "processed_columns": result.get("processed_columns"),
                "aggregation_summary": result.get("aggregation_results"),
            }
            response_data["message"] = "任务已完成，可以下载结果文件"

            logger.bind(type="SERVICE").info(
                f"Request from {client_ip} - Task {task_id} status: finished"
            )

        # 如果任务失败，添加错误信息
        elif job.is_failed:
            error_info = str(job.exc_info) if job.exc_info else "未知错误"
            response_data["error"] = error_info
            response_data["message"] = "任务处理失败"

            logger.bind(type="SERVICE").error(
                f"Request from {client_ip} - Task {task_id} status: failed, error: {error_info}"
            )

        # 如果任务正在处理
        elif job.is_started:
            response_data["message"] = "任务正在处理中"

            logger.bind(type="SERVICE").info(
                f"Request from {client_ip} - Task {task_id} status: started"
            )

        # 如果任务在队列中
        elif job.is_queued:
            # 获取队列位置
            position = job.get_position()
            response_data["queue_position"] = position
            response_data["message"] = (
                f"任务在队列中，当前位置: {position if position is not None else '未知'}"
            )

            logger.bind(type="SERVICE").info(
                f"Request from {client_ip} - Task {task_id} status: queued, position: {position}"
            )

        else:
            response_data["message"] = f"任务状态: {status}"

        return response_data

    except HTTPException as he:
        raise
    except Exception as e:
        logger.bind(type="SERVICE").error(
            f"Request from {client_ip} - Error fetching task status: {e}"
        )
        raise HTTPException(status_code=500, detail=f"查询任务状态失败: {e}")


# 文件下载接口（下载后删除文件）
@app.get("/table_masking/download/{task_id}", summary="下载处理结果文件")
async def download_result(
    task_id: str,
    request: Request = None,
):
    """
    下载表格脱敏任务的处理结果文件。

    - **task_id**: 任务ID（提交任务时返回）

    注意：
    - 只有任务状态为 "finished" 时才能下载
    - 文件下载后会自动删除，请妥善保存
    - 如果需要重新下载，需要重新提交任务

    **调用示例:**
    ```python
    import requests

    task_id = "your-task-id"

    response = requests.get(
        f"http://127.0.0.1:18819/table_masking/download/{task_id}",
        stream=True
    )

    if response.status_code == 200:
        with open("output.csv", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("文件下载成功！")
    else:
        print(f"下载失败: {response.json()}")
    ```
    """
    client_ip = request.client.host if request and request.client else "unknown"

    try:
        # 从队列中获取任务
        job = task_queue.fetch_job(task_id)

        if not job:
            logger.bind(type="SERVICE").warning(
                f"Request from {client_ip} - Task not found for download: {task_id}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"任务不存在: {task_id}。任务可能已过期或ID错误。",
            )

        # 检查任务是否完成
        if not job.is_finished:
            status = job.get_status()
            logger.bind(type="SERVICE").warning(
                f"Request from {client_ip} - Task {task_id} not finished, status: {status}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"任务尚未完成，当前状态: {status}。请稍后再试。",
            )

        # 获取结果
        result = job.result

        if not result or not result.get("success"):
            logger.bind(type="SERVICE").error(
                f"Request from {client_ip} - Task {task_id} result invalid"
            )
            raise HTTPException(status_code=500, detail="任务结果无效或处理失败")

        file_path = result.get("file_path")

        if not file_path:
            logger.bind(type="SERVICE").error(
                f"Request from {client_ip} - Task {task_id} missing file_path"
            )
            raise HTTPException(status_code=500, detail="结果文件路径不存在")

        file_path = Path(file_path)

        if not file_path.exists():
            logger.bind(type="SERVICE").error(
                f"Request from {client_ip} - File not found: {file_path}"
            )
            raise HTTPException(status_code=404, detail="结果文件不存在，可能已被删除")

        # 获取原始文件名
        original_filename = result.get("original_filename", "output.csv")

        logger.bind(type="SERVICE").info(
            f"Request from {client_ip} - Downloading file for task {task_id}: {file_path}"
        )

        # 创建文件响应
        response = FileResponse(
            path=str(file_path),
            filename=original_filename,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{original_filename}"',
                "X-Task-ID": task_id,
            },
        )

        # 添加后台任务：下载完成后删除文件
        @app.on_event("startup")
        async def startup_event():
            pass

        # 使用 background_tasks 在响应发送后删除文件

        def cleanup_file(file_path: Path, task_id: str):
            """清理文件的后台任务"""
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.bind(type="SERVICE").info(
                        f"File deleted after download: {file_path} for task {task_id}"
                    )
            except Exception as e:
                logger.bind(type="SERVICE").error(
                    f"Failed to delete file {file_path} for task {task_id}: {e}"
                )

        # 注意：由于FileResponse的限制，我们需要在这里手动处理文件删除
        # 使用response的background参数
        from starlette.background import BackgroundTask

        response.background = BackgroundTask(cleanup_file, file_path, task_id)

        logger.bind(type="SERVICE").success(
            f"Request from {client_ip} - File download started for task {task_id}"
        )

        return response

    except HTTPException as he:
        raise
    except Exception as e:
        logger.bind(type="SERVICE").error(
            f"Request from {client_ip} - Error downloading file: {e}"
        )
        raise HTTPException(status_code=500, detail=f"下载文件失败: {e}")


if __name__ == "__main__":
    # 超时检测
    current_date = datetime.now().date()
    trial_end_date = date(2026, 1, 16)
    if current_date > trial_end_date:
        logger.warning("试用期已过，退出程序。")
        sys.exit(1)

    # Uvicorn
    from typing import Any

    import uvicorn

    # 解决uvicorn创建子进程超时无限循环的问题
    original_uvicorn_is_alive = uvicorn.supervisors.multiprocess.Process.is_alive

    def patched_is_alive(self: Any) -> bool:
        timeout = 90
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

    # gunicorn在python内部的配置问题比较大，暂时不使用
