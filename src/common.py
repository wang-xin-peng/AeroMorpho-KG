"""
项目的通用工具函数
功能：
1. 文件路径操作：创建父目录、查找PDF文件
2. JSONL格式读写：JSON Lines格式，每行一个JSON对象
3. 环境变量安全读取：带默认值的环境变量获取
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_parent(path: str) -> None:
    """
    确保文件所在目录存在，如果不存在则递归创建。
    Args:
        path: 文件路径
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def list_pdf_files(pdf_dir: str) -> List[Path]:
    """
    列出指定目录下的所有PDF文件，按文件名排序
    Args:
        pdf_dir: PDF文件所在目录路径
    Returns:
        排序后的PDF文件路径列表
    """

    return sorted(Path(pdf_dir).glob("*.pdf"))


def load_jsonl(path: str) -> List[Dict]:
    """
    加载JSONL格式文件，返回字典列表
    Args:
        path: JSONL文件路径
    Returns:
        字典列表，每行解析为一个字典。
        如果文件不存在，返回空列表。
    """

    items: List[Dict] = []
    if not Path(path).exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  
            if not line:          
                continue
            items.append(json.loads(line))
    
    return items


def dump_jsonl(path: str, rows: Iterable[Dict]) -> None:
    """
    将字典列表保存为JSONL格式文件。
    Args:
        path: 输出文件路径
        rows: 字典的可迭代对象
    """

    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # ensure_ascii=False 确保中文不被转义为 \uXXXX 格式
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_env(name: str, default: str = "") -> str:
    """
    安全地读取环境变量，支持默认值和自动去除首尾空白
    Args:
        name: 环境变量名称
        default: 默认值，当环境变量不存在时返回
    Returns:
        环境变量的值（已去除首尾空白），或默认值
    """
    
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default