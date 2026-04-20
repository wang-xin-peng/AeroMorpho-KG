"""
加载并格式化DeepKE使用的领域Schema配置
Schema的作用：
1. 定义知识图谱中需要抽取的关系类型
2. 将Schema格式化为OneKE模型可理解的提示词指令
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class DeepKESchema:
    """
    DeepKE Schema数据结构,描述知识抽取任务的领域知识
    只包含 relation_types（关系类型列表）
    instruction 在 extract_triples.py 中硬编码
    """
    relation_types: List[Dict]   # 关系类型列表，每个包含 name, description

    @classmethod
    def from_json(cls, path: str) -> "DeepKESchema":
        """
        从JSON文件加载关系类型配置
        文件格式：直接是关系类型数组
        [
          {"name": "具有", "description": "..."},
          {"name": "优化", "description": "..."},
          ...
        ]
        
        Args:
            path: JSON配置文件路径
        Returns:
            DeepKESchema实例
        """

        with open(path, "r", encoding="utf-8") as f:
            relation_types = json.load(f)
        
        # 确保是列表格式
        if not isinstance(relation_types, list):
            raise ValueError(f"relation_types.json应该是数组格式，但得到了: {type(relation_types)}")
        
        return cls(relation_types=relation_types)
    
    def get_relation_names(self) -> List[str]:
        """
        获取所有关系名称列表
        Returns:
            关系名称列表
        """
        return [r.get("name", "") for r in self.relation_types]
    
    def get_relation_schema_dict(self, relation_names: List[str]) -> Union[List[str], Dict[str, str]]:
        """
        根据关系名称列表，生成 OneKE schema 格式
        如果所有关系都没有 description，返回字符串列表（基础模式）
        如果有 description，返回字典（增强模式）
        
        Args:
            relation_names: 需要抽取的关系名称列表
        Returns:
            OneKE schema 格式（列表或字典）
        """
        # 构建关系名到完整信息的映射
        relation_map = {r.get("name", ""): r for r in self.relation_types}
        
        # 检查是否所有关系都有 description
        has_description = any(
            relation_map.get(name, {}).get("description", "").strip()
            for name in relation_names
        )
        
        if not has_description:
            # 基础模式：只返回关系名列表
            return relation_names
        else:
            # 增强模式：返回 {关系名: 描述} 字典
            schema_dict = {}
            for name in relation_names:
                rel_info = relation_map.get(name, {})
                desc = rel_info.get("description", "").strip()
                schema_dict[name] = desc if desc else name
            return schema_dict
