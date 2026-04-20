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
    DeepKE Schema数据结构,描述知识抽取任务的领域知识，包括：
    1.instruction: 系统指令，告诉模型要做什么
    2.relation_types: 关系类型列表（支持 name + description + example）
    """
    instruction: str
    relation_types: List[Dict]   # 关系类型列表，每个包含 name, description, example

    @classmethod
    def from_json(cls, path: str) -> "DeepKESchema":
        """
        从JSON文件加载Schema配置。
        支持两种格式：
        1. 简单格式：{"instruction": "...", "schema": ["关系1", "关系2"]}
        2. 完整格式：{"instruction": "...", "relation_types": [{"name": "关系1", "description": "..."}]}
        
        Args:
            path: JSON配置文件路径
        Returns:
            DeepKESchema实例
        """

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        
        instruction = payload.get("instruction", "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。")
        
        # 兼容两种格式
        if "relation_types" in payload:
            # 完整格式
            relation_types = payload["relation_types"]
        elif "schema" in payload:
            # 简单格式，转换为完整格式
            schema = payload["schema"]
            if isinstance(schema, list) and len(schema) > 0 and isinstance(schema[0], str):
                # 纯字符串列表
                relation_types = [{"name": r, "description": "", "example": []} for r in schema]
            else:
                relation_types = schema
        else:
            relation_types = []
        
        return cls(
            instruction=instruction,
            relation_types=relation_types,
        )
    
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
