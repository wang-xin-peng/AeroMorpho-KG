"""
加载并格式化DeepKE使用的领域Schema配置
Schema的作用：
1. 定义知识图谱中需要抽取的实体类型
2. 定义需要抽取的关系类型
3. 将Schema格式化为OneKE模型可理解的提示词指令块
"""

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DeepKESchema:
    """
    DeepKE Schema数据结构,描述知识抽取任务的领域知识，包括：
    1.instruction: 系统指令，告诉模型要做什么
    2.entity_types: 实体类型列表，每个类型包含名称和描述
    3.relation_types: 关系类型列表，每个类型包含名称和描述
    """
    instruction: str
    entity_types: List[Dict]   # 实体类型定义列表
    relation_types: List[Dict] # 关系类型定义列表

    @classmethod
    def from_json(cls, path: str) -> "DeepKESchema":
        """
        从JSON文件加载Schema配置。
        Args:
            path: JSON配置文件路径
        Returns:
            DeepKESchema实例
        """

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        
        return cls(
            instruction=payload.get("instruction", ""),
            entity_types=payload.get("entity_types", []),
            relation_types=payload.get("relation_types", []),
        )

    def to_instruction_block(self) -> str:
        """
        将Schema格式化为OneKE模型可理解的指令块
        输出格式示例：
            你是专门进行关系抽取的专家...
            
            [实体类型]
            - 变形方式: 飞行器改变外形的方法
            - 气动特性: 与空气动力学相关的性能参数
            
            [关系类型]
            - 具有: 实体拥有某个属性或部件
            - 优化: 方法对目标产生改善效果
            
            输出格式要求：仅输出 JSON 数组，不要额外解释。
        
        Returns:
            格式化后的指令字符串
        """
        
        # 构建实体类型的描述行
        entity_lines = [
            f"- {x.get('name', '').strip()}: {x.get('description', '').strip()}"
            for x in self.entity_types
        ]
        
        # 构建关系类型的描述行
        relation_lines = [
            f"- {x.get('name', '').strip()}: {x.get('description', '').strip()}"
            for x in self.relation_types
        ]
        
        # 组装完整的指令块
        return (
            f"{self.instruction}\n\n"
            f"[实体类型]\n{chr(10).join(entity_lines)}\n\n"  # chr(10) 即换行符\n
            f"[关系类型]\n{chr(10).join(relation_lines)}\n\n"
            "输出格式要求：仅输出 JSON 数组，不要额外解释。"
        )