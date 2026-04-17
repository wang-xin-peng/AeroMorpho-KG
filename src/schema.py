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
        按照OneKE的KGzh指令格式，支持example示例：
        1. 使用OneKE标准指令
        2. 包含详细的schema描述
        3. 融入example示例提升抽取效果
        
        Returns:
            格式化后的指令字符串
        """
        
        # 构建实体类型的描述行（包含example提示）
        entity_lines = []
        for x in self.entity_types:
            name = x.get('name', '').strip()
            desc = x.get('description', '').strip()
            examples = x.get('example', [])
            
            # 基本描述
            line = f"- {name}: {desc}"
            
            # 如果有example，添加示例提示
            if examples and len(examples) >= 2:
                pos_ex = examples[0].get('output', {})
                neg_ex = examples[1].get('output', {})
                pos_vals = pos_ex.get(name, [])
                
                if pos_vals:
                    line += f" [正例: {', '.join(pos_vals[:2])}]"
            
            entity_lines.append(line)
        
        # 构建关系类型的描述行（包含example提示）
        relation_lines = []
        for x in self.relation_types:
            name = x.get('name', '').strip()
            desc = x.get('description', '').strip()
            examples = x.get('example', [])
            
            # 基本描述
            line = f"- {name}: {desc}"
            
            # 如果有example，添加示例提示
            if examples and len(examples) >= 1:
                pos_ex = examples[0].get('output', [])
                if pos_ex and len(pos_ex) > 0:
                    # 提取第一个三元组示例
                    first_triple = pos_ex[0]
                    if isinstance(first_triple, dict):
                        h = first_triple.get('head', '')
                        t = first_triple.get('tail', '')
                        if h and t:
                            line += f" [例: {h}-{name}->{t}]"
            
            relation_lines.append(line)
        
        # 组装完整的指令块，使用OneKE标准指令
        return (
            f"你是一个图谱实体知识结构化专家。根据输入实体类型(entity type)的schema描述，从文本中抽取出相应的实体实例和其属性信息，不存在的属性不输出, 属性存在多值就返回列表，并输出为可解析的json格式。\n\n"
            f"[实体类型]\n{chr(10).join(entity_lines)}\n\n"  # chr(10) 即换行符\n
            f"[关系类型]\n{chr(10).join(relation_lines)}\n\n"
            "请按照JSON字符串的格式回答。"
        )