"""
将融合后的三元组写入Neo4j图数据库
功能：
1. 读取三元组JSONL文件
2. 连接Neo4j图数据库
3. 创建唯一性约束（防止重复节点）
4. 将三元组转换为图结构存储
   - 每个实体（head/tail）作为一个节点
   - 每个关系作为一条边
5. 支持导入后查询和可视化
"""

import argparse

from dotenv import load_dotenv
from neo4j import GraphDatabase

from common import load_jsonl, safe_env


def run_load(triples_path: str) -> None:
    """
    将三元组数据导入Neo4j数据库。
    流程：
    1. 从环境变量读取Neo4j连接配置
    2. 建立数据库连接
    3. 创建唯一性约束（确保实体不重复）
    4. 遍历每条三元组，创建节点和关系
    5. 关闭连接
    
    Args:
        triples_path: 三元组JSONL文件路径
    """

    # Step 1: 获取Neo4j连接配置
    uri = safe_env("NEO4J_URI")        # Neo4j服务器地址
    user = safe_env("NEO4J_USER")      # 用户名
    password = safe_env("NEO4J_PASSWORD")  # 密码
    
    # 验证配置完整性
    if not uri or not user or not password:
        raise ValueError("Neo4j 环境变量缺失，请检查 .env 中 NEO4J_URI/USER/PASSWORD")
    
    # Step 2: 加载三元组数据
    triples = load_jsonl(triples_path)
    if not triples:
        raise ValueError(f"没有可导入的数据: {triples_path}")
    
    print(f"准备导入 {len(triples)} 条三元组...")
    
    # Step 3: 连接Neo4j数据库
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Step 4: 创建唯一性约束
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        print("已确保唯一性约束存在")
        
        # Step 5: 逐条导入三元组
        count = 0
        for t in triples:
            # 清理关系类型：移除反引号，将空格替换为下划线
            rel = t["relation"].replace("`", "").replace(" ", "_")
            
            # Step 5a: 构建Cypher查询语句
            # MERGE: 如果节点/关系不存在则创建，存在则匹配
            # (h:Entity {name: $head}) - 头实体节点，标签为Entity
            # (ta:Entity {name: $tail}) - 尾实体节点
            # [r:`{rel}`] - 关系，关系类型由变量指定
            cypher = f"""
            MERGE (h:Entity {{name: $head}})
            MERGE (ta:Entity {{name: $tail}})
            MERGE (h)-[r:`{rel}`]->(ta)
            ON CREATE SET r.source = $source, r.method = $method
            """
            
            # Step 5b: 执行Cypher查询
            session.run(
                cypher,
                head=t["head"],
                tail=t["tail"],
                source=t.get("source", ""),  
                method=t.get("method", ""),  
            )
            count += 1
            
            if count % 100 == 0:
                print(f"已导入 {count}/{len(triples)} 条三元组...")
        
        print(f"成功导入 {count} 条三元组")
    
    # Step 6: 关闭数据库连接
    driver.close()
    print(f"[DONE] Neo4j 导入完成: {len(triples)} triples")


def main() -> None:
    """
    使用示例：
    # 使用默认路径
    python import_to_neo4j.py
    # 指定三元组文件路径
    python import_to_neo4j.py --triples ./output/triples_fused.jsonl
    """

    parser = argparse.ArgumentParser(description="将融合后的三元组导入Neo4j图数据库")
    parser.add_argument(
        "--triples",
        required=False,
        default="data/triples_fused/triples_fused.jsonl",
        help="融合后三元组jsonl文件路径（默认: data/triples_fused/triples_fused.jsonl）",
    )
    
    args = parser.parse_args()
    load_dotenv()
    run_load(args.triples)


if __name__ == "__main__":
    main()