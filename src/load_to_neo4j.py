import argparse

from dotenv import load_dotenv
from neo4j import GraphDatabase

from common import load_jsonl, safe_env


def run_load(triples_path: str) -> None:
    uri = safe_env("NEO4J_URI")
    user = safe_env("NEO4J_USER")
    password = safe_env("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise ValueError("Neo4j 环境变量缺失，请检查 .env 中 NEO4J_URI/USER/PASSWORD")

    triples = load_jsonl(triples_path)
    if not triples:
        raise ValueError(f"没有可导入的数据: {triples_path}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        for t in triples:
            rel = t["relation"].replace("`", "").replace(" ", "_")
            cypher = f"""
            MERGE (h:Entity {{name: $head}})
            MERGE (ta:Entity {{name: $tail}})
            MERGE (h)-[r:`{rel}`]->(ta)
            ON CREATE SET r.source = $source, r.method = $method
            """
            session.run(
                cypher,
                head=t["head"],
                tail=t["tail"],
                source=t.get("source", ""),
                method=t.get("method", ""),
            )
    driver.close()
    print(f"[DONE] Neo4j 导入完成: {len(triples)} triples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True, help="融合后三元组 jsonl")
    args = parser.parse_args()
    load_dotenv()
    run_load(args.triples)


if __name__ == "__main__":
    main()
