from dotenv import load_dotenv

from src.parse_docs import run_parse


if __name__ == "__main__":
    load_dotenv()
    # 示例：批量解析 `变构飞行器` 目录下 PDF 到 `data/parsed`
    import asyncio

    asyncio.run(run_parse(pdf_dir="变构飞行器", out_dir="data/parsed", skip_exists=True))