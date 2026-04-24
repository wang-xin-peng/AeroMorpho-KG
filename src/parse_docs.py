"""
用LlamaCloud将PDF解析为Markdown
功能：
1. 使用LlamaCloud云端API进行解析
2. 批量处理目录中的所有PDF文件
3. 每个PDF输出一个.md文件
"""

import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from pypdf import PdfReader
from tqdm import tqdm
from common import list_pdf_files, safe_env

# 加载LLAMA_CLOUD_API_KEY
load_dotenv()


async def parse_one(client: AsyncLlamaCloud, pdf_path: Path, out_dir: Path) -> str:
    """
    使用LlamaCloud云端API解析单个PDF文件（异步）
    工作流程：
    1. 上传PDF文件到LlamaCloud云端
    2. 调用解析接口，使用agentic模式（智能解析表格、图表）
    3. 获取Markdown格式的解析结果
    4. 逐页保存到本地.md文件
    Args:
        client: LlamaCloud异步客户端实例
        pdf_path: PDF文件路径
        out_dir: 输出目录
    Returns:
        保存的Markdown文件路径
    """

    # Step 1: 上传PDF文件到云端
    file = await client.files.create(file=str(pdf_path), purpose="parse")
    
    # Step 2: 调用解析接口
    result = await client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],  # 要求返回markdown格式
    )
    
    # Step 3: 拼接各页的Markdown内容
    markdown = ""
    for page in result.markdown.pages:
        markdown += page.markdown + "\n\n"  # 每页后加两个换行分隔
    
    # Step 4: 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 5: 保存到本地文件
    # 文件名与PDF相同，扩展名改为.md
    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    
    return str(out_path)


async def run_parse(
    pdf_dir: str,
    out_dir: str,
    skip_exists: bool = True,
) -> None:
    """
    批量解析PDF的主函数
    Args:
        pdf_dir: PDF文件所在目录
        out_dir: 输出Markdown文件目录
        skip_exists: 如果输出文件已存在，是否跳过（True=跳过，False=重新解析）
    """
    
    print("="*60)
    print("解析PDF")
    print("="*60)

    # Step 1: 检查API Key
    api_key = safe_env("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("未检测到 LLAMA_CLOUD_API_KEY。请在.env文件中配置或设置环境变量")
    
    # Step 2: 创建LlamaCloud客户端（llama模式）
    # 如果使用local模式，client为None
    client = AsyncLlamaCloud() 
    
    # Step 3: 获取所有PDF文件
    pdfs = list_pdf_files(pdf_dir)
    if not pdfs:
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_dir}")
    
    # Step 4: 逐个处理PDF文件
    out_path = Path(out_dir)
    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        target = out_path / f"{pdf.stem}.md"
        
        # 如果输出文件已存在且允许跳过，则跳过该文件
        if skip_exists and target.exists():
            continue
        
        try:
            saved = await parse_one(client, pdf, out_path)
            print(f"[OK] {pdf.name} -> {saved}")
        except Exception as exc:
            print(f"[FAILED] {pdf.name}: {exc}")


def main() -> None:
    """
    使用示例：
    python parse_pdfs.py --pdf-dir ./pdf --out-dir ./ragtest/input --parser llama
    """
    parser = argparse.ArgumentParser(description="将PDF解析为Markdown格式")
    
    parser.add_argument(
        "--pdf-dir", 
        required=True, 
        help="PDF文件所在目录路径"
    )
    
    parser.add_argument(
        "--out-dir", 
        required=True, 
        help="解析结果Markdown文件输出目录"
    )
    
    parser.add_argument(
        "--no-skip-exists", 
        action="store_true", 
        help="已存在文件也重新解析（默认会跳过已存在的文件）"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # 运行异步解析
    asyncio.run(
        run_parse(
            args.pdf_dir,
            args.out_dir,
            skip_exists=not args.no_skip_exists,  # --no-skip-exists 时 skip_exists=False
            parser_mode=args.parser,
        )
    )


if __name__ == "__main__":
    main()