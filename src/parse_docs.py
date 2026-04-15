import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from pypdf import PdfReader
from tqdm import tqdm

from common import list_pdf_files, safe_env

load_dotenv()

async def parse_one(client: AsyncLlamaCloud, pdf_path: Path, out_dir: Path) -> str:
    file = await client.files.create(file=str(pdf_path), purpose="parse")
    result = await client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )
    markdown = ""
    for page in result.markdown.pages:
        markdown += page.markdown + "\n\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return str(out_path)


def parse_one_local(pdf_path: Path, out_dir: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"## Page {i}\n\n{text}\n")
    markdown = "\n".join(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return str(out_path)


async def run_parse(
    pdf_dir: str,
    out_dir: str,
    skip_exists: bool = True,
    parser_mode: str = "llama",
) -> None:
    api_key = safe_env("LLAMA_CLOUD_API_KEY")
    if parser_mode == "llama" and not api_key:
        raise ValueError("未检测到 LLAMA_CLOUD_API_KEY。")
    client = AsyncLlamaCloud() if parser_mode == "llama" else None
    pdfs = list_pdf_files(pdf_dir)
    if not pdfs:
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_dir}")

    out_path = Path(out_dir)
    for pdf in tqdm(pdfs, desc="Parsing PDFs"):
        target = out_path / f"{pdf.stem}.md"
        if skip_exists and target.exists():
            continue
        try:
            if parser_mode == "llama":

                saved = await parse_one(client, pdf, out_path)
            else:
                saved = parse_one_local(pdf, out_path)
            print(f"[OK] {pdf.name} -> {saved}")
        except Exception as exc:
            print(f"[FAILED] {pdf.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True, help="PDF 目录")
    parser.add_argument("--out-dir", required=True, help="解析结果 Markdown 输出目录")
    parser.add_argument("--no-skip-exists", action="store_true", help="已存在文件也重新解析")
    parser.add_argument(
        "--parser",
        choices=["llama", "local"],
        default="llama",
        help="解析后端：llama=LlamaCloud，local=pypdf",
    )
    args = parser.parse_args()

    load_dotenv()
    asyncio.run(
        run_parse(
            args.pdf_dir,
            args.out_dir,
            skip_exists=not args.no_skip_exists,
            parser_mode=args.parser,
        )
    )


if __name__ == "__main__":
    main()
