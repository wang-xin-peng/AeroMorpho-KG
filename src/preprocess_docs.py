"""
文档预处理模块
功能：
1. 清理解析后的Markdown文档中的无关内容
2. 移除作者信息、出版机构、参考文献、页眉页脚等
3. 处理表格数据（HTML表格转Markdown）
4. 处理图表引用（保留描述，移除图片标记）
5. 按章节切分文档
6. 保留核心技术内容用于知识抽取
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup


def convert_html_table_to_markdown(html_table: str) -> str:
    """
    将HTML表格转换为Markdown表格
    
    Args:
        html_table: HTML表格字符串
    
    Returns:
        Markdown格式的表格
    """
    try:
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return html_table
        
        markdown_lines = []
        
        # 处理表头
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                if headers:
                    markdown_lines.append('| ' + ' | '.join(headers) + ' |')
                    markdown_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        # 处理表体
        tbody = table.find('tbody')
        rows = tbody.find_all('tr') if tbody else table.find_all('tr')
        
        for row in rows:
            # 跳过已经处理过的表头行
            if thead and row.parent == thead:
                continue
            
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if cells:
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(markdown_lines)
    
    except Exception as e:
        # 如果转换失败，返回原始内容
        return html_table


def process_tables(text: str) -> str:
    """
    处理文档中的所有表格
    将HTML表格转换为Markdown格式
    """
    # 查找所有<table>标签
    table_pattern = r'<table[^>]*>.*?</table>'
    
    def replace_table(match):
        html_table = match.group(0)
        return convert_html_table_to_markdown(html_table)
    
    text = re.sub(table_pattern, replace_table, text, flags=re.DOTALL | re.IGNORECASE)
    
    return text


def process_image_references(text: str) -> str:
    """
    处理图片引用
    策略：
    1. 保留图片描述文字（如"图2.1 沙漠蝗虫翅膀的形态图示"）
    2. 移除图片标记（如![xxx](page_xx_image_x.jpg)）
    3. 保留图表标题作为上下文
    """
    # 移除图片markdown标记，但保留alt文本（如果有描述性内容）
    def replace_image(match):
        alt_text = match.group(1)
        # 如果alt文本有实际描述（不只是"图片"、"image"等），保留它
        if alt_text and len(alt_text) > 3 and not re.match(r'^(图片|image|img|pic)$', alt_text, re.IGNORECASE):
            return alt_text
        return ''
    
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', replace_image, text)
    
    # 保留图表标题（这些通常包含重要信息）
    # 格式：图X.X xxx 或 表X.X xxx
    # 不做处理，让它们保留在文本中
    
    return text


def split_by_chapters(text: str) -> List[Tuple[str, str]]:
    """
    按章节切分文档
    
    Returns:
        List of (chapter_title, chapter_content)
    """
    # 匹配章节标题模式
    # 支持：# 第X章、## 第X章、# X. 、# Chapter X等
    chapter_pattern = r'^(#+)\s*(第?\s*[0-9一二三四五六七八九十]+\s*[章节]|Chapter\s+\d+|[0-9]+\.)\s*(.*)$'
    
    chapters = []
    current_chapter = None
    current_content = []
    
    lines = text.split('\n')
    
    for line in lines:
        match = re.match(chapter_pattern, line, re.IGNORECASE)
        
        if match:
            # 保存上一章节
            if current_chapter:
                chapters.append((current_chapter, '\n'.join(current_content)))
            
            # 开始新章节
            current_chapter = line.strip()
            current_content = []
        else:
            if current_chapter:
                current_content.append(line)
            else:
                # 文档开头的内容（标题、摘要等）
                current_content.append(line)
    
    # 保存最后一章
    if current_chapter:
        chapters.append((current_chapter, '\n'.join(current_content)))
    elif current_content:
        # 如果没有章节标记，整个文档作为一个章节
        chapters.append(("全文", '\n'.join(current_content)))
    
    return chapters


def split_into_sentences(text: str) -> List[str]:
    """
    将文本按句子切分
    使用简单规则：按 。！？；\n 分割
    
    Args:
        text: 输入文本
    
    Returns:
        句子列表
    """
    # 按标点符号分句
    sentences = re.split(r'[。！？；\n]+', text)
    
    # 清理空句子和过短的句子
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    return sentences


def remove_references(text: str) -> str:
    """
    移除参考文献部分
    策略：
    1. 如果文档有多个一级参考文献标题(# 参考文献)，说明是书籍，每章都有参考文献，不删除
    2. 如果文档只有1个一级参考文献标题，说明是论文，删除最后的参考文献部分
    3. 如果文档没有一级参考文献标题，检查二级标题(## 参考文献)，删除最后一个
    """
    # 查找所有一级参考文献标题
    level1_refs = list(re.finditer(r'\n# 参\s*考\s*文\s*献\s*\n', text, re.IGNORECASE))
    level1_refs += list(re.finditer(r'\n# References?\s*\n', text, re.IGNORECASE))
    
    # 过滤掉二级标题（## 参考文献）
    level1_refs = [m for m in level1_refs if not m.group(0).startswith('\n## ')]
    
    if len(level1_refs) > 1:
        # 多个一级参考文献，说明是书籍，不删除
        return text
    elif len(level1_refs) == 1:
        # 只有一个一级参考文献，删除它及之后的内容
        return text[:level1_refs[0].start()]
    else:
        # 没有一级参考文献，查找二级参考文献
        level2_refs = list(re.finditer(r'\n## 参\s*考\s*文\s*献\s*\n', text, re.IGNORECASE))
        level2_refs += list(re.finditer(r'\n## References?\s*\n', text, re.IGNORECASE))
        
        if level2_refs:
            # 删除最后一个二级参考文献及之后的内容
            return text[:level2_refs[-1].start()]
    
    return text


def remove_author_info(text: str) -> str:
    """
    移除作者信息
    常见模式：
    - 作者：xxx
    - 通讯作者：xxx
    - 单位：xxx
    - （xxx大学 xxx学院）
    - 上标标记: <sup>1, *</sup>
    - 邮箱地址
    """
    # 移除作者行
    text = re.sub(r'^作者[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^通讯作者[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^联系人[：:].+$', '', text, flags=re.MULTILINE)
    
    # 移除上标标记 (如 <sup>1, *</sup>)
    text = re.sub(r'`<sup>`[^`]+`</sup>`', '', text)
    text = re.sub(r'<sup>[^<]+</sup>', '', text)
    
    # 移除邮箱地址
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # 移除单位信息（括号包裹的地址）
    text = re.sub(r'（[^）]*(?:大学|学院|研究院|研究所|公司|集团)[^）]*\d{6}[^）]*）', '', text)
    text = re.sub(r'\([^)]*(?:University|Institute|College|Company)[^)]*\d{5}[^)]*\)', '', text, flags=re.IGNORECASE)
    
    return text


def remove_thesis_metadata(text: str) -> str:
    """
    移除学位论文的元数据和格式内容
    包括:
    - 封面信息(培养单位、学科、研究生、指导教师等)
    - 授权声明
    - 目录
    - 插图清单
    - 附表清单
    - 符号和缩略语说明
    - 评阅人和答辩委员会名单
    - 中图分类号、文献标识码、文章编号等
    - 收稿日期、基金项目等
    - 引用格式
    - 出版信息、版权信息、ISBN、DOI等
    - 封底信息（责任编辑、内容简介、定价等）
    """
    # 移除DOI
    text = re.sub(r'\*\*DOI[：:]\*\*.+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^DOI[：:].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 移除ISBN
    text = re.sub(r'ISBN\s+[\d\-]+', '', text, flags=re.IGNORECASE)
    
    # 移除出版信息
    text = re.sub(r'\*\*出版发行\*\*[：:].+', '', text)
    text = re.sub(r'\*\*地址邮编\*\*[：:].+', '', text)
    text = re.sub(r'\*\*经\s*售\*\*[：:].+', '', text)
    text = re.sub(r'\*\*印\s*刷\*\*[：:].+', '', text)
    text = re.sub(r'\*\*开\s*本\*\*[：:].+', '', text)
    text = re.sub(r'\*\*印\s*张\*\*[：:].+', '', text)
    text = re.sub(r'\*\*字\s*数\*\*[：:].+', '', text)
    text = re.sub(r'\*\*版\s*印\s*次\*\*[：:].+', '', text)
    text = re.sub(r'\*\*印\s*数\*\*[：:].+', '', text)
    text = re.sub(r'\*\*定\s*价\*\*[：:].+', '', text)
    text = re.sub(r'\*\*国防书店\*\*[：:].+', '', text)
    text = re.sub(r'\*\*发行邮购\*\*[：:].+', '', text)
    text = re.sub(r'\*\*发行传真\*\*[：:].+', '', text)
    text = re.sub(r'\*\*发行业务\*\*[：:].+', '', text)
    
    # 移除封底信息（责任编辑、责任校对、内容简介等）
    text = re.sub(r'\*\*责任编辑[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*责任校对[：:]\*\*.+', '', text)
    text = re.sub(r'#+\s*内容简介.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'定价[：:]\s*[\d\.]+\s*元', '', text)
    text = re.sub(r'上架建议.+', '', text)
    text = re.sub(r'Barcode.+', '', text)
    
    # 移除包含ISBN的表格（封底信息）- 匹配整个表格块
    text = re.sub(r'\|[^\n]*ISBN[^\n]*\|[^\n]*\n(\|[^\n]*\n)*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|[^\n]*URL[^\n]*\|[^\n]*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|[^\n]*无人机[^\n]*\|[^\n]*\n', '', text)
    text = re.sub(r'\|[^\n]*787118[^\n]*\|[^\n]*\n', '', text)  # ISBN条形码
    
    # 移除WILEY标记（出版社标记）
    text = re.sub(r'#+\s*WILEY\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'QR Code with.+', '', text, flags=re.IGNORECASE)
    
    # 移除版权信息
    text = re.sub(r'Translation from the English Language edition:.+?John Wiley & Sons Limited\.', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Copyright © \d{4}.+?All Rights Reserved\.', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Authorised translation from.+?John Wiley & Sons Limited\.', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Copies of this book sold without.+?illegal', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Wiley Genuine Product Seal', '', text, flags=re.IGNORECASE)
    
    # 移除图书在版编目(CIP)数据
    text = re.sub(r'#+\s*图书在版编目\s*\(CIP\)\s*数据.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'中国版本图书馆\s*CIP\s*数据核字.+', '', text, flags=re.IGNORECASE)
    
    # 移除著作权合同登记
    text = re.sub(r'#+\s*著作权合同登记.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'图字[：:].+', '', text)
    
    # 移除mark标签内容
    text = re.sub(r'<mark>.+?</mark>', '', text, flags=re.DOTALL)
    
    # 移除引用格式行
    text = re.sub(r'^引用格式[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^RAN M P.+?\(in Chinese\)\.?$', '', text, flags=re.MULTILINE | re.DOTALL)
    
    # 移除中图分类号、文献标识码等
    text = re.sub(r'\*\*中图分类号[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*文献标识码[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*文章编号[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*收稿日期[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*网络出版时间[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*网络出版地址[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*基金项目[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*退修日期[：:]\*\*.+', '', text)
    text = re.sub(r'\*\*录用日期[：:]\*\*.+', '', text)
    
    # 移除logo标记
    text = re.sub(r'\blogo\b', '', text, flags=re.IGNORECASE)
    
    # 移除培养单位、学科等封面信息
    text = re.sub(r'\*\*?培养单位\*\*?[：:].+', '', text)
    text = re.sub(r'\*\*?学\s*科\*\*?[：:].+', '', text)
    text = re.sub(r'\*\*?研\s*究\s*生\*\*?[：:].+', '', text)
    text = re.sub(r'\*\*?指导教师\*\*?[：:].+', '', text)
    
    # 移除Dissertation Submitted等英文封面信息
    text = re.sub(r'Dissertation Submitted to.+?by\s+\w+\s+\w+', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除授权声明部分
    patterns_to_remove = [
        r'#+\s*关于学位论文使用授权的说明.+?(?=\n#+\s|\Z)',
        r'#+\s*学位论文公开评阅人和答辩委员会名单.+?(?=\n#+\s|\Z)',
        r'#+\s*公开评阅人名单.+?(?=\n#+\s|\Z)',
        r'#+\s*答辩委员会名单.+?(?=\n#+\s|\Z)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # 移除目录标题及其后的内容(直到下一个一级标题)
    # 只移除明确标记为"目录"的章节
    text = re.sub(r'#+\s*目\s*录\s*\n.+?(?=\n#\s+[^#])', '', text, flags=re.DOTALL)
    
    # 移除插图清单
    text = re.sub(r'#+\s*插图清单.+?(?=\n#+\s[^插]|\Z)', '', text, flags=re.DOTALL)
    
    # 移除附表清单
    text = re.sub(r'#+\s*附表清单.+?(?=\n#+\s[^附]|\Z)', '', text, flags=re.DOTALL)
    
    # 移除符号和缩略语说明
    text = re.sub(r'#+\s*符号和缩略语说明.+?(?=\n#+\s[^符]|\Z)', '', text, flags=re.DOTALL)
    
    # 移除致谢、声明、个人简历等
    text = re.sub(r'#+\s*致\s*谢.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'#+\s*声\s*明.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'#+\s*个人简历.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'#+\s*指导教师评语.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'#+\s*答辩委员会决议书.+?(?=\n#+\s|\Z)', '', text, flags=re.DOTALL)
    
    return text


def remove_abstract_keywords(text: str) -> str:
    """
    移除摘要和关键词（可选，根据需求决定是否保留）
    这里选择保留摘要，因为摘要通常包含重要信息
    只移除"摘要："、"关键词："等标签
    """
    # 移除关键词标签行（但保留关键词内容）
    text = re.sub(r'^\*\*关键词[：:]\*\*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^关键词[：:]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\*Keywords?[：:]\*\*\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Keywords?[：:]\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text


def remove_page_markers(text: str) -> str:
    """
    移除页码、页眉、页脚标记
    常见模式：
    - <page_footer>xxx</page_footer>
    - 第 X 页
    - Page X
    """
    # 移除XML标签
    text = re.sub(r'<page_footer>.*?</page_footer>', '', text, flags=re.DOTALL)
    text = re.sub(r'<page_header>.*?</page_header>', '', text, flags=re.DOTALL)
    
    # 移除页码
    text = re.sub(r'^第\s*\d+\s*页\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\d+\s*/\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text


def remove_urls_and_dois(text: str) -> str:
    """
    移除URL和DOI链接（通常在参考文献中）
    """
    # 移除URL
    text = re.sub(r'https?://[^\s\)]+', '', text)
    
    # 移除DOI
    text = re.sub(r'doi[：:]?\s*10\.\d+/[^\s]+', '', text, flags=re.IGNORECASE)
    
    return text


def remove_figure_table_captions(text: str) -> str:
    """
    移除图表标题（可选）
    保留图表内容描述，只移除"图X"、"表X"等标记
    """
    # 移除独立的图表标题行
    text = re.sub(r'^图\s*\d+[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^表\s*\d+[：:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Fig\.?\s*\d+[：:].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^Table\s+\d+[：:].+$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text


def remove_citation_markers(text: str) -> str:
    """
    移除文中的引用标记
    常见模式：
    - [1]
    - [1-3]
    - [1,2,3]
    - <sup>[1]</sup>
    """
    # 移除上标引用
    text = re.sub(r'<sup>\[[\d,\-\s]+\]</sup>', '', text)
    
    # 移除方括号引用（但保留其他方括号内容）
    text = re.sub(r'\[[\d,\-\s]+\]', '', text)
    
    return text


def clean_whitespace(text: str) -> str:
    """
    清理多余的空白字符
    - 移除多余的空行（超过2个连续换行）
    - 移除行首行尾空格
    """
    # 移除每行首尾空格
    lines = [line.strip() for line in text.split('\n')]
    
    # 移除多余空行（保留最多1个空行）
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned_lines.append(line)
            prev_empty = False
        else:
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
    
    return '\n'.join(cleaned_lines)


def preprocess_document(text: str, keep_abstract: bool = True, process_tables_flag: bool = True, process_images_flag: bool = True) -> str:
    """
    完整的文档预处理流程
    
    Args:
        text: 原始文档文本
        keep_abstract: 是否保留摘要部分
        process_tables_flag: 是否处理表格（HTML转Markdown）
        process_images_flag: 是否处理图片引用
    
    Returns:
        清理后的文档文本
    """
    # 1. 移除学位论文元数据（目录、插图清单、符号说明等）
    text = remove_thesis_metadata(text)
    
    # 2. 移除参考文献（最先做，避免后续处理参考文献内容）
    text = remove_references(text)
    
    # 3. 移除作者和单位信息
    text = remove_author_info(text)
    
    # 4. 移除页码和页眉页脚
    text = remove_page_markers(text)
    
    # 5. 移除URL和DOI
    text = remove_urls_and_dois(text)
    
    # 6. 移除引用标记
    text = remove_citation_markers(text)
    
    # 7. 处理表格（HTML转Markdown）
    if process_tables_flag:
        text = process_tables(text)
    
    # 8. 处理图片引用（保留描述，移除图片标记）
    if process_images_flag:
        text = process_image_references(text)
    
    # 9. 移除图表标题（可选）
    # text = remove_figure_table_captions(text)
    
    # 10. 处理摘要和关键词
    if not keep_abstract:
        text = remove_abstract_keywords(text)
    
    # 11. 再次清理元数据（因为表格转换可能产生新的Markdown格式的元数据）
    text = remove_thesis_metadata(text)
    
    # 12. 清理空白字符
    text = clean_whitespace(text)
    
    return text


def process_directory(
    input_dir: str,
    output_dir: str,
    keep_abstract: bool = True,
    split_chapters: bool = False,
    sentence_mode: bool = False,
) -> int:
    """
    批量处理目录中的所有Markdown文件
    
    Args:
        input_dir: 输入目录（解析后的Markdown）
        output_dir: 输出目录（预处理后的Markdown）
        keep_abstract: 是否保留摘要
        split_chapters: 是否按章节切分并保存为独立文件
        sentence_mode: 是否输出句子级别的切分（用于细粒度抽取）
    
    Returns:
        处理的文件数量
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理所有Markdown文件
    md_files = list(input_path.glob("*.md"))
    processed_count = 0
    
    for md_file in md_files:
        print(f"处理: {md_file.name}")
        
        # 读取原始文档
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        original_length = len(text)
        
        # 预处理
        cleaned_text = preprocess_document(text, keep_abstract=keep_abstract)
        cleaned_length = len(cleaned_text)
        
        reduction = (1 - cleaned_length / original_length) * 100
        print(f"  原始长度: {original_length}, 清理后: {cleaned_length}, 减少: {reduction:.1f}%")
        
        # 根据模式保存文件
        if split_chapters:
            # 按章节切分并保存
            chapters = split_by_chapters(cleaned_text)
            print(f"  检测到 {len(chapters)} 个章节")
            
            base_name = md_file.stem
            for idx, (chapter_title, chapter_content) in enumerate(chapters, 1):
                # 清理章节标题作为文件名
                safe_title = re.sub(r'[^\w\s\-]', '', chapter_title)[:50]
                safe_title = safe_title.strip().replace(' ', '_')
                
                chapter_file = output_path / f"{base_name}_chapter{idx:02d}_{safe_title}.md"
                chapter_file.write_text(f"# {chapter_title}\n\n{chapter_content}", encoding="utf-8")
                print(f"    保存章节: {chapter_file.name}")
        
        elif sentence_mode:
            # 句子级别切分
            sentences = split_into_sentences(cleaned_text)
            print(f"  切分为 {len(sentences)} 个句子")
            
            base_name = md_file.stem
            sentence_file = output_path / f"{base_name}_sentences.txt"
            sentence_file.write_text('\n'.join(sentences), encoding="utf-8")
            print(f"    保存句子文件: {sentence_file.name}")
        
        else:
            # 保存完整的清理后文档
            output_file = output_path / md_file.name
            output_file.write_text(cleaned_text, encoding="utf-8")
        
        processed_count += 1
    
    print(f"\n[完成] 共处理 {processed_count} 个文件")
    print(f"输出目录: {output_dir}")
    
    return processed_count


def main():
    """
    使用示例：
    python preprocess_docs.py \
        --input-dir data/parsed \
        --output-dir data/preprocessed \
        --keep-abstract
    
    按章节切分：
    python preprocess_docs.py \
        --input-dir data/parsed \
        --output-dir data/preprocessed_chapters \
        --split-chapters
    
    句子级别切分：
    python preprocess_docs.py \
        --input-dir data/parsed \
        --output-dir data/preprocessed_sentences \
        --sentence-mode
    """
    parser = argparse.ArgumentParser(description="预处理解析后的Markdown文档")
    parser.add_argument("--input-dir", required=True, help="输入目录（解析后的Markdown）")
    parser.add_argument("--output-dir", required=True, help="输出目录（预处理后的Markdown）")
    parser.add_argument("--keep-abstract", action="store_true", help="保留摘要部分")
    parser.add_argument("--split-chapters", action="store_true", help="按章节切分并保存为独立文件")
    parser.add_argument("--sentence-mode", action="store_true", help="输出句子级别的切分（用于细粒度抽取）")
    
    args = parser.parse_args()
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keep_abstract=args.keep_abstract,
        split_chapters=args.split_chapters,
        sentence_mode=args.sentence_mode,
    )


if __name__ == "__main__":
    main()
