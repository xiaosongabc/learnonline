#!/usr/bin/env python3
"""
批量读取 .docx 文件中的标题，并将标题作为新文件名。
标题按固定模板正则提取：
  2000年3月山东省生态系统生产总值核算专题报告

用法示例：
  # 先预览将要重命名的结果（默认 dry-run）
  python rename_gep.py /path/to/docs

  # 真正执行重命名
  python rename_gep.py /path/to/docs --apply
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}
INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')
TITLE_PATTERN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*山东省生态系统生产总值核算专题报告")


def normalize_text(text: str) -> str:
    """规范化文本：合并空白并移除空格。"""
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "")


def sanitize_filename(name: str) -> str:
    """移除 Windows/通用文件系统不允许的字符。"""
    name = INVALID_FILENAME_CHARS.sub("_", name)
    return name.rstrip(" .")


def read_docx_text(docx_path: Path) -> str:
    """读取 docx 中所有段落并拼接为一个字符串。"""
    try:
        with zipfile.ZipFile(docx_path) as zf:
            xml_data = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, FileNotFoundError):
        return ""

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return ""

    texts: list[str] = []
    for p in root.findall(".//w:p", NS):
        segs = [t.text for t in p.findall(".//w:t", NS) if t.text]
        if segs:
            texts.append("".join(segs))

    return normalize_text("".join(texts))


def extract_title(docx_path: Path) -> str | None:
    """按固定模板正则提取标题。"""
    content = read_docx_text(docx_path)
    if not content:
        return None

    match = TITLE_PATTERN.search(content)
    if not match:
        return None

    year, month = match.group(1), str(int(match.group(2)))
    title = f"{year}年{month}月山东省生态系统生产总值核算专题报告"
    return sanitize_filename(title)


def unique_target_path(target: Path) -> Path:
    """若目标已存在，则自动追加 (2), (3) ... 避免覆盖。"""
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    i = 2
    while True:
        candidate = target.with_name(f"{stem}({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def iter_docx_files(root: Path) -> list[Path]:
    """只处理一级目录下的 .docx 文件。"""
    return sorted([p for p in root.glob("*.docx") if p.is_file()])


def rename_docs(root: Path, apply: bool) -> None:
    files = iter_docx_files(root)
    if not files:
        print(f"未找到 .docx 文件：{root}")
        return

    renamed = 0
    skipped = 0

    for path in files:
        title = extract_title(path)
        if not title:
            print(f"[跳过] 未匹配到标题模板: {path.name}")
            skipped += 1
            continue

        target = path.with_name(f"{title}{path.suffix}")
        if target.resolve() == path.resolve():
            print(f"[跳过] 文件名已是目标标题: {path.name}")
            skipped += 1
            continue

        target = unique_target_path(target)
        print(f"[重命名] {path.name}  ->  {target.name}")

        if apply:
            path.rename(target)
            renamed += 1

    if apply:
        print(f"\n完成：实际重命名 {renamed} 个，跳过 {skipped} 个。")
    else:
        print(f"\n预览完成：可重命名 {len(files) - skipped} 个，跳过 {skipped} 个。")
        print("提示：加上 --apply 才会真正改名。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 docx 标题并批量重命名（仅一级目录）")
    parser.add_argument("path", nargs="?", default=".", help="包含 docx 的目录，默认当前目录")
    parser.add_argument("--apply", action="store_true", help="实际执行重命名（默认仅预览）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"目录不存在或不可用: {root}")

    rename_docs(root=root, apply=args.apply)


if __name__ == "__main__":
    main()
