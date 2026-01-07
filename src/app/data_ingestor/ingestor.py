import re, pathlib
from typing import List, Dict, Tuple

H1 = re.compile(r'^\s*#\s+(.*)$', re.MULTILINE)
H2_SPLIT = re.compile(r'^\s*##\s+(.*)$', re.MULTILINE)

def read_markdown(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def extract_title(md_text: str, fallback: str) -> str:
    m = H1.search(md_text)
    return (m.group(1).strip() if m else fallback).strip()

def section_aware_splits(md_text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section_title or '', section_text). Includes preface as ''.
    """
    sections = []
    last_end = 0
    last_title = ""
    for m in H2_SPLIT.finditer(md_text):
        sec_title = m.group(1).strip()
        sec_start = m.start()
        # previous block
        if sec_start > last_end:
            block = md_text[last_end:sec_start].strip()
            if block:
                sections.append((last_title, block))
        last_title = sec_title
        last_end = m.end()
    # tail
    tail = md_text[last_end:].strip()
    if tail:
        sections.append((last_title, tail))
    if not sections:
        sections = [("", md_text)]
    return sections

def normalize_ws(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple char-based chunking with overlap; keeps lists/paragraphs intact where possible.
    """
    text = normalize_ws(text)
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        # Try to break at a sentence end for better readability
        window = text[start:end]
        cut = max(window.rfind(s) for s in ("\n", ". ", "? ", "! ", "\n\n"))
        if cut == -1 or cut < int(chunk_chars * 0.5):
            cut = len(window)
        chunk = window[:cut].strip()
        if chunk:
            chunks.append(chunk)
        end = cut if cut > 0 else chunk_chars
        start = max(0, start + end - overlap)

    # ensure tail
    if chunks and chunks[-1] != text[-len(chunks[-1]):]:
        tail_start = chunks[-1] and text.rfind(chunks[-1]) + len(chunks[-1]) or 0 ## TODO: could be affected by duplicate chunks
        if tail_start < len(text):
            tail = text[tail_start:].strip()
            if tail:
                chunks.append(tail[:chunk_chars])
    # remove tiny fragments
    chunks = [c for c in chunks if len(c) > 10] ## TODO: chunks with less than 10 chars, probably still should be appended or somehow kept
    return chunks or [text]

def md_to_chunks(md_text: str, source_name: str, chunk_chars=1200, overlap=200) -> List[Dict]:
    title = extract_title(md_text, fallback=pathlib.Path(source_name).stem)
    sections = section_aware_splits(md_text)
    items = []
    idx = 0
    for sec_title, sec_text in sections:
        for part in chunk_text(sec_text, chunk_chars=chunk_chars, overlap=overlap):
            idx += 1
            items.append({
                "id": f"{source_name}#chunk_{idx:03d}",
                "title": title,
                "section": sec_title or "Summary",
                "source": source_name,
                "text": part
            })
    return items