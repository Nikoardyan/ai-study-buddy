"""
Multi-format document loader untuk materi belajar.
Support: PDF, DOCX, TXT, MD
"""
from pathlib import Path
import pdfplumber
from docx import Document


def load_pdf(filepath: str) -> dict:
    """Load PDF dengan info per halaman."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "page_number": page_num,
                    "content": text
                })
    
    return {
        "filename": Path(filepath).name,
        "format": "pdf",
        "pages": pages,
        "total_pages": len(pages)
    }


def load_docx(filepath: str) -> dict:
    """Load Microsoft Word document."""
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    
    return {
        "filename": Path(filepath).name,
        "format": "docx",
        "pages": [{
            "page_number": 1,
            "content": "\n\n".join(paragraphs)
        }],
        "total_pages": 1
    }


def load_text(filepath: str) -> dict:
    """Load plain text atau markdown."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return {
        "filename": Path(filepath).name,
        "format": Path(filepath).suffix[1:],
        "pages": [{
            "page_number": 1,
            "content": content
        }],
        "total_pages": 1
    }


def load_document(filepath: str) -> dict:
    """Universal loader - auto-detect format."""
    extension = Path(filepath).suffix.lower()
    
    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_text,
        ".md": load_text,
    }
    
    if extension not in loaders:
        raise ValueError(f"Format {extension} tidak di-support")
    
    return loaders[extension](filepath)


def load_all_from_folder(folder_path: str) -> list[dict]:
    """Load semua dokumen dari folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder {folder_path} tidak ada!")
        return []
    
    supported = {".pdf", ".docx", ".txt", ".md"}
    all_files = []
    for ext in supported:
        all_files.extend(folder.glob(f"*{ext}"))
    
    print(f"Ketemu {len(all_files)} file di folder")
    
    documents = []
    for file in all_files:
        try:
            print(f"   Loading {file.name}...")
            doc = load_document(str(file))
            documents.append(doc)
            print(f"      OK: {doc['total_pages']} halaman")
        except Exception as e:
            print(f"      Error: {e}")
    
    return documents