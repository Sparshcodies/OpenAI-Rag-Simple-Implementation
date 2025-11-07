
import re, io
from typing import List
import pandas as pd
from utils.logging_utils import log_error

try:
    from docx import Document
    from pypdf import PdfReader
except Exception:
    raise ImportError("Install required modules first: pip install -r requirements.txt")


def extract_docx_text(docx_bytes: bytes) -> str:
    try:
        doc = Document(docx_bytes)
        paragraphs = []

        for p in doc.paragraphs:
            txt = re.sub(r"\s+", " ", p.text).strip()
            if txt:
                paragraphs.append(txt)

        final_text = "\n".join(paragraphs)
        return final_text

    except Exception as e:
        log_error(f"DOCX extraction failed: {str(e)}")


def extract_csv_text(csv_bytes: bytes) -> str:
    try:
        try:
            df = pd.read_csv(csv_bytes)
        except:
            df = pd.read_csv(csv_bytes, encoding_errors="ignore")

        final_text = df.astype(str).to_string(index=False)
        return final_text

    except Exception as e:
        log_error(f"CSV extraction failed: {str(e)}")
        return ""


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                log_error(f"PDF page parse failed: {str(e)}")
                continue

        final_text = "\n".join(pages)
        return final_text

    except Exception as e:
        log_error(f"PDF extraction failed: {str(e)}")


def extract_txt_text(txt_bytes: bytes) -> str:
    try:
        final_text = txt_bytes.decode("utf-8", errors="ignore")
        return final_text

    except Exception as e:
        log_error(f"TXT extraction failed: {str(e)}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    try:
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
            i += max(1, chunk_size - overlap)
        return chunks

    except Exception as e:
        log_error(f"Chunking failed: {str(e)}")
