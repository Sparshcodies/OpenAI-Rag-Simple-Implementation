# utils/doc_history_utils.py

import os
import json
from datetime import datetime
from utils.logging_utils import log_error, log_query

FILE_PATH = "indexed_docs.json"

def _load():
    try:
        if not os.path.exists(FILE_PATH):
            return []
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Failed loading indexed docs: {str(e)}")
        return []

def _save(data):
    try:
        with open(FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        log_error(f"Failed saving indexed docs: {str(e)}")

def add_document(name: str, path: str):
    try:
        docs = _load()
        docs.append({
            "name": name,
            "path": path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        _save(docs)
    except Exception as e:
        log_error(f"Failed adding document: {str(e)}")

def get_documents():
    try:
        return _load()
    except Exception as e:
        log_error(f"Failed fetching docs: {str(e)}")
        return []

def clear_documents():
    try:
        _save([])
    except Exception as e:
        log_error(f"Failed clearing docs: {str(e)}")
