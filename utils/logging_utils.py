import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

error_log_file = os.path.join(LOG_DIR, "error_warning.log")
query_log_file = os.path.join(LOG_DIR, "query_response.log")

error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.WARNING)
err_handler = logging.FileHandler(error_log_file)
err_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
error_logger.addHandler(err_handler)

query_logger = logging.getLogger("query_logger")
query_logger.setLevel(logging.INFO)
query_handler = logging.FileHandler(query_log_file)
query_handler.setFormatter(logging.Formatter("%(asctime)s - QUERY: %(message)s"))
query_logger.addHandler(query_handler)


def log_query(query: str, answer: str):
    query_logger.info(f"Query: {query} | Answer: {answer}")


def log_error(message: str):
    error_logger.error(message)


