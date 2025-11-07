import os
from typing import Dict, Any
from dotenv import load_dotenv
from utils.vector_utils import VectorStore
from utils.logging_utils import log_error, log_query
from openai import OpenAI
from google.genai import Client

load_dotenv()

class QueryEngine:
    def __init__(self, db_path: str = "./chroma_db"):
        try:
            self.store = VectorStore(db_path=db_path)
            self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()
            if self.provider == "openai":
                key = os.environ.get("OPENAI_API_KEY")
                self.client = OpenAI(api_key=key) if key else None
                self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

            elif self.provider == "gemini":
                key = os.environ.get("GEMINI_API_KEY")
                self.client = Client(api_key=key) if key else None
                self.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        except Exception as e:
            log_error(f"Initialization failed: {str(e)}")
            print("Error initializing QueryEngine. Check logs for details.")

    def ask(self, query: str, top_k: int = 4, threshold: float = 0.20) -> Dict[str, Any]:
        try:
            hits = self.store.search(query, k=top_k)
            print("🔍 DEBUG >> hits:", hits)
            if not hits or hits[0][1] < threshold:
                log_error("No relevant context found for the query.")
                return {"answer": "I don’t know based on the provided documents.", "sources": []}

            context = "\n\n".join([h[2] for h in hits])
            if not self.client:
                answer = hits[0][2]
                log_error("API key missing. Returning top context instead.")
                return {"answer": answer, "sources": [{"id": h[0], "similarity": h[1]} for h in hits]}

            prompt = f"""
            You answer strictly from the context below. If the answer is not fully present, reply only:
            "I don’t know based on the provided documents."

            Context:
            {context}

            Question: {query}

            Answer:
            """.strip()

            # ------------------- OPENAI -------------------
            if self.provider == "openai":
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                answer = res.choices[0].message.content.strip()

            # ------------------- GEMINI -------------------
            elif self.provider == "gemini":
                res = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
                answer = res.text.strip()


            log_query(query, answer)
            return {"answer": answer, "sources": [{"id": h[0], "similarity": h[1]} for h in hits]}
        except Exception as e:
            log_error(f"Error in ask(): {str(e)}")
            return {"answer": "An internal error occurred while processing the query.", "sources": []}
        
engine = QueryEngine()