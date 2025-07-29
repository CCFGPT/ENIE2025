from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

texts = [
    "ENIE® 2025 se realizará el 10 y 11 de octubre en el Colegio Concepción Fraternidad, Concepción, Chile.",
    "El Colegio Concepción Fraternidad pertenece a la Corporación Educacional Masónica de Concepción (COEMCO).",
    "COEMCO administra colegios que promueven la excelencia académica y valores laicos.",
    "El ENIE® es un evento educativo nacional que fomenta la innovación, la informática y el emprendimiento educativo.",
    "En versiones anteriores del ENIE se abordaron temáticas sobre IA, programación y educación digital."
]

db = FAISS.from_texts(texts, embeddings)
db.save_local("enie_index")
print("Índice FAISS creado exitosamente.")
