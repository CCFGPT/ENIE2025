from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.load_local("enie_index", embeddings)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message")

    answer = qa_chain.run(user_message)

    return {"reply": answer}
