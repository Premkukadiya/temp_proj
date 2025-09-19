import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Groq chat model
chat = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

class DoubtRequest(BaseModel):
    subject: str
    doubt: str

@app.get("/")
async def root():
    return {"message": "Your service is live! Use POST /ask to solve doubts."}

@app.post("/ask")
async def solve_doubt(data: DoubtRequest):
    prompt = f"You are an expert in {data.subject}. Answer step by step in a clear structured way:\n\n{data.doubt}"
    response = chat.invoke([HumanMessage(content=prompt)])
    
    formatted_answer = (
        response.content
        .replace("\n\n", "</p><p>")
        .replace("\n", "<br>")
    )
    formatted_answer = f"<p>{formatted_answer}</p>"

    return {"answer": formatted_answer}
