from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

response = llm.invoke("What is Vertex AI?")

print(response)