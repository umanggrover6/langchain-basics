from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

chat_history = []

system_message = SystemMessage("You're an expert math teacher.")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

print("---Message History---")
print(chat_history)