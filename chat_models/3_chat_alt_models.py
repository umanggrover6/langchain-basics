from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-opus-4-20250514")

messages = [SystemMessage("You're an expert math teacher."),
            HumanMessage("What is the square root of 79?")]

response = llm.invoke(messages)
print(response.content)
