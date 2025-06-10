from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

# messages = [SystemMessage("You're an expert social media content strategist."),
#             HumanMessage("Give a short tip to create engaging posts on instagram.")]

messages = [SystemMessage("You're an expert social media content strategist."),
            HumanMessage("Give a short tip to create engaging posts on instagram."),
            AIMessage("Focus on High-Quality Visuals & Compelling Storytelling.** People scroll fast!  Grab their attention with a visually stunning image or video, then use your caption to tell a concise, engaging story that connects with your audience's emotions or interests."),
            HumanMessage("Give two 5 word captions for my shoes campaign.")]

response = llm.invoke(messages)

print(response.content)