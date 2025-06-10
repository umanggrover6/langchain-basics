from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as key strength. Keep it to 4 lines max."

# prompt_template = ChatPromptTemplate.from_template(template=template)

# prompt =  prompt_template.invoke({
#     "tone": "energetic", 
#     "company": "samsung", 
#     "position": "AI Engineer", 
#     "skill": "AI"
# })

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = llm.invoke(prompt)
print(result)
