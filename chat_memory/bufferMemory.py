from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
    )
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")


# ConversationBufferMemory --> RunnableWithMessageHistory
system_prompt = "You are a helpful assistant called Zeta."
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template('{query}')
])

pipeline = prompt_template | llm

chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
)

pipeline_with_history.invoke(
    {"query":"Hi, My name is Umang."},
    config={"session_id":"id_123"}
)

response = pipeline_with_history.invoke(
    {"query":"What is my name?"},
    config={"session_id":"id_123"}
)

print(response.content)


