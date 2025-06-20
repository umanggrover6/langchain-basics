from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
    )
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []

system_prompt = "You are a helpful assistant called Zeta."
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template('{query}')
])

pipeline = prompt_template | llm


chat_map = {}
def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    print(f"get_chat_history called with session_id={session_id} and k={k}")
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k=k)
    # remove anything beyond the last
    return chat_map[session_id]


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        )
    ]
)

pipeline_with_history.invoke(
    {"query": "Hi, my name is Umang"},
    config={"configurable": {"session_id": "id_k14", "k": 14}}
)

# manually insert history
chat_map["id_k14"].add_user_message("Hi, my name is Umang")
chat_map["id_k14"].add_ai_message("I'm an AI model called Boss.")
chat_map["id_k14"].add_user_message("I'm researching the different types of conversational memory.")
chat_map["id_k14"].add_ai_message("That's interesting, what are some examples?")
chat_map["id_k14"].add_user_message("I've been looking at ConversationBufferMemory and ConversationBufferWindowMemory.")
chat_map["id_k14"].add_ai_message("That's interesting, what's the difference?")
chat_map["id_k14"].add_user_message("Buffer memory just stores the entire conversation, right?")
chat_map["id_k14"].add_ai_message("That makes sense, what about ConversationBufferWindowMemory?")
chat_map["id_k14"].add_user_message("Buffer window memory stores the last k messages, dropping the rest.")
chat_map["id_k14"].add_ai_message("Very cool!")

# response = pipeline_with_history.invoke(
#     {"query": "what is my name again?"},
#     config={"configurable": {"session_id": "id_k14", "k": 4}}
# )


response = pipeline_with_history.invoke(
    {"query": "what is my name again?"},
    config={"configurable": {"session_id": "id_k14", "k": 14}}
)

print("AI:", response.content)
