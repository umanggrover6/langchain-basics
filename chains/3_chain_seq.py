from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and can convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)


prepare_for_translation = RunnableLambda(lambda output:{"text":output, "language":"french"})



chain = prompt_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser()

result = chain.invoke({"animal": "cat", "fact_count": 2})
print(result)