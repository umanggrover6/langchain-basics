from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")


no_cot_system_prompt = """
Be a helpful assistant and answer the user's question.

You MUST answer the question directly without any other
text or explanation.
"""

no_cot_prompt_template = ChatPromptTemplate.from_messages([
    ("system", no_cot_system_prompt),
    ("user", "{query}"),
])

query = (
    "How many keystrokes are needed to type the numbers from 1 to 500?"
)

no_cot_pipeline = no_cot_prompt_template | llm
no_cot_result = no_cot_pipeline.invoke({"query": query}).content
print(no_cot_result)

# Define the chain-of-thought prompt template
cot_system_prompt = """
Be a helpful assistant and answer the user's question.

To answer the question, you must:

- List systematically and in precise detail all
  subproblems that need to be solved to answer the
  question.
- Solve each sub problem INDIVIDUALLY and in sequence.
- Finally, use everything you have worked through to
  provide the final answer.
"""

cot_prompt_template = ChatPromptTemplate.from_messages([
    ("system", cot_system_prompt),
    ("user", "{query}"),
])

cot_pipeline = cot_prompt_template | llm

cot_result = cot_pipeline.invoke({"query": query}).content
print(cot_result)
