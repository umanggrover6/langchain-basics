from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
from IPython.display import display, Markdown

load_dotenv()

llm = ChatVertexAI(model="gemini-2.0-flash-lite")

examples = [{"input":"What is AI?",
             "output":"AI stands for Artificial Intelligence.."},
            {"input": "What is WWW?",
             "output":"WWW stands for World Wide Web."},
            {"input":"What is the colour of sky?",
             "output":"Sorry, I can only help with full forms."}]

example_prompt = ChatPromptTemplate.from_messages([
    ('human','What is {input}?'),
    ('ai','{output}')
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [('system','You only tell the full forms'),
     few_shot_prompt,
     ('human','{input}')
     ])

chain = final_prompt | llm


result_1 = chain.invoke("API")
print(result_1)

result_2 = chain.invoke("What is the capital of India?")
print(result_2)