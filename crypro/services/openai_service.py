

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# Example schema for structured LLM output
class AnalysisResponse(BaseModel):
    decision: str  # BUY, SELL, or HOLD
    reasoning: str
    confidence: float  # 0.0 - 1.0
    data_used: list[str]

# LLM Setup
gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = PydanticOutputParser(pydantic_object=AnalysisResponse)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a crypto analysis agent. Use all the input data to return one action:
            BUY, SELL, or HOLD. Provide reasoning and confidence level.
            Wrap your response in this format only: {format_instructions}
            """
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# No tools used for now (could be added)
tools = []

agent = create_tool_calling_agent(
    llm=gpt,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def ask_openai(query: str):
    try:
        raw_response = agent_executor.invoke({"query": query})
        return parser.parse(raw_response.get("output")).dict()
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return {"decision": "HOLD", "reasoning": "Error occurred", "confidence": 0.0, "data_used": []}