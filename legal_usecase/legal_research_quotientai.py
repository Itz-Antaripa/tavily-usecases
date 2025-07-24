"""
Legal Research with Tavily + Quotient AI

1.  Ask a legal question in natural language
2.  LangChain agent with Tavily tools to fetch top-ranked legal search with citations
3.  Pass sources + question to OpenAI for legal memo generation
4.  Quotient for automatic monitoring and evaluation

Prerequisites:
pip install tavily-python quotientai openai tiktoken

export TAVILY_API_KEY=tvly-xxx
export OPENAI_API_KEY=sk-xxx
export QUOTIENT_API_KEY=qta-xxx
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract
from langchain.schema import HumanMessage
from quotientai import QuotientAI, DetectionType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
quotient = QuotientAI()

# Initialize Quotient monitoring
quotient.logger.init(
    app_name="legal-research-agent",
    environment="production",
    sample_rate=1.0,
    detections=[DetectionType.HALLUCINATION, DetectionType.DOCUMENT_RELEVANCY],
    detection_sample_rate=1.0,
    tags={"feature": "legal-research", "model": "gpt-4o"}
)


def create_legal_research_agent():
    """Create a LangChain agent specialized for legal research"""

    # Legal-focused Tavily tools
    tavily_search = TavilySearch(
        max_results=10,
        search_depth="advanced",
        include_domains=[
            "law.cornell.edu",
            "supreme.justia.com",
            "caselaw.findlaw.com",
            "courtlistener.com",
            "ecfr.gov",
            "federalregister.gov"
        ]
    )

    tavily_extract = TavilyExtract()
    tools = [tavily_search, tavily_extract]

    # Legal research prompt
    today = datetime.now().strftime("%B %d, %Y")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a senior legal research associate. Your task is to research legal questions and draft professional memoranda.

RESEARCH PROCESS:
1. Search for authoritative legal sources (statutes, case law, regulations)
2. Extract detailed content from the most relevant sources
3. Draft a comprehensive legal memorandum

MEMORANDUM FORMAT:
MEMORANDUM

TO: Client
FROM: Legal Research Team  
DATE: {today}
RE: [Legal Question]

EXECUTIVE SUMMARY
[Brief 2-3 sentence answer]

LEGAL ANALYSIS
[Detailed analysis with proper citations etc.]

CONCLUSION
[Clear conclusion addressing the question]

REQUIREMENTS:
- Use only the sources you retrieve
- Cite every assertion with source numbers [1], [2], etc.
- Be precise and avoid speculation
- If sources don't fully address the issue, state this explicitly"""),

        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor


def extract_documents_from_response(response: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract documents from agent response for Quotient logging"""

    documents = []

    for step in response.get("intermediate_steps", []):
        tool_call, tool_output = step
        tool_name = getattr(tool_call, "tool", "")

        # Handle TavilyExtract - full content
        if tool_name == "tavily_extract":
            for result in tool_output.get('results', []):
                doc = {
                    "page_content": result.get('raw_content', ''),
                    "metadata": {"source": result.get('url', ''), "tool": "tavily_extract"}
                }
                documents.append(doc)

        # Handle TavilySearch - snippets
        elif tool_name == "tavily_search":
            for result in tool_output.get('results', []):
                doc = {
                    "page_content": result.get('content', ''),
                    "metadata": {"source": result.get('url', ''), "tool": "tavily_search"}
                }
                documents.append(doc)

    return documents


def conduct_legal_research(query: str) -> Dict[str, Any]:
    """
    Main legal research workflow using LangChain agent + Quotient monitoring

    Args:
        query: Legal question to research

    Returns:
        Research results with memo and evaluation
    """
    logger.info(f"Starting legal research: {query}")
    start_time = datetime.now()

    try:
        # Create legal research agent
        agent_executor = create_legal_research_agent()

        # Run the agent
        response = agent_executor.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # Extract results
        memo_content = response['output']
        documents = extract_documents_from_response(response)

        logger.info(f"Generated memo with {len(documents)} source documents")

        # Log to Quotient for monitoring
        log_id = quotient.log(
            user_query=query,
            model_output=memo_content,
            documents=documents,
            tags={"query_type": "legal_research"}
        )

        # Get evaluation results
        detection_result = quotient.poll_for_detection(log_id=log_id)

        # Compile results
        duration = (datetime.now() - start_time).total_seconds()

        results = {
            "query": query,
            "memo": memo_content,
            "documents": documents,
            "quotient_log_id": log_id,
            "quotient_evaluation": detection_result,
            "total_sources": len(documents),
            "duration_seconds": duration,
            "generated_at": start_time.isoformat()
        }

        logger.info(f"Legal research completed in {duration:.2f}s")
        return results

    except Exception as e:
        logger.error(f"Error in legal research workflow: {e}")
        return {
            "query": query,
            "error": str(e),
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }


if __name__ == "__main__":
    # Run single example
    question = "What is the current standard for granting summary judgment under Rule 56 of the U.S. Federal Rules of Civil Procedure?"

    try:
        results = conduct_legal_research(question)
        print(results)

    except Exception as e:
        print(f"Error: {e}")
