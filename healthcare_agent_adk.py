"""Healthcare research agent built on Google ADK + Tavily."""

import argparse
import asyncio
import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _require_env_var(name: str) -> str:
    """Fetch an environment variable or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing environment variable '{name}'. Please add it to your .env file or shell environment."
        )
    return value


# Initialize Tavily once we know the API key is present
TAVILY_API_KEY = _require_env_var("TAVILY_API_KEY")
GOOGLE_API_KEY = _require_env_var("GOOGLE_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)


# Create Tavily tool functions
def search_medical(query: str) -> dict:
    """Search medical information from trusted sources."""
    normalized_query = query.strip()
    if not normalized_query:
        return {"status": "error", "message": "Query must not be empty."}

    try:
        logger.info("Calling Tavily medical search with query: %s", normalized_query)
        response = tavily.search(
            f"medical {normalized_query} site:mayoclinic.org OR site:nih.gov OR site:cdc.gov",
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        sources = [r.get('url') for r in response.get('results', [])]
        logger.info(
            "Tavily medical search completed | sources=%d | answer_snippet=%s",
            len(sources),
            (response.get('answer') or '')[:200]
        )
        return {
            "status": "success",
            "answer": response.get('answer', ''),
            "sources": sources
        }
    except Exception as e:
        logger.exception("Tavily medical search failed")
        return {"status": "error", "message": str(e)}


def check_drug_interactions(drugs: list[str]) -> dict:
    """Check drug interactions between medications."""
    drug_list: list[str] = [drug.strip() for drug in drugs if drug and drug.strip()]
    if not drug_list:
        return {"status": "error", "message": "At least one drug name is required."}

    try:
        query = f"drug interactions {' '.join(drug_list)} FDA warnings"
        logger.info("Calling Tavily drug interaction search with drugs: %s", ", ".join(drug_list))
        response = tavily.search(query, include_answer=True)
        logger.info(
            "Tavily drug interaction search completed | answer_snippet=%s",
            (response.get('answer') or '')[:200]
        )
        return {
            "status": "success",
            "drugs": drug_list,
            "interactions": response.get('answer', 'No information found')
        }
    except Exception as e:
        logger.exception("Tavily drug interaction search failed")
        return {"status": "error", "message": str(e)}


# Constants
AGENT_NAME = "medical_assistant"
APP_NAME = "medical_app"
USER_ID = "user1234"
SESSION_ID = "session_medical"
GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_QUERY = "In 2025, what are the number of people in UAE that's facing mental illness?"

# Create the agent
medical_agent = LlmAgent(
    name=AGENT_NAME,
    model=GEMINI_MODEL,
    instruction="""You are a medical research assistant. Use the tools to search 
    trusted medical sources. Always remind users this is educational only and to 
    consult healthcare providers for medical advice.""",
    description="Medical research assistant with Tavily search",
    tools=[search_medical, check_drug_interactions]
)

# Session and Runner setup
session_service = InMemorySessionService()

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)

runner = Runner(
    agent=medical_agent,
    app_name=APP_NAME,
    session_service=session_service
)


def _format_tool_section(title: str, payload: dict) -> str:
    status = payload.get("status", "unknown")
    section_lines = [f"[{title}] status={status}"]
    if status == "success":
        if title == "search_medical":
            answer = payload.get("answer") or ""
            sources = payload.get("sources") or []
            section_lines.append(f"Answer: {answer[:400]}")
            if sources:
                section_lines.append("Sources:")
                section_lines.extend(f"- {src}" for src in sources)
        elif title == "check_drug_interactions":
            drugs = payload.get("drugs") or []
            section_lines.append(f"Drugs: {', '.join(drugs)}")
            section_lines.append(f"Interactions: {payload.get('interactions', '')[:400]}")
    else:
        section_lines.append(f"Message: {payload.get('message', 'No details')}" )
    return "\n".join(section_lines)


def gather_tavily_context(query: str, drugs: Optional[List[str]] = None) -> tuple[dict, str]:
    """Always invoke Tavily tools and build a summary block for the LLM."""
    tool_results: dict[str, dict] = {}

    search_result = search_medical(query)
    tool_results["search_medical"] = search_result

    interaction_result: Optional[dict] = None
    cleaned_drugs = [drug for drug in (drugs or []) if drug.strip()]
    if cleaned_drugs:
        interaction_result = check_drug_interactions(cleaned_drugs)
        tool_results["check_drug_interactions"] = interaction_result

    summary_sections = [
        _format_tool_section("search_medical", search_result)
    ]
    if interaction_result is not None:
        summary_sections.append(_format_tool_section("check_drug_interactions", interaction_result))

    summary_text = "Tavily tool summary:\n" + "\n\n".join(summary_sections)
    logger.info("Prepared Tavily context for agent:\n%s", summary_text)
    return tool_results, summary_text


# Agent Interaction
async def ask_agent_async(query: str, drugs: Optional[List[str]] = None) -> str:
    """Call the agent with a query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query must not be empty.")

    _, tavily_summary = gather_tavily_context(normalized_query, drugs)

    content = types.Content(
        role="user",
        parts=[
            types.Part(text=normalized_query),
            types.Part(text=tavily_summary)
        ]
    )
    logger.info("Dispatching query to agent: %s", normalized_query)

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            logger.debug("Runner event received: %s", event)
            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    response = event.content.parts[0].text.strip()
                    logger.info("Agent final response snippet: %s", response[:200])
                    return response
    except Exception as e:
        logger.exception("Agent run failed")
        return f"Error: {e}"

    return "No response generated"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running the agent manually."""
    parser = argparse.ArgumentParser(description="Run the healthcare research agent")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question to ask the agent (defaults to a sample medical research query)."
    )
    parser.add_argument(
        "--drugs",
        nargs="*",
        default=None,
        help="Optional list of drug names to explicitly check for interactions."
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    result = await ask_agent_async(args.query, args.drugs)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
