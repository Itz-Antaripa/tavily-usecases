"""Healthcare research agent built on Google ADK + Tavily."""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tavily import TavilyClient

load_dotenv()


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
        response = tavily.search(
            f"medical {normalized_query} site:mayoclinic.org OR site:nih.gov OR site:cdc.gov",
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        return {
            "status": "success",
            "answer": response.get('answer', ''),
            "sources": [r.get('url') for r in response.get('results', [])]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_drug_interactions(drugs: list[str]) -> dict:
    """Check drug interactions between medications."""
    drug_list: list[str] = [drug.strip() for drug in drugs if drug and drug.strip()]
    if not drug_list:
        return {"status": "error", "message": "At least one drug name is required."}

    try:
        query = f"drug interactions {' '.join(drug_list)} FDA warnings"
        response = tavily.search(query, include_answer=True)
        return {
            "status": "success",
            "drugs": drug_list,
            "interactions": response.get('answer', 'No information found')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Constants
AGENT_NAME = "medical_assistant"
APP_NAME = "medical_app"
USER_ID = "user1234"
SESSION_ID = "session_medical"
GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_QUERY = "What are the latest breakthroughs in treating Alzheimer's disease?"

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


# Agent Interaction
async def ask_agent_async(query: str) -> str:
    """Call the agent with a query."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query must not be empty.")

    content = types.Content(role="user", parts=[types.Part(text=normalized_query)])
    print(f"\nQuery: {normalized_query}")

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    response = event.content.parts[0].text.strip()
                    print(f"Response: {response[:500]}...")
                    return response
    except Exception as e:
        print(f"Error: {e}")
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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    result = await ask_agent_async(args.query)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
