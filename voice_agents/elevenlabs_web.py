"""
Simple Voice Agent with powered by real-time web-search
Uses Tavily for web search, ElevenLabs for speech and OpenAI Agents SDK for building the agent.

Requirements:
- pip install openai-agents tavily-python elevenlabs
- Set API keys in environment variables (TAVILY_API_KEY, ELEVENLABS_API_KEY, and OPENAI_API_KEY)
"""
from tavily import TavilyClient
from agents import Agent, Runner
from agents import function_tool
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


@function_tool
def web_search(query: str) -> str:
    """Search the web using Tavily"""
    try:
        results = tavily.search(query=query, max_results=4)
        return f"Search results: {results}"
    except Exception as e:
        return f"Search failed: {str(e)}"


def voice_response(text):
    """Convert text to speech using ElevenLabs"""
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    play(audio)


# Create agent with tools
agent = Agent(
    name="Voice Web Assistant",
    instructions="""You are a helpful voice assistant that can search the web and speak responses.

    When a user asks a question:
    1. Use web_search to find current information
    2. Provide a clear, concise answer based on the search results

    Keep responses conversational and under 2-3 sentences.""",
    tools=[web_search]
)


def main():
    """Ask the agent a question and get a voice response"""
    question = "Give me updates about the current Wimbledon 2025?"

    result = Runner.run_sync(agent, question)
    print(f"Response: {result.final_output}")

    voice_response(result.final_output)


if __name__ == "__main__":
    main()
