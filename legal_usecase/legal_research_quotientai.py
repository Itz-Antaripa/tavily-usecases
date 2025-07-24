"""
1.  Ask a plain-English legal question
2.  Fetch top-ranked legal search with citations using Tavily
3.  Pass sources + question to OpenAI to draft a memo
4.  Stream everything to Quotient AI so the
    • Relevance evaluator filters noisy docs
    • Hallucination / citation checks run automatically

Prerequisites:
pip install tavily-python quotientai openai tiktoken

export TAVILY_API_KEY=tvly-xxx
export OPENAI_API_KEY=sk-xxx
export QUOTIENT_API_KEY=qta-xxx
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum

from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI
from quotientai import QuotientAI, DetectionType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Initialize clients
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
quotient = QuotientAI()

# Initialize Quotient logger for legal research
quotient.logger.init(
    app_name="legal-research-workflow",
    environment="production",
    detections=[
        DetectionType.HALLUCINATION,
        DetectionType.DOCUMENT_RELEVANCY
    ],
    detection_sample_rate=1.0,
    tags={"feature": "legal-research"}
)


# Search for legal content using Tavily with legal optimization
def search_legal_content(query: str, max_results: int = 10):
    """
    Args:
        query: Legal question
        max_results: Maximum number of results to return from Tavily

    Returns:
        List of legal sources with content and citations
    """
    logger.info(f"Searching legal content for: {query}")

    # Enhanced query for legal research
    legal_query = f"{query} federal law legal statute case precedent regulation"

    try:
        response = tavily.search(
            query=legal_query,
            max_results=max_results,
            search_depth="advanced",  # Uses academic/legal sources
            include_citations=True,
            include_raw_content=True,
            include_domains=[
                "law.cornell.edu",
                "supreme.justia.com",
                "caselaw.findlaw.com",
                "courtlistener.com",
                "ecfr.gov",
                "federalregister.gov"
            ]
        )

        # Format documents for legal analysis in quotient evaluation
        docs = [
            f"[{i + 1}] {hit['title']}: {hit['url']}\n{hit['content']}"
            for i, hit in enumerate(response["results"])
        ]

        logger.info(f"Found {len(docs)} legal sources")
        return docs

    except Exception as e:
        logger.error(f"Error in legal search: {e}")
        return []


# Generate legal memorandum using verified sources
def generate_legal_memo(query: str, docs: List[str]):
    """
    Args:
        query: Original legal question
        docs: Legal sources from Tavily search

    Returns:
        Tuple of (memo_content, message_history)
    """
    logger.info(f"Generating legal memo with {len(docs)} sources")

    if not docs:
        return "No legal sources found to generate memorandum.", []

    # Prepare context and messages
    context = "\n\n".join(docs)

    system_prompt = """You are a senior legal associate drafting a professional legal memorandum.

REQUIREMENTS:
- Answer the legal question using ONLY the provided sources
- Every legal assertion must cite the supporting source in square brackets [1], [2], etc.
- Use proper legal memorandum format
- If sources don't fully address the issue, state this explicitly
- Be precise and avoid speculation
- Include relevant case law, statutes, and regulations when available

FORMAT:
MEMORANDUM

TO: Client
FROM: Legal Research Team
DATE: [Current Date]
RE: [Legal Question]

EXECUTIVE SUMMARY
[Brief 2-3 sentence answer]

LEGAL ANALYSIS
[Detailed analysis with citations]

CONCLUSION
[Clear conclusion addressing the question]
"""

    user_prompt = f"""Question: {query}

Legal Sources:
{context}

Please draft a comprehensive legal memorandum addressing this question."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Low temperature for factual accuracy
        )

        memo_content = response.choices[0].message.content.strip()

        # Add current date to memo
        current_date = datetime.now().strftime("%B %d, %Y")
        memo_content = memo_content.replace("[Current Date]", current_date)
        memo_content = memo_content.replace("[Legal Question]", query)

        return memo_content, messages

    except Exception as e:
        logger.error(f"Error generating memo: {e}")
        return f"Error generating legal memorandum: {str(e)}", messages


# Log and evaluate with Quotient AI
def evaluate_with_quotient(
        query: str,
        memo_content: str,
        docs: List[str],
        messages: List[Dict[str, Any]]
):
    """Args:
        query: Original legal question
        memo_content: Generated legal memo
        docs: Source documents
        messages: Message history for context

    Returns:
        Quotient detection results
    """
    logger.info("Logging and evaluating with Quotient")

    try:
        # Log the interaction with Quotient
        log_id = quotient.log(
            user_query=query,
            model_output=memo_content,
            documents=docs,
            message_history=messages
        )

        # Poll for detection results
        detection_result = quotient.poll_for_detection(log_id=log_id)

        logger.info(f"Quotient evaluation completed for log_id: {log_id}")
        return detection_result

    except Exception as e:
        logger.error(f"Error with Quotient evaluation: {e}")
        return {"error": str(e), "status": "failed"}


def calculate_confidence(detection_result, num_sources: int) -> ConfidenceLevel:
    """Calculate confidence level based on Quotient Log object and source count"""

    #  Handle error case
    if isinstance(detection_result, dict) and detection_result.get("error"):
        return ConfidenceLevel.LOW

    # Work directly with Log object attributes
    has_hallucination = getattr(detection_result, 'has_hallucination', True)  # Default to True for safety
    evaluations = getattr(detection_result, 'evaluations', [])

    # Calculate accuracy from evaluations
    if evaluations:
        clean_sentences = sum(1 for eval in evaluations if not getattr(eval, 'is_hallucinated', True))
        accuracy_rate = clean_sentences / len(evaluations)
    else:
        accuracy_rate = 0.0

    # High confidence: No hallucinations + high accuracy + sufficient sources
    if not has_hallucination and accuracy_rate >= 0.9 and num_sources >= 5:
        return ConfidenceLevel.HIGH
    # Medium confidence: No hallucinations + good accuracy + some sources
    elif not has_hallucination and accuracy_rate >= 0.7 and num_sources >= 3:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# Main legal research workflow
def conduct_legal_research(query: str, max_results: int = 10):
    logger.info(f"Starting legal research workflow: {query}")
    start_time = datetime.now()

    try:
        # Search legal content with Tavily
        docs = search_legal_content(query, max_results)
        if not docs:
            return {
                "query": query,
                "error": "No legal sources found",
                "confidence": ConfidenceLevel.LOW.value,
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
        print("Tavily sources: ", docs)

        # Generate legal memo
        memo_content, messages = generate_legal_memo(query, docs)

        # Evaluate with Quotient
        detection_result = evaluate_with_quotient(query, memo_content, docs, messages)

        # Calculate confidence
        confidence = calculate_confidence(detection_result, len(docs))

        # Compile results
        duration = (datetime.now() - start_time).total_seconds()

        results = {
            "query": query,
            "memo": memo_content,
            "sources": docs,
            "quotient_evaluation": detection_result,
            "confidence_level": confidence.value,
            "total_sources": len(docs),
            "generated_at": start_time.isoformat(),
            "duration_seconds": duration
        }

        logger.info(f"Legal research completed in {duration:.2f}s with {confidence.value} confidence")
        return results

    except Exception as e:
        logger.error(f"Error in legal research workflow: {e}")
        return {
            "query": query,
            "error": str(e),
            "confidence": ConfidenceLevel.LOW.value,
            "duration_seconds": (datetime.now() - start_time).total_seconds()
        }


if __name__ == "__main__":
    # Run single example
    question = "What are the requirements for establishing personal jurisdiction in federal court?"

    try:
        results = conduct_legal_research(question)
        print(results)

    except Exception as e:
        print(f"Error: {e}")