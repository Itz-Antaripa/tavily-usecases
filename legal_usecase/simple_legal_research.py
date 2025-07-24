"""
legal_research_pipeline.py
---------------------------------
1.  Ask a plain-English legal question
2.  Fetch top-ranked sources with Tavily
3.  Pass sources + question to an LLM to draft a memo
4.  Stream everything to Quotient so the
    • Relevance evaluator filters noisy docs
    • Hallucination / citation checks run automatically

Prerequisites:
pip install tavily-python quotientai openai tiktoken

export TAVILY_API_KEY=tvly-xxx
export OPENAI_API_KEY=sk-xxx
export QUOTIENT_API_KEY=qta-xxx
"""

import os
import textwrap
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI
from quotientai import QuotientAI, DetectionType

load_dotenv()

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

quotient = QuotientAI()

quotient.logger.init(
    app_name="legal-research-demo",
    environment="dev",
    detections=[DetectionType.HALLUCINATION,
                DetectionType.DOCUMENT_RELEVANCY],
    detection_sample_rate=1.0,
    tags={"feature": "legal-research"},
)


def legal_research(legal_query: str, max_results: int = 10):
    # Real-time web search
    tavily_resp = tavily.search(
        legal_query,
        max_results=max_results,
        search_depth="advanced",  # uses web & academic/legal sources
        include_citations=True,
    )

    docs = [
        f"[{i + 1}] {hit['title']} — {hit['url']}\n{hit['content']}"
        for i, hit in enumerate(tavily_resp["results"])
    ]

    # Prompt the LLM
    context = "\n\n".join(docs)
    system = ("You are a senior associate. Draft a concise legal memo that "
              "answers the user’s question **using only the Sources**. "
              "Every assertion must cite the supporting source in square "
              "brackets, e.g. [3]. If the sources do not cover the issue, "
              "say so explicitly.")
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": f"Question: {legal_query}\n\nSources:\n{context}"}
    ]

    llm_resp = llm.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    draft = llm_resp.choices[0].message.content.strip()

    # Log + evaluate with Quotient
    log_id = quotient.log(
        user_query=legal_query,
        model_output=draft,
        documents=docs,
        message_history=messages,
    )
    detection = quotient.poll_for_detection(log_id=log_id)

    return docs, draft, detection


# Run Example
if __name__ == "__main__":
    question = (
        "What is the current standard for granting summary judgment under "
        "Rule 56 of the U.S. Federal Rules of Civil Procedure?"
    )
    web_response, memo, evaluation = legal_research(question)
    print(f"\n--- Web Response ---\n{web_response}")
    print("\n--- Draft Memo ---\n")
    print(textwrap.indent(memo, "  "))
    print(f"\n--- Quotient Evaluation ---\n{evaluation}")
