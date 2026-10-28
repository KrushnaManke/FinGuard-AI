"""Optional evidence-constrained LLM drafting."""

import os
import re
import time

from dotenv import load_dotenv

load_dotenv()


def validate_citations(text: str, sources: list[dict]) -> dict:
    allowed = {source["source"] for source in sources}
    citations = re.findall(r"\[([^\[\]]+)\]", text)
    invalid = sorted({citation for citation in citations if citation not in allowed})
    return {
        "citations_found": citations,
        "invalid_citations": invalid,
        "valid": bool(citations) and not invalid,
        "human_verification_required": True,
    }


def draft_investigation(investigation: dict) -> dict:
    """Draft a cited analyst summary; never return a decision."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to enable optional LLM drafting.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the LLM extra with `pip install -e '.[llm]'`.") from error

    if not investigation["sources"]:
        raise RuntimeError("Cannot ask the LLM for a policy-grounded draft without a retrieved source.")
    sources = "\n\n".join(
        f"[{item['source']}]\n{item['excerpt']}" for item in investigation["sources"]
    )
    evidence = "\n".join(
        f"- [{item['source_type']}:{item['source_ref']}] {item['claim']}"
        for item in investigation["evidence"]
    )
    prompt = (
        "Write a concise draft for a financial analyst. Use only the evidence below. "
        "Do not infer customer, merchant, location, or policy facts. Cite retrieved documents "
        "by their exact source filename. If no source supports a policy statement, say so. "
        "This is not a decision; finish by stating that analyst review is required.\n\n"
        f"Investigation status:\n{investigation['draft']}\n\nEvidence ledger:\n{evidence}\n\n"
        f"Retrieved documents:\n{sources}"
    )
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    started = time.perf_counter()
    response = OpenAI().responses.create(
        model=model_name,
        input=prompt,
    )
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    input_rate = os.getenv("FINGUARD_LLM_INPUT_COST_PER_1M") or None
    output_rate = os.getenv("FINGUARD_LLM_OUTPUT_COST_PER_1M") or None
    estimated_cost = None
    if input_rate is not None and output_rate is not None:
        try:
            input_price, output_price = float(input_rate), float(output_rate)
        except ValueError as error:
            raise ValueError("LLM token price environment values must be numeric.") from error
        if input_price < 0 or output_price < 0:
            raise ValueError("LLM token prices cannot be negative.")
        estimated_cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    text = response.output_text
    return {
        "text": text,
        "model": model_name,
        "latency_seconds": time.perf_counter() - started,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost,
        "citation_validation": validate_citations(text, investigation["sources"]),
        "human_review_required": True,
    }
