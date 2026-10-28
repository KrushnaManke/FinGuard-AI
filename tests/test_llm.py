from finguard.llm import validate_citations


def test_llm_citation_validation_accepts_only_retrieved_source_names():
    sources = [{"source": "policy.md"}]

    assert validate_citations("Analyst review required [policy.md].", sources)["valid"] is True
    result = validate_citations("The rule is in [made_up_policy.md].", sources)
    assert result["valid"] is False
    assert result["invalid_citations"] == ["made_up_policy.md"]


def test_llm_citation_validation_requires_a_citation():
    result = validate_citations("Analyst review required.", [{"source": "policy.md"}])

    assert result["valid"] is False
    assert result["human_verification_required"] is True
