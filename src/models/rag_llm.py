from dataclasses import dataclass


@dataclass
class RagLLMConfig:
    model_name: str
    system_prompt: str


def build_rag_prompt(question: str, evidence_passages: list[str], config: RagLLMConfig) -> list[dict[str, str]]:
    joined_evidence = "\n\n".join(
        f"Passage {index + 1}: {passage}" for index, passage in enumerate(evidence_passages)
    )
    user_prompt = f"Question: {question}\n\nEvidence:\n{joined_evidence}"
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
