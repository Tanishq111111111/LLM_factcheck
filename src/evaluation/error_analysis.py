from dataclasses import dataclass

from src.evaluation.labels import CorrectnessLabel, RiskLabel


@dataclass
class ErrorRecord:
    question_id: str
    question: str
    prediction: str
    gold_answer: str
    correctness: CorrectnessLabel
    risk: RiskLabel
    error_type: str
    notes: str = ""
