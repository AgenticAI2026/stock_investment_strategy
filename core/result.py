from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class StageResult:
    """
    Agent 실행 결과 기록
    """

    stage: str
    status: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @staticmethod
    def success(stage: str, inputs=None, outputs=None, metrics=None):
        return StageResult(
            stage=stage,
            status="success",
            inputs=inputs or [],
            outputs=outputs or [],
            metrics=metrics or {},
        )

    @staticmethod
    def failed(stage: str, error: Exception):
        return StageResult(
            stage=stage,
            status="failed",
            error=str(error),
        )