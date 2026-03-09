from abc import ABC, abstractmethod
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths
import json


class BaseAgent(ABC):
    """
    모든 Agent의 기본 클래스
    """

    stage: str

    @abstractmethod
    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        """
        agent 실행 로직
        """
        pass

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        """
        agent 실행 wrapper
        """
        try:
            result = self.run(ctx, ap)

        except Exception as e:
            result = StageResult.failed(self.stage, e)

        # 결과 기록
        stage_dir = ap.stage_dir(self.stage)
        result_path = stage_dir / "stage_result.json"

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, indent=2, ensure_ascii=False)

        return result