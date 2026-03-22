import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


@dataclass(frozen=True)
class RunContext:
    """
    Pipeline 실행에 필요한 공통 환경 정보
    """

    run_id: str
    asof_date: str
    universe: str
    project_root: Path
    artifact_root: Path
    flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def logger(self):
        logger = logging.getLogger(f"pipeline.{self.run_id}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger.propagate = False

        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def create(project_root: Path, asof_date: str, universe: str):
        """
        RunContext 생성 helper
        """
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

        artifact_root = project_root / "artifacts" / f"run_id={run_id}"
        artifact_root.mkdir(parents=True, exist_ok=True)

        return RunContext(
            run_id=run_id,
            asof_date=asof_date,
            universe=universe,
            project_root=project_root,
            artifact_root=artifact_root,
        )