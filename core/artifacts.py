from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo


def make_run_root(base_dir: str = "artifacts") -> Path:
    """
    실행할 때마다 고유한 run_id 폴더 생성
    예: artifacts/run_id=20260311T153000
    """
    run_id = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")
    root = Path(base_dir) / f"run_id={run_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


class ArtifactPaths:
    """
    pipeline artifact 경로 관리
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        """
        stage별 폴더 생성
        """
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file(self, stage: str, filename: str) -> Path:
        """
        stage 내부 파일 경로 반환
        """
        return self.stage_dir(stage) / filename

    # -------- 공통 stage 폴더 --------

    def ingest_dir(self) -> Path:
        return self.stage_dir("ingest")

    def feature_table_dir(self) -> Path:
        return self.stage_dir("feature_table")

    def prep_dir(self) -> Path:
        return self.stage_dir("prep")

    def model_dir(self) -> Path:
        return self.stage_dir("model")

    def inference_dir(self) -> Path:
        return self.stage_dir("inference")

    def report_dir(self) -> Path:
        return self.stage_dir("report")

    # -------- stage별 대표 파일 --------

    def feature_table(self) -> Path:
        return self.file("feature_table", "feature_table.parquet")

    def preprocessing_plan(self) -> Path:
        return self.file("prep", "preprocessing_plan.json")

    def model_selection_v1(self) -> Path:
        return self.file("model", "model_selection_v1.json")

    def model_selection_v2(self) -> Path:
        return self.file("model", "model_selection_v2.json")

    def inference_signals(self) -> Path:
        return self.file("inference", "signals.parquet")

    def report_md(self) -> Path:
        return self.file("report", "report.md")

    def validation_result(self) -> Path:
        return self.file("report", "validation.json")

    def shortform_script(self) -> Path:
        return self.file("report", "shortform.txt")