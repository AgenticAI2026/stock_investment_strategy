from pathlib import Path


class ArtifactPaths:
    """
    pipeline artifact 경로 관리
    """

    def __init__(self, root: Path):
        self.root = root

    def stage_dir(self, stage: str) -> Path:
        """
        stage별 폴더 생성
        """
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -------- stage별 대표 파일 --------

    def feature_table(self):
        return self.stage_dir("feature_table") / "feature_table.parquet"

    def preprocessing_plan(self):
        return self.stage_dir("prep_reco") / "preprocessing_plan.json"

    def model_selection_v1(self):
        return self.stage_dir("model_match_v1") / "model_selection.json"

    def model_selection_v2(self):
        return self.stage_dir("model_match_v2") / "model_selection.json"

    def inference_signals(self):
        return self.stage_dir("infer") / "signals.parquet"

    def report_md(self):
        return self.stage_dir("report_gen") / "report.md"

    def validation_result(self):
        return self.stage_dir("report_validate") / "validation.json"

    def shortform_script(self):
        return self.stage_dir("shortform_rag") / "shortform.txt"