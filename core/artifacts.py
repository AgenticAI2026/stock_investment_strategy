from pathlib import Path

class ArtifactPaths:

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # base helpers
    # -------------------------------------------------

    def stage_dir(self, stage: str) -> Path:
        path = self.root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sub_dir(self, stage: str, sub: str) -> Path:
        path = self.stage_dir(stage) / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file(self, stage: str, filename: str) -> Path:
        path = self.stage_dir(stage)
        return path / filename

    def file_in(self, stage: str, sub: str, filename: str) -> Path:
        path = self.sub_dir(stage, sub)
        return path / filename

    # -------------------------------------------------
    # Stage dirs
    # -------------------------------------------------

    def ingest_dir(self) -> Path:
        return self.stage_dir("ingest")

    def feature_table_dir(self) -> Path:
        return self.stage_dir("feature_table")

    def prep_reco_dir(self) -> Path:
        return self.stage_dir("prep_reco")

    def model_dir(self) -> Path:
        return self.stage_dir("model")

    def inference_dir(self) -> Path:
        return self.stage_dir("inference")

    def report_dir(self) -> Path:
        return self.stage_dir("report")

    # =================================================
    # INGEST OUTPUTS
    # =================================================

    # --- PRICE (single file)
    def price_ohlcv(self) -> Path:
        return self.file_in("ingest", "price", "price_ohlcv.csv")

    def price_foreign(self) -> Path:
        return self.file_in("ingest", "price", "price_foreign.csv")

    # --- USER (single file)
    def user_profile(self) -> Path:
        return self.file_in("ingest", "user", "user_profile.csv")

    def user_event_log(self) -> Path:
        return self.file_in("ingest", "user", "user_event_log.csv")

    # --- FINANCE (multi file dir)
    def finance_dir(self) -> Path:
        return self.sub_dir("ingest", "finance")

    # --- NEWS (multi file dir)
    def news_dir(self) -> Path:
        return self.sub_dir("ingest", "news")

    # =================================================
    # FEATURE OUTPUTS
    # =================================================

    def price_features(self) -> Path:
        return self.file_in("feature_table", "price", "price_features.csv")

    def finance_features_dir(self) -> Path:
        return self.sub_dir("feature_table", "finance")

    def news_features(self) -> Path:
        return self.file_in("feature_table", "news", "news_features.csv")

    def user_features(self) -> Path:
        return self.file_in("feature_table", "user", "user_features.csv")

    def feature_table(self) -> Path:
        return self.file("feature_table", "feature_table.csv")

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------

    def preprocessing_plan(self) -> Path:
        return self.file("prep_reco", "preprocessing_plan.json")
    
    def data_contract(self) -> Path:
        return self.file("prep_reco", "data_contract.json")

    def preprocessing_report(self) -> Path:
        return self.file("prep_reco", "preprocessing_report.md")


    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    def model_selection_v1(self) -> Path:
        return self.file("model", "model_selection_v1.json")

    def model_selection_v2(self) -> Path:
        return self.file("model", "model_selection_v2.json")


    # -------------------------------------------------
    # Inference
    # -------------------------------------------------

    def inference_signals(self) -> Path:
        return self.file("inference", "signals.csv")


    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    def report(self) -> Path:
        return self.file("report", "report.md")

    def validation_result(self) -> Path:
        return self.file("report", "validation.json")

    def shortform_script(self) -> Path:
        return self.file("report", "shortform.txt")