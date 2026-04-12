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
        return self.stage_dir(stage) / filename

    def file_in(self, stage: str, sub: str, filename: str) -> Path:
        return self.sub_dir(stage, sub) / filename

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

    # --- PRICE
    def price_ohlcv(self) -> Path:
        return self.file_in("ingest", "price", "price_ohlcv.csv")

    def price_foreign(self) -> Path:
        return self.file_in("ingest", "price", "price_foreign.csv")

    def ingest_price_dir(self) -> Path:
        return self.sub_dir("ingest", "price")

    # --- USER
    def user_profile(self) -> Path:
        return self.file_in("ingest", "user", "user_profile.csv")

    def user_event_log(self) -> Path:
        return self.file_in("ingest", "user", "user_event_log.csv")

    def ingest_user_dir(self) -> Path:
        return self.sub_dir("ingest", "user")

    # --- FINANCE
    def ingest_finance_dir(self) -> Path:
        return self.sub_dir("ingest", "finance")

    # --- NEWS
    def ingest_news_dir(self) -> Path:
        return self.sub_dir("ingest", "news")

    # =================================================
    # FEATURE TABLE OUTPUTS
    # =================================================

    # --- USER
    def feature_user_dir(self) -> Path:
        return self.sub_dir("feature_table", "user")

    def user_features(self) -> Path:
        return self.file_in("feature_table", "user", "user_features.csv")

    def user_feature_dictionary(self) -> Path:
        return self.file_in("feature_table", "user", "user_feature_dictionary.csv")

    # --- PRICE
    def feature_price_dir(self) -> Path:
        return self.sub_dir("feature_table", "price")

    def price_ohlcv_features(self) -> Path:
        return self.file_in("feature_table", "price", "price_ohlcv_features.csv")

    def price_foreign_features(self) -> Path:
        return self.file_in("feature_table", "price", "price_foreign_features.csv")

    def price_feature_dictionary(self) -> Path:
        return self.file_in("feature_table", "price", "price_feature_dictionary.csv")

    # --- FINANCE
    def feature_finance_dir(self) -> Path:
        return self.sub_dir("feature_table", "finance")

    # finance는 종목별 여러 파일이므로 directory 위주
    def finance_feature_dictionary(self) -> Path:
        return self.file_in("feature_table", "finance", "finance_feature_dictionary.csv")

    # --- NEWS
    def feature_news_dir(self) -> Path:
        return self.sub_dir("feature_table", "news")

    def news_features(self) -> Path:
        return self.file_in("feature_table", "news", "news_features.csv")

    def news_raw(self) -> Path:
        return self.file_in("feature_table", "news", "news_raw.csv")

    def news_feature_dictionary(self) -> Path:
        return self.file_in("feature_table", "news", "news_feature_dictionary.csv")

    def news_raw_dictionary(self) -> Path:
        return self.file_in("feature_table", "news", "news_raw_dictionary.csv")

    # =================================================
    # PREPROCESSING
    # =================================================

    def preprocessing_plan(self) -> Path:
        return self.file("prep_reco", "preprocessing_plan.json")

    def data_contract(self) -> Path:
        return self.file("prep_reco", "data_contract.json")

    def preprocessing_report(self) -> Path:
        return self.file("prep_reco", "preprocessing_report.md")

    # =================================================
    # MODEL
    # =================================================

    def model_matching_result(self) -> Path:
        return self.file("model", "model_matching_result.json")

    def model_selection_v1(self) -> Path:
        return self.file("model", "model_selection_v1.json")

    def model_selection_v2(self) -> Path:
        return self.file("model", "model_selection_v2.json")

    # =================================================
    # INFERENCE
    # =================================================

    def inference_signals(self) -> Path:
        return self.file("inference", "signals.csv")

    # =================================================
    # REPORT
    # =================================================

    def report(self) -> Path:
        return self.file("report", "report.md")

    def validation_result(self) -> Path:
        return self.file("report", "validation.json")

    def shortform_script(self) -> Path:
        return self.file("report", "shortform.txt")