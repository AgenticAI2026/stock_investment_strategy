from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo


def make_run_root(base_dir: str = "artifacts") -> Path:
    run_id = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")
    root = Path(base_dir) / f"run_id={run_id}"
    root.mkdir(parents=True, exist_ok=True)
    return root


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

    def prep_dir(self) -> Path:
        return self.stage_dir("prep")

    def model_dir(self) -> Path:
        return self.stage_dir("model")

    def inference_dir(self) -> Path:
        return self.stage_dir("inference")

    def report_dir(self) -> Path:
        return self.stage_dir("report")

    # =================================================
    # ✅ INGEST OUTPUTS (핵심)
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