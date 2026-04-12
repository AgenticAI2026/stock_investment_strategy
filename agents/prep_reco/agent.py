from __future__ import annotations

import json
import os
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths

#테이블 1개에 대한 메타데이터(설명서)를 담는 설정 객체
@dataclass
class TableSpec:
    domain: str #user, price, finance, news
    name: str
    path: str
    dictionary_path: Optional[str] = None #정의서 경로
    role: str = "feature_table"  # feature_table | raw_text_reference
    temporal: Optional[bool] = None
    time_col: Optional[str] = None
    entity_key: Optional[List[str]] = None


class PreprocessingRecommenderAgent(BaseAgent):
    stage = "prep_reco"

    _DURATION_OR_SCORE_PAT = re.compile(
        r"(dwell|duration|elapsed|lag|lead|concentration|score|ratio|rate|volatility|zscore|count|freq)",
        re.IGNORECASE,
    )

    _AUDIT_TIME_NAMES = {
        "feature_now_utc", "now_utc", "created_at", "updated_at", "fetched_at"
    }

    _DATE_STR_PAT = re.compile(r"^\s*\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*$")
    _YYYYMMDD_PAT = re.compile(r"^\s*\d{8}\s*$")

    _RFC822_PAT = re.compile(
        r"^[A-Za-z]{3},\s+\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+[+\-]\d{4}$"
    )

    _TIME_NAME_PATT = re.compile(
        r"(?:^|_)(pubdate|pub_date|date|datetime|timestamp|trade_date|fetched_at|created_at|updated_at)(?:$|_)",
        re.I
    )

    _DICT_NOTES_ZERO_DIV_PAT = re.compile(r"(0\s*으로\s*나누|분모\s*0|divide\s*by\s*0|division\s*by\s*zero)", re.I)
    _DICT_NOTES_INF_PAT = re.compile(r"(inf|infinite|무한|무한대)", re.I)
    _DICT_NOTES_MINPER_PAT = re.compile(r"(min[_\s-]*periods\s*=?\s*\d+|최소\s*\d+\s*개|최소\s*기간)", re.I)

    _FORMULA_TOKEN_PAT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[가-힣_]+[가-힣0-9_]*")

    def __init__(
        self,
        base_dir: str = "",
        output_dir: str = "",

        # dictionaries (feature dictionaries)
        user_dict: str = "",
        price_dict: str = "",
        finance_dict: str = "",
        news_dict: str = "",

        news_raw_dict: Optional[str] = None,

        # feature tables
        user_features: str = "",
        price_ohlcv_last365: str = "",
        price_foreign_snapshot_today: str = "",
        finance_features_glob: str = "",
        news_features_by_stock: str = "",
        news_raw_merged: str = "",

        training_intent_path: Optional[str] = None,
        split_hint_path: Optional[str] = None,
        sample_rows: int = 200_000,
        encoding: str = "utf-8",
        random_state: int = 42,

        numeric_code_max_unique: int = 50,
        snapshot_time_parse_threshold: float = 0.95,
    ):
        self.base_dir = base_dir or ""
        self.output_dir = output_dir or ""

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.paths = {
            "user_dict": self._abspath(user_dict),
            "price_dict": self._abspath(price_dict),
            "finance_dict": self._abspath(finance_dict),
            "news_dict": self._abspath(news_dict),

            # raw news schema dict
            "news_raw_dict": self._abspath(news_raw_dict) if news_raw_dict else None,

            "user_features": self._abspath(user_features),
            "price_ohlcv_last365": self._abspath(price_ohlcv_last365),
            "price_foreign_snapshot_today": self._abspath(price_foreign_snapshot_today),
            "finance_features_glob": self._abspath(finance_features_glob, is_pattern=True),
            "news_features_by_stock": self._abspath(news_features_by_stock),
            "news_raw_merged": self._abspath(news_raw_merged),
        }

        self.training_intent_path = self._abspath(training_intent_path) if training_intent_path else None
        self.split_hint_path = self._abspath(split_hint_path) if split_hint_path else None

        self.sample_rows = int(sample_rows)
        self.encoding = encoding
        self.random_state = int(random_state)

        self.numeric_code_max_unique = int(numeric_code_max_unique)
        self.snapshot_time_parse_threshold = float(snapshot_time_parse_threshold)

        self.training_intent = self._load_json_maybe(self.training_intent_path) or {}
        self.split_hint = self._load_json_maybe(self.split_hint_path) or {}

        self.table_specs: List[TableSpec] = []

        # dictionaries loaded raw
        self.dicts: Dict[str, pd.DataFrame] = {}

        # parsed dictionary info
        self.dict_feature_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.dict_column_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.tables: Dict[str, pd.DataFrame] = {}
        self.table_profiles: Dict[str, Dict[str, Any]] = {}
        self.inferred_roles: Dict[str, Dict[str, Any]] = {}
        self.feature_sets: Dict[str, Dict[str, List[str]]] = {}
        self.leakage_findings: Dict[str, List[Dict[str, Any]]] = {}

        self.time_series_diagnostics: Dict[str, Dict[str, Any]] = {}


    # ---------------------------
    # Pipeline Entry
    # ---------------------------

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        try:
            runtime_agent = self if self._is_configured() else self._build_runtime_agent(ctx, ap)
            outputs = runtime_agent.run()

            return StageResult.success(
                stage=self.stage,
                outputs=[
                    outputs["preprocessing_plan"],
                    outputs["data_contract"],
                    outputs["preprocessing_report"],
                ]
            )

        except Exception as e:
            ctx.logger.exception("❌ Preprocessing Recommender Failed")
            return StageResult.failed(
                stage=self.stage,
                error=str(e),
            )
       

    def _is_configured(self) -> bool:
        required = [
            self.base_dir,
            self.output_dir,
            self.paths.get("user_dict"),
            self.paths.get("price_dict"),
            self.paths.get("finance_dict"),
            self.paths.get("news_dict"),
        ]
        return all(bool(x) for x in required)

    def _build_runtime_agent(self, ctx: RunContext, ap: ArtifactPaths) -> "PreprocessingRecommenderAgent":
        feature_dir = Path(ap.feature_table_dir())
        output_dir = Path(ap.prep_reco_dir())

        user_dir = feature_dir / "user"
        price_dir = feature_dir / "price"
        finance_dir = feature_dir / "finance"
        news_dir = feature_dir / "news"

        contract_root = Path(ctx.flags.get("contract_root", feature_dir))

        user_dict_name = ctx.flags.get("user_dict", "user_feature_dictionary.csv")
        price_dict_name = ctx.flags.get("price_dict", "price_feature_dictionary.csv")
        finance_dict_name = ctx.flags.get("finance_dict", "finance_feature_dictionary.csv")
        news_dict_name = ctx.flags.get("news_dict", "news_feature_dictionary.csv")
        news_raw_dict_name = ctx.flags.get("news_raw_dict", "news_raw_dictionary.csv")

        user_features_name = ctx.flags.get("user_features", "user_features.csv")
        price_ohlcv_name = ctx.flags.get("price_ohlcv_last365", "price_ohlcv_features.csv")
        price_foreign_name = ctx.flags.get("price_foreign_snapshot_today", "price_foreign_features.csv")
        finance_glob_name = ctx.flags.get("finance_features_glob", "*_finance_features.csv")
        news_features_name = ctx.flags.get("news_features_by_stock", "news_features.csv")
        news_raw_merged_name = ctx.flags.get("news_raw_merged", "news_raw.csv")

        training_intent_name = ctx.flags.get("training_intent_path")
        split_hint_name = ctx.flags.get("split_hint_path")

        return PreprocessingRecommenderAgent(
            base_dir="",
            output_dir=str(output_dir),

            # dictionaries
            user_dict=str(self._resolve_path(user_dict_name, user_dir)),
            price_dict=str(self._resolve_path(price_dict_name, price_dir)),
            finance_dict=str(self._resolve_path(finance_dict_name, finance_dir)),
            news_dict=str(self._resolve_path(news_dict_name, news_dir)),
            news_raw_dict=(
                str(self._resolve_path(news_raw_dict_name, news_dir))
                if news_raw_dict_name
                else None
            ),

            # feature tables
            user_features=str(self._resolve_path(user_features_name, user_dir)),
            price_ohlcv_last365=str(self._resolve_path(price_ohlcv_name, price_dir)),
            price_foreign_snapshot_today=str(self._resolve_path(price_foreign_name, price_dir)),
            finance_features_glob=str(self._resolve_path(finance_glob_name, finance_dir)),
            news_features_by_stock=str(self._resolve_path(news_features_name, news_dir)),
            news_raw_merged=str(self._resolve_path(news_raw_merged_name, news_dir)),

            # optional external hints/contracts
            training_intent_path=(
                str(self._resolve_path(training_intent_name, contract_root))
                if training_intent_name
                else None
            ),
            split_hint_path=(
                str(self._resolve_path(split_hint_name, contract_root))
                if split_hint_name
                else None
            ),

            sample_rows=int(ctx.flags.get("sample_rows", 200_000)),
            encoding=ctx.flags.get("encoding", "utf-8"),
            random_state=int(ctx.flags.get("random_state", 42)),
            numeric_code_max_unique=int(ctx.flags.get("numeric_code_max_unique", 50)),
            snapshot_time_parse_threshold=float(ctx.flags.get("snapshot_time_parse_threshold", 0.95)),
        )

    def _resolve_path(self, p: Optional[str], root: Path) -> Optional[str]:
        if p is None or p == "":
            return None
        p = Path(p)
        if p.is_absolute():
            return str(p)
        return str(root / p)
    
    # ---------------------------
    # Public API
    # ---------------------------

    def run(self) -> Dict[str, str]:
        self._build_table_specs()
        self._load_all_dictionaries()
        self._parse_all_dictionaries()
        self._load_all_tables()

        self._schema_inference_all()
        self._profile_all_tables()

        plan = self._synthesize_preprocessing_plan_requirements_only()
        contract = self._synthesize_data_contract()
        report = self._synthesize_report_md(plan, contract)

        out_plan = os.path.join(self.output_dir, "preprocessing_plan.json")
        out_contract = os.path.join(self.output_dir, "data_contract.json")
        out_report = os.path.join(self.output_dir, "preprocessing_report.md")

        self._write_json(out_plan, plan)
        self._write_json(out_contract, contract)
        self._write_text(out_report, report)

        return {"preprocessing_plan": out_plan, "data_contract": out_contract, "preprocessing_report": out_report}

    # ---------------------------
    # 데이터셋 정의
    # ---------------------------

    def _build_table_specs(self) -> None:
        raw_news_dict_path = self.paths["news_raw_dict"] or self.paths["news_dict"]

        self.table_specs = [
            TableSpec(
                domain="user",
                name="user_snapshot",
                path=self.paths["user_features"],
                dictionary_path=self.paths["user_dict"],
                temporal=False,
                time_col=None,
                entity_key=self._guess_entity_key(["user_id", "uid"]),
            ),
            TableSpec(
                domain="price",
                name="ohlcv_last365",
                path=self.paths["price_ohlcv_last365"],
                dictionary_path=self.paths["price_dict"],
                temporal=True,
                time_col=None,
                entity_key=self._guess_entity_key(["ticker", "code", "stock_code", "종목코드"]),
            ),
            TableSpec(
                domain="price",
                name="foreign_snapshot_today",
                path=self.paths["price_foreign_snapshot_today"],
                dictionary_path=self.paths["price_dict"],
                temporal=False,  # snapshot
                time_col=None,
                entity_key=self._guess_entity_key(["ticker", "code", "stock_code", "종목코드"]),
            ),
            TableSpec(
                domain="finance",
                name="financial_features",
                path=self.paths["finance_features_glob"],
                dictionary_path=self.paths["finance_dict"],
                temporal=True,
                time_col=None,
                entity_key=self._guess_entity_key(["ticker", "code", "stock_code", "종목코드"]),
            ),
            TableSpec(
                domain="news",
                name="news_features_by_stock",
                path=self.paths["news_features_by_stock"],
                dictionary_path=self.paths["news_dict"],
                temporal=False,
                time_col=None,
                entity_key=self._guess_entity_key(["ticker", "code", "stock_code", "종목코드"]),
            ),
            TableSpec(
                domain="news",
                name="news_raw_merged",
                path=self.paths["news_raw_merged"],
                dictionary_path=raw_news_dict_path,
                role="raw_text_reference",
                temporal=True,
                time_col=None,
                entity_key=self._guess_entity_key(["ticker", "code", "stock_code", "종목코드"]),
            ),
        ]

        overrides = (self.split_hint or {}).get("overrides", {})
        for spec in self.table_specs:
            k = f"{spec.domain}.{spec.name}"
            if k in overrides:
                ov = overrides[k]
                spec.temporal = ov.get("temporal", spec.temporal)
                spec.time_col = ov.get("time_col", spec.time_col)
                spec.entity_key = ov.get("entity_key", spec.entity_key)

    def _guess_entity_key(self, candidates: List[str]) -> Optional[List[str]]:
        return candidates

    # ---------------------------
    # 데이터 로딩 + dictionary 파싱 + 내부 메타데이터 구축
    # ---------------------------

    def _abspath(self, p: Optional[str], is_pattern: bool = False) -> Optional[str]:
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    def _load_json_maybe(self, path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_all_dictionaries(self) -> None:
        # domain-level feature dicts
        for dom, dpath in [
            ("user", self.paths["user_dict"]),
            ("price", self.paths["price_dict"]),
            ("finance", self.paths["finance_dict"]),
            ("news", self.paths["news_dict"]),
        ]:
            self.dicts[dom] = self._read_csv(dpath)

        # table-level raw-news dict
        if self.paths.get("news_raw_dict"):
            self.dicts["news_raw_dict"] = self._read_csv(self.paths["news_raw_dict"])

    def _detect_dictionary_type(self, ddf: pd.DataFrame) -> str:
        cols_lower = {str(c).strip().lower() for c in ddf.columns}
        if "feature_name" in cols_lower:
            return "feature"
        if "column_name" in cols_lower:
            return "column"
        if any(c in cols_lower for c in ["role", "semantic_type"]):
            return "column"
        return "unknown"

    def _parse_feature_dictionary(self, ddf: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        cols_lower_map = {str(c).lower(): c for c in ddf.columns}
        fn_col = cols_lower_map.get("feature_name") or cols_lower_map.get("feature") or cols_lower_map.get("name")
        desc_col = cols_lower_map.get("description_ko") or cols_lower_map.get("description") or cols_lower_map.get("desc")
        formula_col = cols_lower_map.get("formula")
        window_col = cols_lower_map.get("window")
        notes_col = cols_lower_map.get("notes") or cols_lower_map.get("note")
        data_source_col = cols_lower_map.get("data_source") or cols_lower_map.get("source")

        meta = {}
        if not fn_col:
            return meta

        for _, row in ddf.iterrows():
            fn = str(row.get(fn_col, "")).strip()
            if not fn:
                continue
            meta[fn] = {
                "feature_name": fn,
                "description_ko": (str(row.get(desc_col, "")).strip() if desc_col else ""),
                "formula": (str(row.get(formula_col, "")).strip() if formula_col else ""),
                "window": (str(row.get(window_col, "")).strip() if window_col else ""),
                "notes": (str(row.get(notes_col, "")).strip() if notes_col else ""),
                "data_source": (str(row.get(data_source_col, "")).strip() if data_source_col else ""),
            }
        return meta

    def _parse_column_dictionary(self, ddf: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        cols_lower_map = {str(c).lower(): c for c in ddf.columns}
        cn_col = cols_lower_map.get("column_name") or cols_lower_map.get("name")
        desc_col = cols_lower_map.get("description_ko") or cols_lower_map.get("description")
        role_col = cols_lower_map.get("role")
        sem_col = cols_lower_map.get("semantic_type") or cols_lower_map.get("semantic")
        model_use_col = cols_lower_map.get("model_use")
        fmt_col = cols_lower_map.get("format_hint")
        notes_col = cols_lower_map.get("notes") or cols_lower_map.get("note")

        meta = {}
        if not cn_col:
            return meta

        for _, row in ddf.iterrows():
            cn = str(row.get(cn_col, "")).strip()
            if not cn:
                continue
            meta[cn] = {
                "column_name": cn,
                "description_ko": (str(row.get(desc_col, "")).strip() if desc_col else ""),
                "role": (str(row.get(role_col, "")).strip() if role_col else ""),
                "semantic_type": (str(row.get(sem_col, "")).strip() if sem_col else ""),
                "model_use": (row.get(model_use_col) if model_use_col else None),
                "format_hint": (str(row.get(fmt_col, "")).strip() if fmt_col else ""),
                "notes": (str(row.get(notes_col, "")).strip() if notes_col else ""),
            }
        return meta

    def _parse_all_dictionaries(self) -> None:
        """
        모든 dictionary를 통합해서
        → feature_meta
        → column_meta
        로 나눔
        """
        self.dict_feature_meta = {}
        self.dict_column_meta = {}

        # domain feature dicts
        for dom in ["user", "price", "finance", "news"]:
            ddf = self.dicts.get(dom)
            if ddf is None or len(ddf) == 0:
                self.dict_feature_meta[dom] = {}
                continue

            dtype = self._detect_dictionary_type(ddf)
            if dtype == "feature":
                self.dict_feature_meta[dom] = self._parse_feature_dictionary(ddf)
            elif dtype == "column":
                self.dict_feature_meta[dom] = {}
                self.dict_column_meta[dom] = self._parse_column_dictionary(ddf)
            else:
                self.dict_feature_meta[dom] = {}

        # table-level raw news schema dict
        raw_ddf = self.dicts.get("news_raw_dict")
        if raw_ddf is not None and len(raw_ddf) > 0:
            dtype = self._detect_dictionary_type(raw_ddf)
            if dtype == "column":
                self.dict_column_meta["news.news_raw_merged"] = self._parse_column_dictionary(raw_ddf)
            else:
                self.dict_feature_meta.setdefault("news", {})
                self.dict_feature_meta["news"].update(self._parse_feature_dictionary(raw_ddf))

    def _load_all_tables(self) -> None:
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if spec.path is None:
                continue

            if any(ch in spec.path for ch in ["*", "?", "["]):
                files = sorted(glob.glob(spec.path))
                if not files:
                    raise FileNotFoundError(f"No files matched pattern: {spec.path}")
                df = self._read_and_concat_csvs(files, spec=spec)
                self.tables[key] = df
            else:
                self.tables[key] = self._read_csv(spec.path)

    def _read_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        try:
            df = pd.read_csv(path, encoding=self.encoding, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949", low_memory=False)

        return self._maybe_sample(df)

    def _read_and_concat_csvs(self, files: List[str], spec: Optional[TableSpec] = None) -> pd.DataFrame:
        parts = []
        for fp in files:
            df = self._read_csv(fp)
            df["_source_file"] = os.path.basename(fp)

            if spec is not None and spec.domain == "finance":
                ticker = self._extract_ticker_from_finance_filename(fp)
                if ticker is not None:
                    df["종목코드"] = str(ticker)

            parts.append(df)

        out = pd.concat(parts, axis=0, ignore_index=True)
        return self._maybe_sample(out)

    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_rows and len(df) > self.sample_rows:
            df = df.sample(self.sample_rows, random_state=self.random_state)
        return df

    # ---------------------------
    # Schema inference
    # ---------------------------

    def _schema_inference_all(self) -> None:
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue

            df = self.tables[key]
            roles = self._infer_roles(spec, df)
            fsets = self._infer_feature_sets(spec, df, roles)

            self.inferred_roles[key] = roles
            self.feature_sets[key] = fsets

    def _infer_roles(self, spec: TableSpec, df: pd.DataFrame) -> Dict[str, Any]:
        cols = list(df.columns)
        role_map = {
            "target": [],
            "time_col": None,
            "id_cols": [],
            "group_cols": [],
            "ticker_col": None,
            "audit_time_cols": [],
        }

        role_map["ticker_col"] = self._guess_ticker_col(df)

        candidate_keys = spec.entity_key or []
        resolved = [c for c in candidate_keys if c in cols]
        for c in resolved:
            role_map["id_cols"].append(c)

        for c in cols:
            cl = c.lower()
            if cl in ("user_id", "userid", "uid") and c not in role_map["id_cols"]:
                role_map["id_cols"].append(c)
        if not role_map["id_cols"]:
            for c in cols:
                if re.search(r"(?:^|_)(id|uid|uuid|key)(?:$|_)", c.lower()):
                    role_map["id_cols"].append(c)

        if spec.time_col and spec.time_col in cols:
            role_map["time_col"] = spec.time_col
        else:
            if spec.temporal is True:
                role_map["time_col"] = self._guess_time_col_from_df(df, strict=False)
            else:
                role_map["time_col"] = self._guess_time_col_from_df(df, strict=True)

        if role_map["time_col"] is None and spec.domain == "finance":
            if "year" in df.columns and self._is_year_like_numeric(df["year"], "year"):
                role_map["time_col"] = "year"

        audit_candidates = []
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in self._AUDIT_TIME_NAMES and self._is_datetime_like_series(df[c]):
                audit_candidates.append(c)
        role_map["audit_time_cols"] = self._dedup_keep_order(audit_candidates)

        if spec.role == "raw_text_reference":
            if "pubDate" in df.columns and self._is_datetime_like_series(df["pubDate"]):
                role_map["time_col"] = "pubDate"

        if spec.temporal is False and role_map["time_col"] in role_map["audit_time_cols"]:
            role_map["time_col"] = None

        role_map["id_cols"] = self._dedup_keep_order([c for c in role_map["id_cols"] if c in cols])
        role_map["group_cols"] = self._dedup_keep_order([c for c in role_map["group_cols"] if c in cols])
        role_map["target"] = self._dedup_keep_order([c for c in role_map["target"] if c in cols])

        if role_map["time_col"] and role_map["time_col"] not in cols:
            role_map["time_col"] = None
        if role_map["ticker_col"] and role_map["ticker_col"] not in cols:
            role_map["ticker_col"] = None

        return role_map

    def _infer_feature_sets(
        self,
        spec: TableSpec,
        df: pd.DataFrame,
        roles: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        cols = [c for c in df.columns if c not in (roles.get("target") or [])]
        time_col = roles.get("time_col")
        if time_col and time_col in cols:
            cols.remove(time_col)

        id_cols = set(roles.get("id_cols") or [])
        ticker_col = roles.get("ticker_col")
        if ticker_col:
            id_cols.add(ticker_col)

        numeric, categorical, text, datetime, boolean = [], [], [], [], []
        numeric_coded_categorical = []

        for c in cols:
            if c in id_cols:
                continue

            series = df[c]
            cl = str(c).strip().lower()

            if pd.api.types.is_object_dtype(series) and self._TIME_NAME_PATT.search(cl):
                if self._is_datetime_like_series(series):
                    datetime.append(c)
                    continue

            if pd.api.types.is_object_dtype(series):
                if ("utc" in cl or "now" in cl) and self._is_datetime_like_series(series):
                    datetime.append(c)
                    continue

            if pd.api.types.is_bool_dtype(series):
                boolean.append(c)
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime.append(c)
                continue

            if pd.api.types.is_numeric_dtype(series):
                nunique = int(series.nunique(dropna=True))
                if nunique > 1 and nunique <= self.numeric_code_max_unique:
                    x = pd.to_numeric(series, errors="coerce").dropna()
                    if len(x) > 0:
                        xr = x.round()
                        int_like = float((np.abs(x - xr) < 1e-6).mean()) >= 0.98
                        if int_like:
                            numeric_coded_categorical.append(c)
                            categorical.append(c)
                            continue
                numeric.append(c)
                continue

            nunique = int(series.nunique(dropna=True))
            if nunique <= 50:
                categorical.append(c)
            else:
                s = series.dropna().astype(str)
                if len(s) == 0:
                    categorical.append(c)
                else:
                    avg_len = float(s.str.len().mean())
                    if avg_len >= 30:
                        text.append(c)
                    else:
                        categorical.append(c)

        if roles.get("time_col") and roles["time_col"] in df.columns:
            datetime = self._dedup_keep_order(datetime + [roles["time_col"]])

        for c in roles.get("audit_time_cols", []) or []:
            if c in df.columns:
                datetime = self._dedup_keep_order(datetime + [c])

        return {
            "numeric": self._dedup_keep_order(numeric),
            "categorical": self._dedup_keep_order(categorical),
            "text": self._dedup_keep_order(text),
            "datetime": self._dedup_keep_order(datetime),
            "bool": self._dedup_keep_order(boolean),
            "categorical_numeric_codes": self._dedup_keep_order(numeric_coded_categorical),
        }

    def _extract_ticker_from_finance_filename(self, fp: str) -> Optional[str]:
        base = os.path.basename(fp)
        m = re.match(r"^(\d{6})", base)
        if m:
            return m.group(1)
        m2 = re.match(r"^(\d{4,8})", base)
        if m2:
            return m2.group(1)
        return None

    # ---------------------------
    # Time column detection
    # ---------------------------

    def _guess_time_col_from_df(self, df: pd.DataFrame, strict: bool = False) -> Optional[str]:
        cols_lower = {str(c).lower(): c for c in df.columns}

        exact = [
            "date", "datetime", "timestamp", "dt",
            "trade_date", "pubdate", "pub_date",
            "fetched_at",
            "feature_now_utc",
        ]

        for k in exact:
            kl = k.lower()
            if kl in cols_lower:
                c = cols_lower[kl]
                cl = str(c).strip().lower()

                if strict and cl in self._AUDIT_TIME_NAMES:
                    continue
                if strict and cl == "feature_now_utc":
                    continue

                if self._is_time_index_candidate(df[c], c, allow_year=not strict, require_strict_parse=strict):
                    if strict and df[c].nunique(dropna=True) <= 1:
                        continue
                    return c

        patt = re.compile(
            r"(?:^|_)(date|datetime|timestamp|trade_date|pubdate|pub_date|fetched_at)(?:$|_)",
            re.I
        )
        for c in df.columns:
            cl = str(c).strip().lower()
            if strict and cl in self._AUDIT_TIME_NAMES:
                continue
            if strict and cl == "feature_now_utc":
                continue

            if patt.search(cl):
                if self._is_time_index_candidate(df[c], c, allow_year=not strict, require_strict_parse=strict):
                    if strict and df[c].nunique(dropna=True) <= 1:
                        continue
                    return c

        return None

    def _is_datetime_like_series(self, s: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if pd.api.types.is_numeric_dtype(s):
            return False

        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
        success = float(parsed.notna().mean())
        return success >= 0.8

    def _is_time_index_candidate(
        self,
        s: pd.Series,
        col_name: str,
        allow_year: bool = True,
        require_strict_parse: bool = False,
    ) -> bool:
        name_l = str(col_name).lower()
        if "time" in name_l and self._DURATION_OR_SCORE_PAT.search(name_l):
            return False

        if pd.api.types.is_datetime64_any_dtype(s):
            return True

        if pd.api.types.is_numeric_dtype(s):
            if allow_year and self._is_year_like_numeric(s, col_name):
                return True
            if self._is_yyyymmdd_numeric(s):
                return True
            return False

        if pd.api.types.is_object_dtype(s):
            sample = s.dropna().astype(str).head(400)
            if len(sample) == 0:
                return False

            date_like_ratio = float(sample.str.match(self._DATE_STR_PAT).mean())
            yyyymmdd_ratio = float(sample.str.match(self._YYYYMMDD_PAT).mean())
            rfc822_ratio = float(sample.str.match(self._RFC822_PAT).mean())

            if require_strict_parse:
                if max(date_like_ratio, yyyymmdd_ratio, rfc822_ratio) < 0.6:
                    return False
            else:
                if max(date_like_ratio, yyyymmdd_ratio, rfc822_ratio) < 0.3:
                    return False

            parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            success = float(parsed.notna().mean())
            threshold = self.snapshot_time_parse_threshold if require_strict_parse else 0.70
            if success >= threshold:
                p = parsed.dropna()
                if len(p) >= 10 and p.nunique() <= 1:
                    return False
                return True

        return False

    def _is_year_like_numeric(self, s: pd.Series, col_name: str) -> bool:
        name_l = str(col_name).lower()
        if name_l != "year" and not re.search(r"(?:^|_)(year|yyyy)(?:$|_)", name_l):
            return False
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) < 10:
            return False
        xr = x.round()
        if float((np.abs(x - xr) < 1e-6).mean()) < 0.95:
            return False
        mn, mx = float(xr.min()), float(xr.max())
        return 1900 <= mn <= 2100 and 1900 <= mx <= 2100

    def _is_yyyymmdd_numeric(self, s: pd.Series) -> bool:
        x = pd.to_numeric(s, errors="coerce").dropna()
        if len(x) < 10:
            return False
        xr = x.round()
        if float((np.abs(x - xr) < 1e-6).mean()) < 0.95:
            return False
        mn, mx = int(xr.min()), int(xr.max())
        if mn < 19000101 or mx > 21001231:
            return False
        sample = xr.astype(int).astype(str).head(200)
        ok = sample.str.match(r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$").mean()
        return float(ok) >= 0.7

    # ---------------------------
    # Profiling
    # ---------------------------

    def _profile_all_tables(self) -> None:
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue
            df = self.tables[key]
            roles = self.inferred_roles.get(key, {})
            fsets = self.feature_sets.get(key, {})
            prof = self._profile_table(df, roles, fsets)
            self.table_profiles[key] = prof

            leaks = self._detect_leakage(df, roles, spec)
            self.leakage_findings[key] = leaks

            ts_diag = self._profile_time_series_diagnostics(spec, df, roles, fsets)
            self.time_series_diagnostics[key] = ts_diag
            
    def _profile_table(self, df: pd.DataFrame, roles: Dict[str, Any], fsets: Dict[str, List[str]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["n_rows"] = int(len(df))
        out["n_cols"] = int(df.shape[1])

        missing = df.isna().mean().sort_values(ascending=False)
        out["missing_top"] = [{"col": c, "missing_rate": float(missing.loc[c])} for c in missing.head(30).index]

        cat_cols = fsets.get("categorical", [])
        card = []
        for c in cat_cols[:200]:
            card.append({"col": c, "n_unique": int(df[c].nunique(dropna=True))})
        card = sorted(card, key=lambda x: x["n_unique"], reverse=True)
        out["categorical_cardinality_top"] = card[:30]

        num_cols = fsets.get("numeric", [])
        outlier_list, heavy_tail_list, log_candidates = [], [], []
        for c in num_cols[:300]:
            s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 50:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_rate = float(((s < lower) | (s > upper)).mean())
                if outlier_rate >= 0.01:
                    outlier_list.append({"col": c, "outlier_rate_iqr": outlier_rate})

            skew, kurt = float(s.skew()), float(s.kurtosis())
            if abs(skew) >= 1.0 or kurt >= 5.0:
                heavy_tail_list.append({"col": c, "skew": skew, "kurtosis": kurt})
            if s.min() >= 0 and skew >= 1.0:
                log_candidates.append({"col": c, "skew": skew})

        out["numeric_outliers_top"] = sorted(outlier_list, key=lambda x: x["outlier_rate_iqr"], reverse=True)[:30]
        out["heavy_tail_top"] = sorted(heavy_tail_list, key=lambda x: (abs(x["skew"]), x["kurtosis"]), reverse=True)[:30]
        out["log_candidates_top"] = sorted(log_candidates, key=lambda x: x["skew"], reverse=True)[:30]

        id_cols = roles.get("id_cols") or []
        time_col = roles.get("time_col")
        key_candidates = []
        if id_cols and time_col:
            key_candidates.append(id_cols + [time_col])
        elif id_cols:
            key_candidates.append(id_cols)
        elif time_col:
            key_candidates.append([time_col])

        uniq_checks = []
        for kc in key_candidates[:3]:
            kc = [c for c in kc if c in df.columns]
            if not kc:
                continue
            uniq_checks.append({"key_cols": kc, "dup_rate": float(df.duplicated(subset=kc).mean())})
        out["key_uniqueness_checks"] = uniq_checks

        dt_cols = fsets.get("datetime", [])
        dt_hints = []
        for c in dt_cols[:50]:
            hint = self._datetime_hint(df[c])
            if hint:
                dt_hints.append({"col": c, **hint})
        out["datetime_hints"] = dt_hints

        return out
    
    def _profile_time_series_diagnostics(
        self,
        spec: TableSpec,
        df: pd.DataFrame,
        roles: Dict[str, Any],
        fsets: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        time_col = roles.get("time_col")
        entity_key = []
        ticker_col = roles.get("ticker_col")
        if ticker_col:
            entity_key.append(ticker_col)
        for c in roles.get("id_cols", []) or []:
            if c not in entity_key:
                entity_key.append(c)

        if not spec.temporal or not time_col or time_col not in df.columns:
            return {
                "enabled": False,
                "reason": "table_is_not_temporal_or_time_col_missing"
            }

        work = df.copy()

        parsed_time = pd.to_datetime(work[time_col], errors="coerce", utc=False)
        valid_mask = parsed_time.notna()
        work = work.loc[valid_mask].copy()
        work[time_col] = parsed_time.loc[valid_mask]

        if len(work) < 30:
            return {
                "enabled": False,
                "reason": "insufficient_valid_time_rows",
                "time_col": time_col
            }

        proxy_col = self._choose_proxy_target_col(work, fsets)
        if proxy_col is None:
            return {
                "enabled": False,
                "reason": "no_numeric_proxy_column_found",
                "time_col": time_col,
                "entity_key": entity_key,
            }

        grouped_series = self._extract_groupwise_series(work, time_col, entity_key, proxy_col)

        if len(grouped_series) == 0:
            return {
                "enabled": False,
                "reason": "no_valid_groupwise_series",
                "time_col": time_col,
                "entity_key": entity_key,
                "proxy_target_col": proxy_col,
            }

        metrics = {
            "autocorr_lag1": [],
            "autocorr_lag5": [],
            "seasonality_strength": [],
            "trend_strength": [],
            "regime_change_score": [],
            "changepoint_density": [],
            "volatility_clustering": [],
            "nonstationarity_score": [],
            "series_lengths": [],
            "cross_sectional_std": [],
        }

        for s in grouped_series:
            vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 30:
                continue

            arr = vals.to_numpy(dtype=float)
            metrics["series_lengths"].append(len(arr))
            metrics["autocorr_lag1"].append(self._safe_autocorr(arr, 1))
            metrics["autocorr_lag5"].append(self._safe_autocorr(arr, 5))
            metrics["seasonality_strength"].append(self._seasonality_strength(arr))
            metrics["trend_strength"].append(self._trend_strength(arr))
            metrics["regime_change_score"].append(self._regime_change_score(arr))
            metrics["changepoint_density"].append(self._changepoint_density(arr))
            metrics["volatility_clustering"].append(self._volatility_clustering(arr))
            metrics["nonstationarity_score"].append(self._nonstationarity_score(arr))

        # cross-sectional dispersion: 같은 시점에 entity 간 분산 정도
        if entity_key:
            try:
                tmp = work[[time_col] + entity_key + [proxy_col]].copy()
                tmp[proxy_col] = pd.to_numeric(tmp[proxy_col], errors="coerce")
                tmp = tmp.dropna(subset=[proxy_col])
                cs = tmp.groupby(time_col)[proxy_col].std().dropna()
                if len(cs) > 0:
                    metrics["cross_sectional_std"] = cs.tolist()
            except Exception:
                pass

        valid_lengths = metrics["series_lengths"]
        if len(valid_lengths) == 0:
            return {
                "enabled": False,
                "reason": "no_series_long_enough_for_diagnostics",
                "time_col": time_col,
                "entity_key": entity_key,
                "proxy_target_col": proxy_col,
            }

        return {
            "enabled": True,
            "time_col": time_col,
            "entity_key": entity_key,
            "proxy_target_col": proxy_col,
            "n_entities": int(len(grouped_series)),
            "median_points_per_entity": int(np.median(valid_lengths)),
            "autocorr_lag1": self._robust_mean(metrics["autocorr_lag1"]),
            "autocorr_lag5": self._robust_mean(metrics["autocorr_lag5"]),
            "seasonality_strength": self._robust_mean(metrics["seasonality_strength"]),
            "trend_strength": self._robust_mean(metrics["trend_strength"]),
            "regime_change_score": self._robust_mean(metrics["regime_change_score"]),
            "changepoint_density": self._robust_mean(metrics["changepoint_density"]),
            "volatility_clustering": self._robust_mean(metrics["volatility_clustering"]),
            "nonstationarity_score": self._robust_mean(metrics["nonstationarity_score"]),
            "cross_sectional_dispersion": self._robust_mean(metrics["cross_sectional_std"]),
        }

    def _choose_proxy_target_col(self, df: pd.DataFrame, fsets: Dict[str, List[str]]) -> Optional[str]:
        preferred = [
            "close", "종가", "y", "target",
            "daily_return", "log_return",
            "revenue_yoy", "operating_income_yoy",
            "recent_news_ratio"
        ]
        num_cols = fsets.get("numeric", []) or []

        for cand in preferred:
            if cand in df.columns and cand in num_cols:
                return cand

        for c in num_cols:
            cl = str(c).lower()
            if any(k in cl for k in ["close", "price", "return", "ret", "revenue", "income", "score", "ratio"]):
                return c

        return num_cols[0] if num_cols else None

    def _extract_groupwise_series(
        self,
        df: pd.DataFrame,
        time_col: str,
        entity_key: List[str],
        value_col: str,
    ) -> List[pd.Series]:
        out = []

        if entity_key:
            g = df.groupby(entity_key, dropna=False)
            for _, sub in g:
                sub = sub.sort_values(time_col)
                s = pd.to_numeric(sub[value_col], errors="coerce").dropna()
                if len(s) >= 30:
                    out.append(s.reset_index(drop=True))
        else:
            sub = df.sort_values(time_col)
            s = pd.to_numeric(sub[value_col], errors="coerce").dropna()
            if len(s) >= 30:
                out.append(s.reset_index(drop=True))

        return out

    def _safe_autocorr(self, arr: np.ndarray, lag: int) -> float:
        if len(arr) <= lag + 5:
            return 0.0
        x = arr[:-lag]
        y = arr[lag:]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))

    def _seasonality_strength(self, arr: np.ndarray) -> float:
        candidates = [5, 20, 60]
        vals = []
        for lag in candidates:
            if len(arr) > lag + 10:
                vals.append(abs(self._safe_autocorr(arr, lag)))
        return float(np.clip(np.max(vals) if vals else 0.0, 0.0, 1.0))

    def _trend_strength(self, arr: np.ndarray) -> float:
        if len(arr) < 20:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        y = arr.astype(float)
        if np.std(y) < 1e-12:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(np.clip(abs(corr), 0.0, 1.0))

    def _regime_change_score(self, arr: np.ndarray) -> float:
        if len(arr) < 40:
            return 0.0
        w = max(10, len(arr) // 8)
        s = pd.Series(arr)
        rm = s.rolling(w).mean().dropna()
        if len(rm) < 5:
            return 0.0
        diffs = rm.diff().abs().dropna()
        base = float(np.nanstd(arr)) + 1e-9
        score = float(diffs.mean() / base)
        return float(np.clip(score, 0.0, 1.0))

    def _changepoint_density(self, arr: np.ndarray) -> float:
        if len(arr) < 40:
            return 0.0
        s = pd.Series(arr)
        w = max(10, len(arr) // 8)
        rm = s.rolling(w).mean().dropna()
        if len(rm) < 5:
            return 0.0
        diffs = rm.diff().abs().dropna()
        thr = diffs.mean() + 2.0 * diffs.std()
        if np.isnan(thr):
            return 0.0
        cp = int((diffs > thr).sum())
        return float(np.clip(cp / max(len(arr), 1), 0.0, 1.0))

    def _volatility_clustering(self, arr: np.ndarray) -> float:
        if len(arr) < 40:
            return 0.0
        ret = np.diff(arr)
        if len(ret) < 20:
            return 0.0
        abs_ret = np.abs(ret)
        return float(np.clip(abs(self._safe_autocorr(abs_ret, 1)), 0.0, 1.0))

    def _nonstationarity_score(self, arr: np.ndarray) -> float:
        if len(arr) < 40:
            return 0.0

        first = arr[: len(arr) // 2]
        second = arr[len(arr) // 2 :]

        mean_shift = abs(np.mean(first) - np.mean(second)) / (np.std(arr) + 1e-9)
        std_shift = abs(np.std(first) - np.std(second)) / (np.std(arr) + 1e-9)
        score = 0.6 * mean_shift + 0.4 * std_shift
        return float(np.clip(score, 0.0, 1.0))

    def _robust_mean(self, xs: List[float]) -> float:
        vals = [float(x) for x in xs if x is not None and not np.isnan(x)]
        if len(vals) == 0:
            return 0.0
        return float(np.median(vals))

    def _datetime_hint(self, s: pd.Series) -> Optional[Dict[str, Any]]:
        if pd.api.types.is_datetime64_any_dtype(s):
            return {"detected": "datetime64", "parse_needed": False}
        if not pd.api.types.is_object_dtype(s):
            return None
        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0:
            return None
        looks_like_date = float(sample.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}").mean())
        looks_like_ts = float(sample.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s+\d{1,2}:\d{2}").mean())
        looks_like_rfc822 = float(sample.str.match(self._RFC822_PAT).mean())
        if looks_like_rfc822 >= 0.5:
            return {"detected": "string_rfc822", "parse_needed": True, "format_hint": "RFC822 (e.g., Tue, 20 Jan 2026 12:02:00 +0900)"}
        if looks_like_ts >= 0.5:
            return {"detected": "string_timestamp", "parse_needed": True, "format_hint": "YYYY-MM-DD HH:MM"}
        if looks_like_date >= 0.5:
            return {"detected": "string_date", "parse_needed": True, "format_hint": "YYYY-MM-DD"}
        return None

    def _time_series_preprocessing_recommendations(
        self,
        spec: TableSpec,
        roles: Dict[str, Any],
        ts_diag: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not ts_diag.get("enabled", False):
            return {
                "enabled": False,
                "reason": ts_diag.get("reason", "not_applicable")
            }

        required = ["sort_by_time", "groupwise_processing", "time_based_split"]
        recommended = []
        optional = []
        avoid = ["random_split", "global_shuffle", "forward_fill_across_split"]
        notes = []

        if roles.get("time_col"):
            required.append("time_col_parse_and_validate")

        if ts_diag.get("autocorr_lag1", 0.0) >= 0.3:
            recommended.append("lag_features")
            notes.append("autocorr_lag1가 높아 lag feature 활용 권장")

        if ts_diag.get("trend_strength", 0.0) >= 0.4:
            recommended.append("difference_or_return_transform")
            notes.append("trend_strength가 높아 차분 또는 수익률 변환 검토")

        if ts_diag.get("seasonality_strength", 0.0) >= 0.35:
            recommended.append("seasonal_lag_features")
            optional.append("fourier_terms")
            notes.append("seasonality_strength가 높아 seasonal lag / Fourier term 검토")

        if ts_diag.get("volatility_clustering", 0.0) >= 0.35:
            recommended.append("rolling_volatility_features")
            optional.append("rolling_normalization")
            notes.append("volatility clustering이 있어 rolling volatility 계열 feature 권장")

        if ts_diag.get("regime_change_score", 0.0) >= 0.3 or ts_diag.get("changepoint_density", 0.0) >= 0.03:
            optional.append("changepoint_aware_model")
            optional.append("regime_indicator_features")
            notes.append("regime 변화가 감지되어 changepoint/regime-aware 접근 검토")

        if ts_diag.get("nonstationarity_score", 0.0) >= 0.4:
            recommended.append("stationarity_adjustment")
            notes.append("비정상성이 높아 차분, detrending, rolling normalization 검토")

        return {
            "enabled": True,
            "required": self._dedup_keep_order(required),
            "recommended": self._dedup_keep_order(recommended),
            "optional": self._dedup_keep_order(optional),
            "avoid": self._dedup_keep_order(avoid),
            "notes": notes,
        }
    
    # ---------------------------
    # Leakage detection
    # ---------------------------

    def _detect_leakage(self, df: pd.DataFrame, roles: Dict[str, Any], spec: TableSpec) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        cols = df.columns

        pattern_rules = [
            (r"(?:^|_)(target|label|y)(?:$|_)", "Possible target-derived column by name"),
            (r"(?:^|_)(future|next|lead|t\+|ahead)(?:$|_)", "Possible future-looking leakage by name"),
            (r"(?:^|_)(tomorrow|after|post)(?:$|_)", "Possible future-looking leakage by name"),
            (r"(?:^|_)(close_t\+1|ret_t\+1|return_t\+1)(?:$|_)", "Explicit next-step target proxy pattern"),
        ]
        for c in cols:
            cl = str(c).lower()
            for pat, reason in pattern_rules:
                if re.search(pat, cl):
                    findings.append({"col": c, "reason": reason, "rule": pat})

        if spec.temporal is False and spec.domain in ("price", "news"):
            for c in cols:
                cl = str(c).lower()
                if any(k in cl for k in ("return", "ret", "yoy", "next", "lead")):
                    findings.append({"col": c, "reason": "Snapshot table contains outcome-like fields; join may leak", "rule": "snapshot_outcome_like"})

        uniq = {}
        for f in findings:
            uniq[(f["col"], f["rule"])] = f
        return list(uniq.values())

    # ---------------------------
    # Dictionary -> requirements (supports feature+column dictionaries)
    # ---------------------------

    def _dictionary_requirements_for_table(self, table_key: str, spec: TableSpec, df: pd.DataFrame) -> Dict[str, Any]:
        # feature-meta (domain)
        fmeta = self.dict_feature_meta.get(spec.domain, {}) or {}
        present_features = [c for c in df.columns if c in fmeta]

        # column-meta (table-level preferred, else domain-level)
        cmeta = (self.dict_column_meta.get(table_key) or self.dict_column_meta.get(spec.domain) or {})
        present_columns = [c for c in df.columns if c in cmeta]

        # feature dict analytics
        window_groups: Dict[str, List[str]] = {}
        for c in present_features:
            w = str(fmeta[c].get("window", "")).strip()
            if not w:
                continue
            window_groups.setdefault(w, []).append(c)

        zero_div, inf_risk, min_periods_related = [], [], []
        notes_samples = []
        for c in present_features:
            notes = str(fmeta[c].get("notes", "")).strip()
            if not notes:
                continue
            notes_samples.append({"feature": c, "notes": notes[:200]})
            if self._DICT_NOTES_ZERO_DIV_PAT.search(notes):
                zero_div.append(c)
            if self._DICT_NOTES_INF_PAT.search(notes):
                inf_risk.append(c)
            if self._DICT_NOTES_MINPER_PAT.search(notes):
                min_periods_related.append(c)

        deps = []
        for c in present_features:
            formula = str(fmeta[c].get("formula", "")).strip()
            if not formula:
                continue
            tokens = self._FORMULA_TOKEN_PAT.findall(formula)
            colset = set(df.columns)
            used = sorted({t for t in tokens if (t in colset and t != c)})
            if used:
                deps.append({"feature": c, "depends_on_columns_present": used})

        data_source_map = {}
        for c in present_features:
            src = str(fmeta[c].get("data_source", "")).strip()
            if src:
                data_source_map[c] = src

        return {
            "dictionary_coverage": {
                "feature_dict": {
                    "n_features_in_dictionary": int(len(fmeta)),
                    "n_features_present_in_table_and_dictionary": int(len(present_features)),
                    "present_features": present_features[:200],
                },
                "column_dict": {
                    "n_columns_in_dictionary": int(len(cmeta)),
                    "n_columns_present_in_table_and_dictionary": int(len(present_columns)),
                    "present_columns": present_columns[:200],
                },
            },
            "window_groups": window_groups,
            "notes_flags": {
                "divide_by_zero_risk_features": zero_div,
                "infinite_or_overflow_risk_features": inf_risk,
                "min_periods_or_min_history_mentioned": min_periods_related,
                "notes_samples": notes_samples[:30],
            },
            "formula_dependencies": deps[:200],
            "data_source_by_feature": data_source_map,
            "suggested_actions": {
                "divide_by_zero": "ensure_safe_divide_and_set_inf_to_nan",
                "inf_handling": "replace_inf_with_nan_then_impute_or_drop",
                "window_features": "ensure_time_sorted_and_min_history_available_before_training_split",
                "column_meta": "honor role/semantic_type/model_use flags to avoid accidental model input",
            },
        }

    # ---------------------------
    # Plan synthesis
    # ---------------------------

    def _synthesize_preprocessing_plan_requirements_only(self) -> Dict[str, Any]:
        split_strategy = self._choose_split_strategy()
        problem_type = self.training_intent.get("problem_type") or self.split_hint.get("problem_type") or "ts_forecast"

        tables_block = {}
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue

            df = self.tables[key]
            roles = self.inferred_roles.get(key, {})
            fsets = self.feature_sets.get(key, {})
            prof = self.table_profiles.get(key, {})
            leaks = self.leakage_findings.get(key, [])

            tables_block[key] = {
                "domain": spec.domain,
                "name": spec.name,
                "role": spec.role,
                "temporal": bool(spec.temporal),
                "roles": roles,
                "feature_sets": fsets,
                "requirements": self._requirements_for_table(spec, df, roles, fsets, prof, leaks, split_strategy),
                "dictionary_requirements": self._dictionary_requirements_for_table(key, spec, df),
                "leakage_findings": leaks,
            }

        plan = {
            "version": "1.3",
            "problem_schema": {
                "problem_type": problem_type,
                "split_strategy": split_strategy,
                "evaluation": self.training_intent.get("evaluation") or self.split_hint.get("evaluation") or {},
                "constraints": self.training_intent.get("constraints") or {},
            },
            "roles_global": self._merge_roles_global(),
            "join_key_requirements_global": self._global_join_key_requirements(),
            "tables": tables_block,
            "leakage_policy": self._default_leakage_policy_requirements_only(),
            "validation": self._plan_validation_checks(split_strategy),
            "notes": {
                "this_file_is_requirements_only": True,
                "no_executable_pipelines_here": True,
                "model_data_matcher_should_decide_exact_transforms": True
            }
        }
        return plan

    def _requirements_for_table(
        self,
        spec: TableSpec,
        df: pd.DataFrame,
        roles: Dict[str, Any],
        fsets: Dict[str, List[str]],
        prof: Dict[str, Any],
        leaks: List[Dict[str, Any]],
        split_strategy: str,
    ) -> Dict[str, Any]:
        ts_diag = self.time_series_diagnostics.get(f"{spec.domain}.{spec.name}", {})

        missing_map = {d["col"]: d["missing_rate"] for d in (prof.get("missing_top") or [])}
        high_missing = [c for c, r in missing_map.items() if r >= 0.5]
        mid_missing = [c for c, r in missing_map.items() if 0.1 <= r < 0.5]

        outlier_cols = [d["col"] for d in (prof.get("numeric_outliers_top") or [])]
        heavy_tail_cols = [d["col"] for d in (prof.get("heavy_tail_top") or [])]
        log_cand_cols = [d["col"] for d in (prof.get("log_candidates_top") or [])]

        card_map = {d["col"]: d["n_unique"] for d in (prof.get("categorical_cardinality_top") or [])}
        low_card, high_card = [], []
        for c in fsets.get("categorical", []):
            nunique = int(card_map.get(c, df[c].nunique(dropna=True)))
            (low_card if nunique <= 50 else high_card).append(c)

        time_col = roles.get("time_col")
        table_is_temporal = bool(spec.temporal)
        requires_time_col = (split_strategy == "time-based" and table_is_temporal)
        time_col_present = bool(time_col)
        time_alignment_risk = bool(table_is_temporal and requires_time_col and not time_col_present)

        raw_text_reqs = None
        if spec.role == "raw_text_reference":
            raw_text_reqs = self._raw_text_reference_requirements(df, roles)

        join_key_reqs = self._table_join_key_requirements(spec, df, roles)

        return {
            "missingness": {
                "high_missing_cols": high_missing,
                "mid_missing_cols": mid_missing,
                "suggested_action": {
                    "high": "consider_drop_or_domain_impute",
                    "mid": "imputation_recommended"
                }
            },
            "numeric_distribution": {
                "outlier_prone_cols": outlier_cols,
                "heavy_tail_cols": heavy_tail_cols,
                "log_candidate_cols": log_cand_cols,
                "suggested_action": {
                    "outliers": "robust_handling_recommended",
                    "heavy_tail": "robust_handling_or_transform_consider",
                    "log_candidates": "monotonic_transform_may_help"
                }
            },
            "scale_sensitivity_hints": {
                "has_numeric_features": bool(len(fsets.get("numeric", [])) > 0),
                "has_heavy_tail_features": bool(len(heavy_tail_cols) > 0),
            },
            "categorical": {
                "low_cardinality_cols": low_card,
                "high_cardinality_cols": high_card,
                "numeric_coded_categorical_cols": fsets.get("categorical_numeric_codes", []),
                "suggested_action": {
                    "low_card": "encoding_required_if_used",
                    "high_card": "compressed_encoding_or_hashing_candidate",
                    "numeric_codes": "treat_as_categorical_not_continuous"
                }
            },
            "text": {
                "text_cols": fsets.get("text", []),
                "suggested_action": "vectorization_required_if_used"
            },
            "datetime": {
                "datetime_cols": fsets.get("datetime", []),
                "parse_needed_cols": [d["col"] for d in (prof.get("datetime_hints", []) or []) if d.get("parse_needed")],
                "audit_time_cols": roles.get("audit_time_cols", []),
                "suggested_action": "parse_if_needed; avoid_using_audit_cols_as_time_index_for_snapshot"
            },
            "split_constraints": {
                "table_is_temporal": table_is_temporal,
                "requires_time_col": requires_time_col,
                "time_col_present": time_col_present,
                "time_alignment_risk": time_alignment_risk
            },
            "time_series_diagnostics": ts_diag,
            "time_series_preprocessing_recommendations": self._time_series_preprocessing_recommendations(
                spec=spec,
                roles=roles,
                ts_diag=ts_diag,
            ),
            "join_keys": join_key_reqs,
            "raw_text_reference": raw_text_reqs,
            "leakage": {
                "suspect_cols": sorted(list({f["col"] for f in leaks})),
                "suggested_action": "quarantine_or_drop_before_training"
            }
        }

    def _raw_text_reference_requirements(self, df: pd.DataFrame, roles: Dict[str, Any]) -> Dict[str, Any]:
        cols = set(df.columns)

        preferred_time = None
        for cand in ["pubDate", "pubdate", "pub_date", "date", "datetime", "timestamp"]:
            if cand in cols:
                preferred_time = cand
                break
        if preferred_time is None and roles.get("time_col"):
            preferred_time = roles["time_col"]

        dedup_keys = []
        if "originallink" in cols:
            dedup_keys.append(["originallink"])
        if "link" in cols:
            dedup_keys.append(["link"])
        if ("title" in cols) and (preferred_time in cols if preferred_time else False):
            dedup_keys.append(["title", preferred_time])
        if ("title" in cols) and ("description" in cols):
            dedup_keys.append(["title", "description"])

        text_fields = [c for c in ["title", "description"] if c in cols]

        return {
            "time_axis": {
                "preferred_time_col": preferred_time,
                "fallback_time_cols": [c for c in ["fetched_at", "created_at"] if c in cols],
                "suggested_action": "use_article_pub_time_for_temporal_alignment; use_fetched_at_only_for_audit",
            },
            "deduplication": {
                "dedup_key_candidates": dedup_keys,
                "suggested_action": "deduplicate_before_aggregation_or_join",
            },
            "aggregation": {
                "group_keys": [k for k in [roles.get("ticker_col"), preferred_time] if k],
                "suggested_windows": ["daily", "weekly"],
                "suggested_action": "aggregate_text_to_stock_time_bucket_before_joining_to_price",
            },
            "text_fields": {
                "fields": text_fields,
                "suggested_action": "keep_raw_text_for_rag_or_nlp; do_not_leak_future_articles_in_time_split",
            },
            "url_fields": [c for c in ["link", "originallink"] if c in cols],
        }

    def _choose_split_strategy(self) -> str:
        forced = self.split_hint.get("split_strategy")
        if forced:
            return forced

        any_temporal = False
        any_group = False
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue
            roles = self.inferred_roles.get(key, {})
            if spec.temporal is True:
                any_temporal = True
            if roles.get("group_cols"):
                any_group = True

        if any_temporal:
            return "time-based"
        if any_group:
            return "group-based"
        return "random"

    # ---------------------------
    # Join key requirements
    # ---------------------------

    def _global_join_key_requirements(self) -> Dict[str, Any]:
        return {
            "canonical_ticker_name": "종목코드",
            "canonicalization": [
                "cast_to_string",
                "strip_whitespace",
                "zero_pad_to_6_if_all_digits",
            ],
            "notes": [
                "Ensure all tables use the same ticker column name and dtype before joins.",
                "If original ticker col is 'ticker'/'code'/'stock_code', map to '종목코드' for join consistency.",
            ],
        }

    def _table_join_key_requirements(self, spec: TableSpec, df: pd.DataFrame, roles: Dict[str, Any]) -> Dict[str, Any]:
        ticker_col = roles.get("ticker_col")
        id_cols = roles.get("id_cols") or []
        time_col = roles.get("time_col")

        req = {
            "ticker_col_detected": ticker_col,
            "suggested_canonical_ticker_col": "종목코드",
            "canonicalization_steps": [
                {"step": "cast_to_string", "applies_to": [ticker_col] if ticker_col else []},
                {"step": "strip_whitespace", "applies_to": [ticker_col] if ticker_col else []},
                {"step": "zero_pad_to_6_if_digits", "applies_to": [ticker_col] if ticker_col else []},
            ],
            "unique_key_suggestion": (id_cols + [time_col]) if (id_cols and time_col) else (id_cols or ([time_col] if time_col else [])),
        }

        if spec.domain == "finance" and "종목코드" in df.columns:
            req["finance_filename_ticker_note"] = "종목코드 from filename should be zfill(6) and string"

        return req

    # ---------------------------
    # Leakage policy / validation
    # ---------------------------

    def _default_leakage_policy_requirements_only(self) -> Dict[str, Any]:
        return {
            "deny_name_patterns": [
                r"(?:^|_)(target|label|y)(?:$|_)",
                r"(?:^|_)(future|next|lead|t\+|ahead)(?:$|_)",
                r"(?:^|_)(close_t\+1|ret_t\+1|return_t\+1)(?:$|_)",
            ],
            "actions": {
                "hard_drop": False,
                "quarantine": True,
                "require_manual_review": True
            },
            "time_alignment": {
                "require_time_col_for_temporal": True,
                "disallow_forward_fill_across_split": True
            }
        }

    def _merge_roles_global(self) -> Dict[str, Any]:
        out = {"targets": [], "time_cols": [], "id_cols": [], "group_cols": [], "ticker_cols": [], "audit_time_cols": []}
        for key, roles in self.inferred_roles.items():
            for t in roles.get("target") or []:
                out["targets"].append({"table": key, "col": t})
            if roles.get("time_col"):
                out["time_cols"].append({"table": key, "col": roles["time_col"]})
            for c in roles.get("audit_time_cols") or []:
                out["audit_time_cols"].append({"table": key, "col": c})
            for c in roles.get("id_cols") or []:
                out["id_cols"].append({"table": key, "col": c})
            for c in roles.get("group_cols") or []:
                out["group_cols"].append({"table": key, "col": c})
            if roles.get("ticker_col"):
                out["ticker_cols"].append({"table": key, "col": roles["ticker_col"]})
        return out

    def _plan_validation_checks(self, split_strategy: str) -> Dict[str, Any]:
        issues = []
        if split_strategy == "time-based":
            has_any_time = any((roles.get("time_col") for roles in self.inferred_roles.values()))
            if not has_any_time:
                issues.append({"severity": "error", "msg": "split_strategy is time-based but no time_col inferred in any table"})

        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue

            if spec.temporal is True:
                time_col = (self.inferred_roles.get(key, {}) or {}).get("time_col")
                if not time_col:
                    issues.append({"severity": "warning", "msg": f"{key} marked temporal but time_col not inferred; set split_hint override"})

            if spec.temporal is False:
                time_col = (self.inferred_roles.get(key, {}) or {}).get("time_col")
                if time_col:
                    issues.append({"severity": "warning", "msg": f"{key} is snapshot(temporal=False) but time_col was inferred ({time_col}). Check overrides/strict time rules."})

        total_leaks = sum(len(v) for v in self.leakage_findings.values())
        if total_leaks > 0:
            issues.append({"severity": "warning", "msg": f"Leakage heuristics flagged {total_leaks} column(s). Review leakage_findings."})

        return {"issues": issues, "passed": not any(i["severity"] == "error" for i in issues)}

    # ---------------------------
    # Data contract synthesis (dictionary_meta attaches feature or column meta correctly)
    # ---------------------------

    def _get_dictionary_meta_for_column(self, table_key: str, domain: str, col: str) -> Optional[Dict[str, Any]]:
        # priority: table-level column meta -> domain-level column meta -> domain-level feature meta
        if table_key in self.dict_column_meta and col in self.dict_column_meta[table_key]:
            return self.dict_column_meta[table_key][col]
        if domain in self.dict_column_meta and col in self.dict_column_meta[domain]:
            return self.dict_column_meta[domain][col]
        if domain in self.dict_feature_meta and col in self.dict_feature_meta[domain]:
            return self.dict_feature_meta[domain][col]
        return None

    def _synthesize_data_contract(self) -> Dict[str, Any]:
        contract_tables = {}
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if key not in self.tables:
                continue

            df = self.tables[key]
            roles = self.inferred_roles.get(key, {})
            fsets = self.feature_sets.get(key, {})
            prof = self.table_profiles.get(key, {})

            missing_map = {d["col"]: d["missing_rate"] for d in (prof.get("missing_top") or [])}
            card_map = {d["col"]: d["n_unique"] for d in (prof.get("categorical_cardinality_top") or [])}
            log_cands = {d["col"] for d in (prof.get("log_candidates_top") or [])}
            heavy_tail = {d["col"] for d in (prof.get("heavy_tail_top") or [])}

            col_info = []
            for c in df.columns:
                dtype = str(df[c].dtype)
                col_info.append(
                    {
                        "name": c,
                        "dtype": dtype,
                        "nullable": bool(df[c].isna().any()),
                        "missing_rate": float(missing_map.get(c, float(df[c].isna().mean()))),
                        "cardinality_est": int(card_map.get(c, df[c].nunique(dropna=True))) if c in fsets.get("categorical", []) else None,
                        "scale_hints": {
                            "log_candidate": bool(c in log_cands),
                            "heavy_tail": bool(c in heavy_tail),
                        } if c in fsets.get("numeric", []) else None,
                        "dictionary_meta": self._get_dictionary_meta_for_column(key, spec.domain, c),
                    }
                )

            time_col = roles.get("time_col")
            id_cols = roles.get("id_cols") or []
            unique_key = None
            if id_cols and time_col:
                unique_key = id_cols + [time_col]
            elif id_cols:
                unique_key = id_cols
            elif time_col:
                unique_key = [time_col]

            unique_key_valid = None
            if unique_key:
                cols_exist = [c for c in unique_key if c in df.columns]
                if cols_exist:
                    unique_key_valid = float(df.duplicated(subset=cols_exist).mean()) == 0.0

            table_obj = {
                "domain": spec.domain,
                "name": spec.name,
                "role": spec.role,
                "roles": roles,
                "feature_sets": fsets,
                "schema": col_info,
                "datetime_format_hints": prof.get("datetime_hints", []),
                "unique_key": unique_key,
                "unique_key_strictly_unique": unique_key_valid,
            }

            if spec.role == "raw_text_reference":
                dedup_key_candidates = []
                if "originallink" in df.columns:
                    dedup_key_candidates.append(["originallink"])
                if "link" in df.columns:
                    dedup_key_candidates.append(["link"])
                if ("title" in df.columns) and ((roles.get("time_col") or "pubDate") in df.columns):
                    dedup_key_candidates.append(["title", roles.get("time_col") or "pubDate"])

                table_obj["uniqueness_semantics"] = {
                    "expected_duplicates": True,
                    "reason": "raw news contains multiple articles per ticker per time bucket; duplicates are normal before dedup/aggregation",
                    "dedup_key_candidates": dedup_key_candidates,
                    "recommended_pre_join_step": "deduplicate_then_aggregate_to_time_bucket",
                }

            contract_tables[key] = table_obj

        return {
            "version": "1.3",
            "tables": contract_tables,
            "assumptions": {
                "timezone": self.split_hint.get("timezone", "Asia/Seoul"),
                "datetime_parsing": "Coerce errors; review datetime_hints for ambiguous columns",
                "join_key_canonicalization": self._global_join_key_requirements(),
            },
        }

    # ---------------------------
    # Report synthesis
    # ---------------------------

    def _synthesize_report_md(self, plan: Dict[str, Any], contract: Dict[str, Any]) -> str:
        lines = []
        lines.append("# Preprocessing Recommender Report\n")
        ps = plan.get("problem_schema", {})
        lines.append("## Global Settings")
        lines.append(f"- problem_type: **{ps.get('problem_type')}**")
        lines.append(f"- split_strategy: **{ps.get('split_strategy')}**\n")

        lines.append("## Global Join-Key Requirements")
        jk = plan.get("join_key_requirements_global", {})
        lines.append(f"- canonical_ticker_name: **{jk.get('canonical_ticker_name')}**")
        for s in (jk.get("canonicalization") or []):
            lines.append(f"  - {s}")
        lines.append("")

        val = plan.get("validation", {})
        lines.append("## Plan Validation")
        lines.append(f"- passed: **{val.get('passed')}**")
        issues = val.get("issues", [])
        if issues:
            lines.append("\n### Issues")
            for i in issues:
                lines.append(f"- [{i['severity'].upper()}] {i['msg']}")
        else:
            lines.append("- No issues detected.")
        lines.append("")

        lines.append("## Table Profiling Highlights")
        for key, prof in self.table_profiles.items():
            lines.append(f"### {key}")
            lines.append(f"- rows: {prof.get('n_rows')}, cols: {prof.get('n_cols')}\n")

            table_plan = (plan.get("tables", {}).get(key, {}) or {})
            dreq = table_plan.get("dictionary_requirements", {}) or {}
            reqs = table_plan.get("requirements", {}) or {}

            # Dictionary coverage
            cov = (dreq.get("dictionary_coverage", {}) or {})
            if cov:
                f = cov.get("feature_dict", {}) or {}
                c = cov.get("column_dict", {}) or {}
                lines.append("**Dictionary coverage**")
                if f:
                    lines.append(
                        f"- feature_dict present: "
                        f"{f.get('n_features_present_in_table_and_dictionary')} / {f.get('n_features_in_dictionary')}"
                    )
                if c:
                    lines.append(
                        f"- column_dict present: "
                        f"{c.get('n_columns_present_in_table_and_dictionary')} / {c.get('n_columns_in_dictionary')}"
                    )
                lines.append("")

            # Time-series diagnostics
            ts_diag = reqs.get("time_series_diagnostics", {}) or {}
            ts_reco = reqs.get("time_series_preprocessing_recommendations", {}) or {}

            if ts_diag.get("enabled"):
                lines.append("**Time-series diagnostics**")
                lines.append(f"- time_col: {ts_diag.get('time_col')}")
                lines.append(f"- proxy_target_col: {ts_diag.get('proxy_target_col')}")
                lines.append(f"- n_entities: {ts_diag.get('n_entities')}")
                lines.append(f"- median_points_per_entity: {ts_diag.get('median_points_per_entity')}")
                lines.append(f"- autocorr_lag1: {ts_diag.get('autocorr_lag1', 0.0):.3f}")
                lines.append(f"- autocorr_lag5: {ts_diag.get('autocorr_lag5', 0.0):.3f}")
                lines.append(f"- seasonality_strength: {ts_diag.get('seasonality_strength', 0.0):.3f}")
                lines.append(f"- trend_strength: {ts_diag.get('trend_strength', 0.0):.3f}")
                lines.append(f"- regime_change_score: {ts_diag.get('regime_change_score', 0.0):.3f}")
                lines.append(f"- changepoint_density: {ts_diag.get('changepoint_density', 0.0):.3f}")
                lines.append(f"- volatility_clustering: {ts_diag.get('volatility_clustering', 0.0):.3f}")
                lines.append(f"- nonstationarity_score: {ts_diag.get('nonstationarity_score', 0.0):.3f}")
                lines.append(f"- cross_sectional_dispersion: {ts_diag.get('cross_sectional_dispersion', 0.0):.3f}")
                lines.append("")

                lines.append("**Time-series preprocessing recommendations**")
                for bucket in ["required", "recommended", "optional", "avoid"]:
                    vals = ts_reco.get(bucket, []) or []
                    if vals:
                        lines.append(f"- {bucket}: {', '.join(vals)}")

                notes = ts_reco.get("notes", []) or []
                if notes:
                    lines.append("- notes:")
                    for note in notes:
                        lines.append(f"  - {note}")
                lines.append("")

            elif "time_series_diagnostics" in reqs:
                lines.append("**Time-series diagnostics**")
                lines.append(f"- enabled: False")
                lines.append(f"- reason: {ts_diag.get('reason', 'not_applicable')}")
                lines.append("")

        return "\n".join(lines)
        
    # ---------------------------
    # Helpers
    # ---------------------------

    def _guess_ticker_col(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ["종목코드", "ticker", "symbol", "code", "stock_code", "종목코드 "]
        cols_lower = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        for c in df.columns:
            if "ticker" in str(c).lower() or "종목" in str(c):
                return c
        return None

    def _dedup_keep_order(self, xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _write_json(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _write_text(self, path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
