from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import subprocess
import shutil

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


@dataclass
class TableSpec:
    domain: str
    name: str
    path: str
    role: str = "feature_table"
    temporal: Optional[bool] = None
    time_col: Optional[str] = None
    entity_key: Optional[List[str]] = None


class ModelDataMatcherAgent(BaseAgent):

    stage = "model_match_v1"

    LEAKAGE_KEYWORDS = {
        "today_source",
        "source",
        "row_type",
        "error",
        "fetched_at",
        "feature_now_utc",
        "_source_file",
    }

    TIME_NAME_PAT = re.compile(
        r"(?:^|_)(date|datetime|timestamp|trade_date|pubdate|pub_date|year)(?:$|_)",
        re.I,
    )

    def __init__(
        self,
        base_dir: str = "",
        output_dir: str = "",
        model_db_dir: str = "",

        # prep_reco outputs
        preprocessing_plan_path: Optional[str] = None,
        data_contract_path: Optional[str] = None,
        preprocessing_report_path: Optional[str] = None,

        # feature tables
        user_features: str = "",
        price_ohlcv_last365: str = "",
        price_foreign_snapshot_today: str = "",
        finance_features_glob: str = "",
        news_features_by_stock: str = "",
        news_raw_merged: str = "",

        encoding: str = "utf-8",
        sample_rows: int = 200_000,
        random_state: int = 42,
    ):
        self.base_dir = base_dir or ""
        self.output_dir = output_dir or ""
        self.model_db_dir = self._abspath(model_db_dir) if model_db_dir else ""

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.paths = {
            "preprocessing_plan": self._abspath(preprocessing_plan_path) if preprocessing_plan_path else None,
            "data_contract": self._abspath(data_contract_path) if data_contract_path else None,
            "preprocessing_report": self._abspath(preprocessing_report_path) if preprocessing_report_path else None,

            "user_features": self._abspath(user_features),
            "price_ohlcv_last365": self._abspath(price_ohlcv_last365),
            "price_foreign_snapshot_today": self._abspath(price_foreign_snapshot_today),
            "finance_features_glob": self._abspath(finance_features_glob, is_pattern=True),
            "news_features_by_stock": self._abspath(news_features_by_stock),
            "news_raw_merged": self._abspath(news_raw_merged),
        }

        self.encoding = encoding
        self.sample_rows = int(sample_rows)
        self.random_state = int(random_state)

        self.preprocessing_plan: Dict[str, Any] = {}
        self.data_contract: Dict[str, Any] = {}

        self.table_specs: List[TableSpec] = []
        self.tables: Dict[str, pd.DataFrame] = {}
        self.table_profiles: Dict[str, Dict[str, Any]] = {}
        self.key_info_by_table: Dict[str, Dict[str, List[str]]] = {}
        self.plan_table_meta: Dict[str, Dict[str, Any]] = {}

    # =========================================================
    # Pipeline entry
    # =========================================================

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        try:
            runtime_agent = self if self._is_configured() else self._build_runtime_agent(ctx, ap)
            outputs = runtime_agent.run()

            return StageResult.success(
                stage=self.stage,
                outputs=[outputs["model_matching_result"]],
            )
        except Exception as e:
            ctx.logger.exception("❌ Model Matcher Failed")
            return StageResult.failed(stage=self.stage, error=str(e))

    def _is_configured(self) -> bool:
        required = [
            self.output_dir,
            self.model_db_dir,
            self.paths.get("price_ohlcv_last365"),
            self.paths.get("preprocessing_plan"),
            self.paths.get("data_contract"),
        ]
        return all(bool(x) for x in required)

    def _build_runtime_agent(self, ctx: RunContext, ap: ArtifactPaths) -> "ModelDataMatcherAgent":
        feature_dir = Path(ap.feature_table_dir())
        prep_reco_dir = Path(ap.prep_reco_dir())
        output_dir = Path(ap.stage_dir(self.stage))

        user_dir = feature_dir / "user"
        price_dir = feature_dir / "price"
        finance_dir = feature_dir / "finance"
        news_dir = feature_dir / "news"

        project_root = Path(ctx.project_root)
        model_db_dir = Path(ctx.flags.get("model_db_dir", project_root / "model_db"))

        user_features_name = ctx.flags.get("user_features", "user_features.csv")
        price_ohlcv_name = ctx.flags.get("price_ohlcv_last365", "price_ohlcv_features.csv")
        price_foreign_name = ctx.flags.get("price_foreign_snapshot_today", "price_foreign_features.csv")
        finance_glob_name = ctx.flags.get("finance_features_glob", "*_finance_features.csv")
        news_features_name = ctx.flags.get("news_features_by_stock", "news_features.csv")
        news_raw_merged_name = ctx.flags.get("news_raw_merged", "news_raw.csv")

        preprocessing_plan_name = ctx.flags.get("preprocessing_plan", "preprocessing_plan.json")
        data_contract_name = ctx.flags.get("data_contract", "data_contract.json")
        preprocessing_report_name = ctx.flags.get("preprocessing_report", "preprocessing_report.md")

        return ModelDataMatcherAgent(
            base_dir="",
            output_dir=str(output_dir),
            model_db_dir=str(model_db_dir),

            preprocessing_plan_path=str(prep_reco_dir / preprocessing_plan_name),
            data_contract_path=str(prep_reco_dir / data_contract_name),
            preprocessing_report_path=str(prep_reco_dir / preprocessing_report_name),

            user_features=str(user_dir / user_features_name),
            price_ohlcv_last365=str(price_dir / price_ohlcv_name),
            price_foreign_snapshot_today=str(price_dir / price_foreign_name),
            finance_features_glob=str(finance_dir / finance_glob_name),
            news_features_by_stock=str(news_dir / news_features_name),
            news_raw_merged=str(news_dir / news_raw_merged_name),

            encoding=ctx.flags.get("encoding", "utf-8"),
            sample_rows=int(ctx.flags.get("sample_rows", 200_000)),
            random_state=int(ctx.flags.get("random_state", 42)),
        )

    def run(self) -> Dict[str, str]:
        self.preprocessing_plan = self._read_json_if_exists(self.paths["preprocessing_plan"]) or {}
        self.data_contract = self._read_json_if_exists(self.paths["data_contract"]) or {}

        self._build_table_specs()
        self._load_all_tables()
        self._profile_all_tables()

        models = self._load_model_db(self.model_db_dir)
        if not models:
            raise RuntimeError(f"No model DB JSON found in: {self.model_db_dir}")

        primary_table = self._choose_primary_table()
        primary_df = self.tables[primary_table]
        primary_prof = self.table_profiles[primary_table]
        primary_key_info = self.key_info_by_table[primary_table]

        signals = self._build_selection_signals(
            primary_table=primary_table,
            primary_df=primary_df,
            primary_prof=primary_prof,
            primary_key_info=primary_key_info,
        )

        scored = []
        for model in models:
            score, reasons, notes = self._score_model(model, signals)
            scored.append({
                "model": model,
                "score": float(score),
                "reasons": reasons,
                "notes": notes,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        if not scored:
            raise RuntimeError("No models were scored.")

        best = scored[0]
        selected_model = best["model"]

        column_pre_map = self._build_column_preprocessing(primary_df, primary_key_info)
        missing_policy = self._decide_missing_policy(primary_prof, selected_model, primary_key_info)
        date_policy = self._decide_date_policy(primary_df, primary_key_info)
        join_keys = self._build_join_keys(date_policy)

        warnings = self._build_validation_warnings(primary_prof, column_pre_map)

        output = {
            "version": "3.0",
            "agent": "model_matcher_vscode_timeseries",
            "inputs": {
                "preprocessing_plan_present": bool(self.preprocessing_plan),
                "data_contract_present": bool(self.data_contract),
                "model_db_dir": self.model_db_dir,
                "primary_table": primary_table,
            },
            "selected_model": {
                "model_id": selected_model.get("model_id"),
                "model_name": selected_model.get("model_name"),
                "family": selected_model.get("family"),
                "target_type": selected_model.get("target_type"),
                "score": float(best["score"]),
                "selection_basis": "prep_reco_diagnostics_plus_raw_signals",
                "signals_used": signals,
                "reasons": best["reasons"],
                "notes": best["notes"],
                "top_k_candidates": [
                    {
                        "model_id": s["model"].get("model_id"),
                        "model_name": s["model"].get("model_name"),
                        "family": s["model"].get("family"),
                        "score": float(s["score"]),
                        "notes": s["notes"][:5],
                    }
                    for s in scored[:5]
                ],
            },
            "dataset_inventory": [
                {
                    "table": key,
                    "domain": self._table_domain_from_key(key),
                    "n_rows": prof["n_rows"],
                    "n_cols": prof["n_cols"],
                    "missing_rate_avg": prof["missing_rate_avg"],
                    "datetime_cols": prof["datetime_cols"][:5],
                }
                for key, prof in self.table_profiles.items()
            ],
            "data_characteristics_primary": primary_prof,
            "key_columns_primary": primary_key_info,
            "prep_reco_diagnostics_primary": self._get_plan_ts_diag(primary_table),
            "column_preprocessing_map": column_pre_map,
            "missing_value_policy": missing_policy,
            "constraints": {
                "drop_company_name_always": True,
                "keep_ticker_always": True,
                "ticker_column_candidates": primary_key_info.get("ticker_cols", []),
                "company_name_column_candidates": primary_key_info.get("company_name_cols", []),
                "default_drop_roles": ["company_name", "leakage_candidate"],
            },
            "date_policy": date_policy,
            "join_keys": join_keys,
            "validation_warnings": warnings,
            "resolved_actions": sorted(list({
                *selected_model.get("preprocessing_dependencies", {}).get("required", []),
                *selected_model.get("preprocessing_dependencies", {}).get("optional", []),
                "drop_company_name",
                "preserve_ticker",
                "datetime_dedup",
                "drop_or_mask_leakage",
            })),
        }

        out_path = os.path.join(self.output_dir, "model_matching_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return {"model_matching_result": out_path}

    # =========================================================
    # Build tables / load files
    # =========================================================

    def _build_table_specs(self) -> None:
        self.table_specs = [
            TableSpec(
                domain="user",
                name="user_snapshot",
                path=self.paths["user_features"],
                temporal=False,
                time_col=None,
                entity_key=["user_id"],
            ),
            TableSpec(
                domain="price",
                name="ohlcv_last365",
                path=self.paths["price_ohlcv_last365"],
                temporal=True,
                time_col="date",
                entity_key=["종목코드"],
            ),
            TableSpec(
                domain="price",
                name="foreign_snapshot_today",
                path=self.paths["price_foreign_snapshot_today"],
                temporal=False,
                time_col=None,
                entity_key=["종목코드"],
            ),
            TableSpec(
                domain="finance",
                name="financial_features",
                path=self.paths["finance_features_glob"],
                temporal=True,
                time_col="year",
                entity_key=["종목코드"],
            ),
            TableSpec(
                domain="news",
                name="news_features_by_stock",
                path=self.paths["news_features_by_stock"],
                temporal=False,
                time_col=None,
                entity_key=["종목코드"],
            ),
            TableSpec(
                domain="news",
                name="news_raw_merged",
                path=self.paths["news_raw_merged"],
                role="raw_text_reference",
                temporal=True,
                time_col="pubDate",
                entity_key=["종목코드"],
            ),
        ]

        self.plan_table_meta = self.preprocessing_plan.get("tables", {}) or {}
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            plan_meta = self.plan_table_meta.get(key, {}) or {}
            roles = plan_meta.get("roles", {}) or {}

            if "temporal" in plan_meta:
                spec.temporal = plan_meta.get("temporal", spec.temporal)
            if roles.get("time_col"):
                spec.time_col = roles.get("time_col")
            ticker_col = roles.get("ticker_col")
            id_cols = roles.get("id_cols") or []
            entity_key = []
            if ticker_col:
                entity_key.append(ticker_col)
            entity_key.extend([c for c in id_cols if c not in entity_key])
            if entity_key:
                spec.entity_key = entity_key

    def _load_all_tables(self) -> None:
        for spec in self.table_specs:
            key = f"{spec.domain}.{spec.name}"
            if not spec.path:
                continue

            if any(ch in spec.path for ch in ["*", "?", "["]):
                files = sorted(glob.glob(spec.path))
                if not files:
                    raise FileNotFoundError(f"No files matched pattern: {spec.path}")
                df = self._read_and_concat_csvs(files, spec)
            else:
                df = self._read_csv(spec.path)

            self.tables[key] = df

    def _read_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        encodings_to_try = [self.encoding, "utf-8-sig", "cp949", "euc-kr", "latin1"]
        last_err = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                return self._maybe_sample(df)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Failed reading CSV: {path} ({last_err})")

    def _read_and_concat_csvs(self, files: List[str], spec: TableSpec) -> pd.DataFrame:
        parts = []
        for fp in files:
            df = self._read_csv(fp)
            df["_source_file"] = os.path.basename(fp)

            if spec.domain == "finance":
                ticker = self._extract_ticker_from_finance_filename(fp)
                if ticker is not None:
                    df["종목코드"] = str(ticker).zfill(6)

            parts.append(df)

        out = pd.concat(parts, axis=0, ignore_index=True)
        return self._maybe_sample(out)

    def _maybe_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sample_rows and len(df) > self.sample_rows:
            return df.sample(self.sample_rows, random_state=self.random_state)
        return df

    # =========================================================
    # Profiling / key inference
    # =========================================================

    def _profile_all_tables(self) -> None:
        for key, df in self.tables.items():
            spec = self._find_spec(key)
            key_info = self._infer_key_columns(df, spec)
            prof = self._profile_table(df)

            self.key_info_by_table[key] = key_info
            self.table_profiles[key] = prof

    def _infer_key_columns(self, df: pd.DataFrame, spec: Optional[TableSpec]) -> Dict[str, List[str]]:
        cols = list(df.columns)
        lcols = [self._safe_lower(c) for c in cols]

        ticker_candidates, name_candidates, date_candidates, user_id_candidates = [], [], [], []

        for c, lc in zip(cols, lcols):
            if lc in ("ticker", "code", "stock_code", "symbol", "종목코드", "종목"):
                ticker_candidates.append(c)
            if "ticker" in lc or "symbol" in lc or "종목코드" in lc:
                ticker_candidates.append(c)

            if "name" in lc or "company" in lc or "corp" in lc or "종목명" in lc or "기업명" in lc:
                name_candidates.append(c)

            if self.TIME_NAME_PAT.search(lc):
                date_candidates.append(c)

            if lc in ("user_id", "userid", "uid") or "user_id" in lc or "유저" in lc:
                user_id_candidates.append(c)

        if spec and spec.time_col and spec.time_col in cols and spec.time_col not in date_candidates:
            date_candidates.append(spec.time_col)

        if spec and spec.entity_key:
            for c in spec.entity_key:
                if c in cols and c not in ticker_candidates and c == "종목코드":
                    ticker_candidates.append(c)
                if c in cols and c not in user_id_candidates and c == "user_id":
                    user_id_candidates.append(c)

        return {
            "ticker_cols": self._uniq(ticker_candidates),
            "company_name_cols": self._uniq(name_candidates),
            "date_cols": self._uniq(date_candidates),
            "user_id_cols": self._uniq(user_id_candidates),
        }

    def _profile_table(self, df: pd.DataFrame) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "missing_rate_avg": float(df.isna().mean().mean()) if df.size else 0.0,
            "missing_rate_max": float(df.isna().mean().max()) if df.size else 0.0,
            "numeric_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "high_cardinality_cols": [],
            "sparsity_ratio": 0.0,
            "multicollinearity_avg": 0.0,
            "outlier_rate_avg": 0.0,
            "heavy_tail_score_avg": 0.0,
            "datetime_parse_success": {},
        }

        for c in df.columns:
            s = df[c]
            if self._is_datetime_like(s,c):
                info["datetime_cols"].append(c)
            elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                info["numeric_cols"].append(c)
            else:
                info["categorical_cols"].append(c)

            if s.dtype == object:
                nunique = s.nunique(dropna=True)
                if nunique > max(100, int(0.2 * max(len(s), 1))):
                    info["high_cardinality_cols"].append(c)

        if info["numeric_cols"]:
            nums = df[info["numeric_cols"]].apply(self._numeric_series)
            info["sparsity_ratio"] = self._sparsity_ratio(nums)
            info["multicollinearity_avg"] = self._multicollinearity_avg(nums)

            out_rates, ht_scores = [], []
            for c in nums.columns:
                out_rates.append(self._mad_outlier_rate(nums[c]))
                ht_scores.append(self._heavy_tail_score(nums[c]))
            info["outlier_rate_avg"] = float(np.nanmean(out_rates)) if out_rates else 0.0
            info["heavy_tail_score_avg"] = float(np.nanmean(ht_scores)) if ht_scores else 0.0

        for c in info["datetime_cols"]:
            _, rate = self._coerce_datetime(df[c])
            info["datetime_parse_success"][c] = rate

        return info

    # =========================================================
    # Signal building
    # =========================================================

    def _build_selection_signals(
        self,
        primary_table: str,
        primary_df: pd.DataFrame,
        primary_prof: Dict[str, Any],
        primary_key_info: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        task_type = self._infer_task_type()
        plan_ts_diag = self._get_plan_ts_diag(primary_table)

        gpu_available = self._gpu_available()

        signals: Dict[str, Any] = {
            "task_type": task_type,
            "n_rows": primary_prof["n_rows"],
            "n_features": primary_prof["n_cols"],
            "missing_rate_avg": primary_prof["missing_rate_avg"],
            "missing_rate_max": primary_prof["missing_rate_max"],
            "sparsity_ratio": primary_prof["sparsity_ratio"],
            "multicollinearity_avg": primary_prof["multicollinearity_avg"],
            "outlier_rate_max": primary_prof["outlier_rate_avg"],
            "heavy_tail_score_max": primary_prof["heavy_tail_score_avg"],

            # 시계열 신호
            "autocorr_strength_min": 0.0,
            "seasonality_strength_min": 0.0,
            "trend_strength_min": 0.0,
            "trend_strength_max": 1.0,
            "regime_change_score_min": 0.0,
            "regime_change_score_max": 1.0,
            "changepoint_count_min": 0,
            "changepoint_density_min": 0.0,
            "changepoint_density_max": 1.0,
            "nonstationarity_score_min": 0.0,
            "nonstationarity_score_max": 1.0,
            "sequence_length": 0,

            # generic
            "nonlinearity_score_min": 0.0,
            "feature_interaction_score_min": 0.0,
            "horizon_steps": self._infer_horizon_steps(),
            "gpu_available": gpu_available,

            # 디버깅용
            "proxy_col_used": None,
            "proxy_family_used": None,
            "prep_reco_ts_diag_used": bool(plan_ts_diag.get("enabled", False)),
        }

        # --------------------------------------------------
        # 0) 원본 데이터 기반 generic signals
        # --------------------------------------------------
        ht = primary_prof["heavy_tail_score_avg"]
        outlier = primary_prof["outlier_rate_avg"]
        mc = primary_prof["multicollinearity_avg"]

        signals["nonlinearity_score_min"] = float(np.clip(0.3 * ht + 0.4 * outlier + 0.3 * mc, 0, 1))
        signals["feature_interaction_score_min"] = float(np.clip(0.2 * ht + 0.3 * outlier + 0.5 * mc, 0, 1))

        # --------------------------------------------------
        # 1) prep_reco 진단 결과 반영
        #    -> 초기값/참고값으로만 넣음
        # --------------------------------------------------
        if plan_ts_diag.get("enabled"):
            signals["autocorr_strength_min"] = float(plan_ts_diag.get("autocorr_lag1", 0.0))
            signals["seasonality_strength_min"] = float(plan_ts_diag.get("seasonality_strength", 0.0))
            signals["trend_strength_min"] = float(plan_ts_diag.get("trend_strength", 0.0))
            signals["trend_strength_max"] = float(plan_ts_diag.get("trend_strength", 0.0))
            signals["regime_change_score_min"] = float(plan_ts_diag.get("regime_change_score", 0.0))
            signals["regime_change_score_max"] = float(plan_ts_diag.get("regime_change_score", 0.0))
            signals["changepoint_density_min"] = float(plan_ts_diag.get("changepoint_density", 0.0))
            signals["changepoint_density_max"] = float(plan_ts_diag.get("changepoint_density", 0.0))
            signals["nonstationarity_score_min"] = float(plan_ts_diag.get("nonstationarity_score", 0.0))
            signals["nonstationarity_score_max"] = float(plan_ts_diag.get("nonstationarity_score", 0.0))
            signals["sequence_length"] = int(plan_ts_diag.get("median_points_per_entity", 0))

        # --------------------------------------------------
        # 2) 원본 데이터에서 다시 proxy 계산
        #    -> 핵심 시계열 신호는 raw_ts 우선 overwrite
        # --------------------------------------------------
        raw_ts = self._build_raw_ts_signals(primary_table, primary_df, primary_key_info)

        ts_priority_keys = {
            "autocorr_strength_min",
            "seasonality_strength_min",
            "trend_strength_min",
            "regime_change_score_min",
            "changepoint_count_min",
            "changepoint_density_min",
            "nonstationarity_score_min",
            "sequence_length",
        }

        passthrough_keys = {
            "proxy_col_used",
            "proxy_family_used",
        }

        for k, v in raw_ts.items():
            # 문자열/메타 정보는 그대로 기록
            if k in passthrough_keys:
                signals[k] = v
                continue

            # 숫자만 처리
            if not isinstance(v, (int, float, np.integer, np.floating)):
                continue

            # 핵심 시계열 신호는 raw_ts를 우선 사용
            if k in ts_priority_keys:
                signals[k] = float(v)
                continue

            # 나머지는 기존 방식 유지
            if k in signals:
                signals[k] = max(float(signals[k]), float(v))
            else:
                signals[k] = float(v)

        # --------------------------------------------------
        # 3) max 계열 보정
        # --------------------------------------------------
        signals["trend_strength_max"] = float(signals["trend_strength_min"])
        signals["regime_change_score_max"] = float(signals["regime_change_score_min"])
        signals["changepoint_density_max"] = float(signals["changepoint_density_min"])
        signals["nonstationarity_score_max"] = float(signals["nonstationarity_score_min"])

        return signals

    def _build_raw_ts_signals(
        self,
        primary_table: str,
        primary_df: pd.DataFrame,
        primary_key_info: Dict[str, List[str]],
    ) -> Dict[str, Any]:

        out = {
            "autocorr_strength_min": 0.0,
            "seasonality_strength_min": 0.0,
            "trend_strength_min": 0.0,
            "regime_change_score_min": 0.0,
            "changepoint_count_min": 0,
            "changepoint_density_min": 0.0,
            "nonstationarity_score_min": 0.0,
            "sequence_length": 0,
            "proxy_col_used": None,
            "proxy_family_used": None,
        }

        # -------------------------
        # 1. 테이블 조건 체크
        # -------------------------
        spec = self._find_spec(primary_table)
        if spec is None or not spec.temporal:
            return out

        time_col = spec.time_col or (primary_key_info.get("date_cols") or [None])[0]
        if not time_col or time_col not in primary_df.columns:
            return out

        # -------------------------
        # 2. proxy 선택 (방법 B 핵심)
        # -------------------------
        proxy_candidates = self._choose_proxy_cols(primary_df)

        proxy_col = (
            proxy_candidates.get("return_like")
            or proxy_candidates.get("level_like")
            or proxy_candidates.get("fundamental_like")
        )

        if not proxy_col:
            return out

        # proxy 기록 (디버깅 핵심)
        if proxy_col == proxy_candidates.get("return_like"):
            out["proxy_family_used"] = "return_like"
        elif proxy_col == proxy_candidates.get("level_like"):
            out["proxy_family_used"] = "level_like"
        elif proxy_col == proxy_candidates.get("fundamental_like"):
            out["proxy_family_used"] = "fundamental_like"

        out["proxy_col_used"] = proxy_col

        # -------------------------
        # 3. entity key 설정
        # -------------------------
        entity_cols = []
        if spec.entity_key:
            entity_cols.extend([c for c in spec.entity_key if c in primary_df.columns])

        # -------------------------
        # 4. datetime 정리
        # -------------------------
        work = primary_df.copy()
        parsed, parse_rate = self._coerce_datetime(work[time_col])

        if parse_rate < 0.5:
            return out

        work[time_col] = parsed
        work = work[work[time_col].notna()].copy()

        # -------------------------
        # 5. 시계열 그룹 추출
        # -------------------------
        grouped_series = self._extract_groupwise_series(
            df=work,
            time_col=time_col,
            entity_cols=entity_cols,
            value_col=proxy_col,
        )

        if not grouped_series:
            return out

        # -------------------------
        # 6. signal 계산
        # -------------------------
        ac1_list, seas_list, trend_list = [], [], []
        regime_list, cp_dens_list, nonstat_list = [], [], []
        lens, cp_count_list = [], []

        for s in grouped_series:
            x = self._numeric_series(s).dropna()

            if len(x) < 30:
                continue

            arr = x.to_numpy(dtype=float)

            # 🔥 핵심 개선: return_like일 경우 안정화
            if out["proxy_family_used"] == "return_like":
                # 평균 제거 → trend 과대평가 방지
                arr = arr - np.mean(arr)

            lens.append(len(arr))

            ac1_list.append(self._safe_autocorr(arr, 1))
            seas_list.append(self._seasonality_strength(arr))
            trend_list.append(self._trend_strength(arr))
            regime_list.append(self._regime_change_score(arr))

            cp_density, cp_count = self._changepoint_density_and_count(arr)
            cp_dens_list.append(cp_density)
            cp_count_list.append(cp_count)

            nonstat_list.append(self._nonstationarity_score(arr))

        if not lens:
            return out

        # -------------------------
        # 7. robust aggregation
        # -------------------------
        out["autocorr_strength_min"] = self._robust_mean(ac1_list)
        out["seasonality_strength_min"] = self._robust_mean(seas_list)
        out["trend_strength_min"] = self._robust_mean(trend_list)
        out["regime_change_score_min"] = self._robust_mean(regime_list)
        out["changepoint_density_min"] = self._robust_mean(cp_dens_list)
        out["changepoint_count_min"] = int(np.median(cp_count_list)) if cp_count_list else 0
        out["nonstationarity_score_min"] = self._robust_mean(nonstat_list)
        out["sequence_length"] = int(np.median(lens))

        return out

    # =========================================================
    # Model scoring
    # =========================================================

    def _load_model_db(self, model_db_dir: str) -> List[Dict[str, Any]]:
        all_models: List[Dict[str, Any]] = []
        if not os.path.exists(model_db_dir):
            return all_models

        for fn in sorted(os.listdir(model_db_dir)):
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(model_db_dir, fn)
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_models.extend(data)
        return all_models

    def _score_model(self, model: Dict[str, Any], signals: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        score = 0.0
        reasons: List[str] = []
        notes: List[str] = []

        req = model.get("data_requirements", {}) or {}
        app = model.get("applicability_signals", {}) or {}
        excl = model.get("exclusion_rules", []) or {}

        task = signals.get("task_type")
        target_types = model.get("target_type", []) or []
        if task and task not in target_types:
            return -1e9, reasons, [f"exclude: task_type={task} not in {target_types}"]

        score += 20
        reasons.append("target_type matched (+20)")

        n_rows = float(signals.get("n_rows", 0))
        min_rows = float(req.get("min_rows", 0))
        if n_rows < min_rows:
            return -1e9, reasons, [f"exclude: n_rows({n_rows}) < min_rows({min_rows})"]

        score += min(20, (n_rows / (min_rows + 1e-9)) * 5)
        reasons.append(f"n_rows sufficient (+{min(20, (n_rows / (min_rows + 1e-9)) * 5):.1f})")

        missing_rate = float(signals.get("missing_rate_avg", 0))
        supports_missing = bool(req.get("supports_missing", True))
        if missing_rate > 0 and not supports_missing:
            score -= 15
            reasons.append("missing present but model doesn't support missing (-15)")
        else:
            score += 5
            reasons.append("missing compatibility (+5)")

        if "n_rows_range" in app:
            lo, hi = app["n_rows_range"]
            if lo <= n_rows <= hi:
                score += 10
            else:
                score -= 5
            reasons.append("n_rows_range fit (±10)")

        def th_min(k: str, w: float = 8.0) -> None:
            nonlocal score
            if k in app and k in signals:
                if float(signals[k]) >= float(app[k]):
                    score += w
                    reasons.append(f"{k} >= {app[k]} (+{w})")
                else:
                    score -= w
                    reasons.append(f"{k} < {app[k]} (-{w})")

        def th_max(k: str, w: float = 8.0) -> None:
            nonlocal score
            if k in app and k in signals:
                if float(signals[k]) <= float(app[k]):
                    score += w
                    reasons.append(f"{k} <= {app[k]} (+{w})")
                else:
                    score -= w
                    reasons.append(f"{k} > {app[k]} (-{w})")

        for k in [
            "seasonality_strength_min",
            "autocorr_strength_min",
            "trend_strength_min",
            "regime_change_score_min",
            "nonstationarity_score_min",
            "nonlinearity_score_min",
            "feature_interaction_score_min",
            "changepoint_density_min",
            "changepoint_count_min",
            "sequence_length",
            "horizon_steps",
        ]:
            th_min(k, 8)

        for k in [
            "missing_rate_max",
            "outlier_rate_max",
            "heavy_tail_score_max",
            "multicollinearity_avg_max",
            "seasonality_strength_max",
            "trend_strength_max",
            "regime_change_score_max",
            "changepoint_density_max",
            "nonstationarity_score_max",
        ]:
            th_max(k, 8)

        # *_range 처리
        range_keys = [
            ("sequence_length_range", "sequence_length"),
            ("horizon_steps_range", "horizon_steps"),
        ]
        for app_key, signal_key in range_keys:
            if app_key in app and signal_key in signals:
                lo, hi = app[app_key]
                v = float(signals[signal_key])
                if lo <= v <= hi:
                    score += 8
                    reasons.append(f"{signal_key} in range {app_key} (+8)")
                else:
                    score -= 8
                    reasons.append(f"{signal_key} out of range {app_key} (-8)")

        gpu_required = bool(model.get("operational", {}).get("gpu_required", False))
        if gpu_required and not bool(signals.get("gpu_available", False)):
            score -= 25
            notes.append("deprioritize: gpu_required but gpu not available (-25)")

        for rule in excl:
            cond = rule.get("if", {}) or {}
            action = str(rule.get("then", ""))
            ok = True

            for ck, cv in cond.items():
                if ck.endswith("_gt"):
                    k = ck[:-3]
                    if float(signals.get(k, -1e9)) <= float(cv):
                        ok = False
                elif ck.endswith("_lt"):
                    k = ck[:-3]
                    if float(signals.get(k, 1e9)) >= float(cv):
                        ok = False
                else:
                    if signals.get(ck) != cv:
                        ok = False

            if ok:
                if "exclude" in action:
                    return -1e9, reasons, [f"exclude rule triggered: {cond} -> {action}"]
                if "deprioritize" in action:
                    score -= 10
                    notes.append(f"deprioritize rule: {cond} -> {action} (-10)")

        # family-level bonus / penalty
        family = str(model.get("family", "")).lower()
        if family in {"statistical_time_series", "seasonal_statistical_time_series"}:
            if signals.get("regime_change_score_min", 0.0) > 0.35:
                score -= 6
                notes.append("regime change relatively high for classical statistical TS (-6)")
        if "tree_boosting_time_series" in family or "tree_ensemble_time_series" in family:
            if signals.get("nonlinearity_score_min", 0.0) > 0.5:
                score += 4
                notes.append("nonlinearity favorable for lag-based tree models (+4)")
        if "deep" in family:
            if signals.get("sequence_length", 0) >= 120:
                score += 4
                notes.append("sufficient sequence length for deep sequence model (+4)")

        return float(score), reasons, notes

    # =========================================================
    # Output helpers
    # =========================================================

    def _build_column_preprocessing(
        self,
        df: pd.DataFrame,
        key_info: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        col_map: Dict[str, Any] = {}
        ticker_cols = set(key_info.get("ticker_cols", []))
        name_cols = set(key_info.get("company_name_cols", []))
        date_cols = set(key_info.get("date_cols", []))
        user_id_cols = set(key_info.get("user_id_cols", []))

        for c in df.columns:
            s = df[c]
            lc = self._safe_lower(c)

            item = {"column": c, "role": None, "keep": True, "actions": []}

            if c in name_cols or "종목명" in lc or "기업명" in lc or "company_name" in lc or lc.endswith("_name"):
                item["role"] = "company_name"
                item["keep"] = False
                item["actions"].append("drop_column")
                col_map[c] = item
                continue

            if c in ticker_cols:
                item["role"] = "ticker_id"
                item["keep"] = True
                item["actions"].append("preserve_as_id")
                col_map[c] = item
                continue

            if c in user_id_cols:
                item["role"] = "user_id"
                item["keep"] = True
                item["actions"].append("preserve_as_id")
                col_map[c] = item
                continue

            if self._is_leakage_candidate(c):
                item["role"] = "leakage_candidate"
                item["keep"] = False
                item["actions"].append("drop_or_mask_leakage")
                col_map[c] = item
                continue

            if c in date_cols or self._is_datetime_like(s, c):
                item["role"] = "datetime"
                item["actions"].append("to_datetime")
                item["actions"].append("date_floor")
                col_map[c] = item
                continue

            if s.dtype == object:
                nunique = s.nunique(dropna=True)
                avg_len = float(s.dropna().astype(str).head(100).map(len).mean()) if s.dropna().shape[0] else 0.0
                if avg_len >= 30 or "text" in lc or "title" in lc or "content" in lc or "description" in lc:
                    item["role"] = "text"
                    item["actions"].append("text_optional_drop_or_embed")
                else:
                    item["role"] = "categorical"
                    item["actions"].append("encoding")
                    if nunique > 1000:
                        item["actions"].append("high_cardinality_strategy")
                col_map[c] = item
                continue

            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                item["role"] = "numeric"
                x = self._numeric_series(s)
                if self._heavy_tail_score(x) > 0.5 or self._mad_outlier_rate(x) > 0.1:
                    item["actions"].append("robust_scaling_optional")
                    item["actions"].append("winsorize_optional")
                else:
                    item["actions"].append("scaling_optional")
                col_map[c] = item
                continue

            item["role"] = "unknown"
            item["actions"].append("inspect")
            col_map[c] = item

        return col_map

    def _decide_missing_policy(
        self,
        table_profile: Dict[str, Any],
        selected_model: Dict[str, Any],
        key_info: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        supports_missing = bool(selected_model.get("data_requirements", {}).get("supports_missing", True))
        miss_avg = float(table_profile.get("missing_rate_avg", 0))
        miss_max = float(table_profile.get("missing_rate_max", 0))

        policy = {
            "strategy": None,
            "per_type": {"numeric": None, "categorical": None, "datetime": None},
            "drop_row_threshold": 0.7,
            "drop_col_threshold": 0.9,
            "id_columns_never_impute": list(set(key_info.get("ticker_cols", []) + key_info.get("user_id_cols", []))),
            "notes": [],
        }

        if miss_avg == 0:
            policy["strategy"] = "no_action"
            policy["per_type"] = {"numeric": "none", "categorical": "none", "datetime": "none"}
            return policy

        if supports_missing:
            policy["strategy"] = "model_can_handle_missing_but_still_standardize"
            policy["per_type"] = {
                "numeric": "median_impute_optional",
                "categorical": "missing_token",
                "datetime": "drop_invalid_datetime_rows",
            }
            policy["notes"].append("Model supports missing, but normalize missing representation consistently.")
        else:
            policy["strategy"] = "impute_required"
            policy["per_type"] = {
                "numeric": "median_impute",
                "categorical": "most_frequent_or_missing_token",
                "datetime": "drop_invalid_datetime_rows",
            }
            policy["notes"].append("Selected model does not support missing -> imputation required.")

        if miss_max > 0.95:
            policy["notes"].append("Some columns have extremely high missing rate; consider dropping non-ID columns.")

        return policy

    def _decide_date_policy(
        self,
        df: pd.DataFrame,
        key_info: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        date_cols = key_info.get("date_cols", [])
        best, best_rate = None, -1.0

        for c in date_cols:
            if c in df.columns:
                _, rate = self._coerce_datetime(df[c])
                if rate > best_rate:
                    best_rate = rate
                    best = c

        ticker_cols = key_info.get("ticker_cols", [])
        keys: List[str] = []
        if ticker_cols:
            keys.append(ticker_cols[0])
        if best:
            keys.append(best)

        return {
            "datetime_column": best,
            "parse_success_rate": best_rate if best else 0.0,
            "timezone": "Asia/Seoul",
            "dedup_keys": keys,
            "dedup_strategy": "keep_last",
            "notes": [f"Deduplicate rows by keys={keys}, strategy=keep_last."] if keys else ["No reliable dedup keys found."],
        }

    def _build_join_keys(self, date_policy: Dict[str, Any]) -> Dict[str, Any]:
        join_keys = {
            "price": {"keys": [], "preferred_table": None},
            "news": {"keys": [], "preferred_table": None},
            "user": {"keys": [], "preferred_table": None},
            "finance": {"keys": [], "preferred_table": None},
        }

        dt_col = date_policy.get("datetime_column")
        primary_key_info = self.key_info_by_table.get(self._choose_primary_table(), {})
        ticker_cols = primary_key_info.get("ticker_cols", [])

        if ticker_cols:
            join_keys["price"]["keys"].append(ticker_cols[0])
            join_keys["news"]["keys"].append(ticker_cols[0])
            join_keys["finance"]["keys"].append(ticker_cols[0])

        if dt_col:
            join_keys["price"]["keys"].append(dt_col)
            join_keys["news"]["keys"].append(dt_col)
            join_keys["finance"]["keys"].append(dt_col)

        for key in self.tables.keys():
            dom = self._table_domain_from_key(key)
            if dom in join_keys and join_keys[dom]["preferred_table"] is None:
                join_keys[dom]["preferred_table"] = key

        return join_keys

    def _build_validation_warnings(
        self,
        primary_prof: Dict[str, Any],
        column_pre_map: Dict[str, Any],
    ) -> List[str]:
        warnings: List[str] = []
        leakage_cols = [c for c, meta in column_pre_map.items() if meta.get("role") == "leakage_candidate"]

        if leakage_cols:
            warnings.append(f"Leakage candidates detected and default-dropped: {leakage_cols[:10]}")
        if primary_prof["missing_rate_max"] > 0.95:
            warnings.append("Some columns have >95% missing. Consider dropping non-ID columns.")

        return warnings

    # =========================================================
    # Prep reco integration
    # =========================================================

    def _get_plan_ts_diag(self, table_key: str) -> Dict[str, Any]:
        table_meta = (self.preprocessing_plan.get("tables", {}) or {}).get(table_key, {}) or {}
        reqs = table_meta.get("requirements", {}) or {}
        return reqs.get("time_series_diagnostics", {}) or {}

    def _infer_task_type(self) -> str:
        problem_type = None
        if isinstance(self.preprocessing_plan, dict):
            problem_schema = self.preprocessing_plan.get("problem_schema", {}) or {}
            problem_type = problem_schema.get("problem_type")

        if not problem_type and isinstance(self.data_contract, dict):
            problem_type = self.data_contract.get("problem_type") or self.data_contract.get("task_type")

        if problem_type:
            pl = str(problem_type).lower()
            if "ts" in pl or "time_series" in pl or "forecast" in pl:
                return "time_series"

        return "time_series"

    def _infer_horizon_steps(self) -> int:
        problem_schema = self.preprocessing_plan.get("problem_schema", {}) or {}
        evaluation = problem_schema.get("evaluation", {}) or {}
        constraints = problem_schema.get("constraints", {}) or {}

        for obj in [evaluation, constraints]:
            for k in ["horizon_steps", "forecast_horizon", "pred_horizon"]:
                if k in obj:
                    try:
                        return int(obj[k])
                    except Exception:
                        pass

        return 20

    # =========================================================
    # Series helpers
    # =========================================================

    def _choose_proxy_cols(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        def first_existing(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in df.columns and c in num_cols:
                    return c
            return None

        return {
            "return_like": first_existing(["daily_return", "log_return"]),
            "level_like": first_existing(["close", "종가"]),
            "fundamental_like": first_existing(["revenue_yoy", "operating_income_yoy"]),
        }

    def _extract_groupwise_series(
        self,
        df: pd.DataFrame,
        time_col: str,
        entity_cols: List[str],
        value_col: str,
    ) -> List[pd.Series]:
        out: List[pd.Series] = []

        if entity_cols:
            grouped = df.groupby(entity_cols, dropna=False)
            for _, sub in grouped:
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
        return float(np.clip(abs(np.corrcoef(x, y)[0, 1]), 0.0, 1.0))

    def _seasonality_strength(self, arr: np.ndarray) -> float:
        candidates = [5, 20, 60]
        vals = []
        for lag in candidates:
            if len(arr) > lag + 10:
                vals.append(self._safe_autocorr(arr, lag))
        return float(np.clip(max(vals) if vals else 0.0, 0.0, 1.0))

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
        return float(np.clip(float(diffs.mean() / base), 0.0, 1.0))

    def _changepoint_density_and_count(self, arr: np.ndarray) -> Tuple[float, int]:
        if len(arr) < 40:
            return 0.0, 0
        s = pd.Series(arr)
        w = max(10, len(arr) // 8)
        rm = s.rolling(w).mean().dropna()
        if len(rm) < 5:
            return 0.0, 0

        diffs = rm.diff().abs().dropna()
        thr = float(diffs.mean() + 2.0 * diffs.std()) if len(diffs) else np.nan
        if np.isnan(thr):
            return 0.0, 0

        cp = int((diffs > thr).sum())
        density = float(np.clip(cp / max(len(arr), 1), 0.0, 1.0))
        return density, cp

    def _nonstationarity_score(self, arr: np.ndarray) -> float:
        if len(arr) < 40:
            return 0.0
        first = arr[: len(arr) // 2]
        second = arr[len(arr) // 2 :]

        scale = np.std(arr) + 1e-9
        mean_shift = abs(np.mean(first) - np.mean(second)) / scale
        std_shift = abs(np.std(first) - np.std(second)) / scale
        score = 0.6 * mean_shift + 0.4 * std_shift
        return float(np.clip(score, 0.0, 1.0))

    def _robust_mean(self, xs: List[float]) -> float:
        vals = [float(x) for x in xs if x is not None and not np.isnan(x)]
        if not vals:
            return 0.0
        return float(np.median(vals))

    # =========================================================
    # Generic helpers
    # =========================================================

    def _choose_primary_table(self) -> str:
        # 1순위: prep reco 상 price.ohlcv_last365
        preferred = "price.ohlcv_last365"
        if preferred in self.tables:
            return preferred

        # 2순위: temporal + rows 큰 테이블
        temporal_keys = []
        for key in self.tables.keys():
            spec = self._find_spec(key)
            if spec and spec.temporal:
                temporal_keys.append(key)

        if temporal_keys:
            return max(temporal_keys, key=lambda k: self.table_profiles[k]["n_rows"])

        # 3순위: 전체 rows 최대
        return max(self.tables.keys(), key=lambda k: self.table_profiles[k]["n_rows"])

    def _find_spec(self, table_key: str) -> Optional[TableSpec]:
        for spec in self.table_specs:
            if f"{spec.domain}.{spec.name}" == table_key:
                return spec
        return None

    def _table_domain_from_key(self, table_key: str) -> str:
        return table_key.split(".", 1)[0] if "." in table_key else "unknown"

    def _extract_ticker_from_finance_filename(self, fp: str) -> Optional[str]:
        base = os.path.basename(fp)
        m = re.match(r"^(\d{6})", base)
        if m:
            return m.group(1)
        m2 = re.match(r"^(\d{4,8})", base)
        if m2:
            return m2.group(1)
        return None

    def _is_leakage_candidate(self, colname: str) -> bool:
        lc = self._safe_lower(colname)
        return any(k in lc for k in self.LEAKAGE_KEYWORDS)

    def _gpu_available(self) -> bool:
        try:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible not in (None, "", "-1"):
                return True

            nvidia_smi = shutil.which("nvidia-smi")
            if not nvidia_smi:
                return False

            result = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                check=False
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if result.returncode != 0:
                return False

            return ("GPU" in stdout) or ("NVIDIA" in stdout) or ("GPU" in stderr) or ("NVIDIA" in stderr)

        except Exception:
            return False

    def _read_json_if_exists(self, path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _abspath(self, p: Optional[str], is_pattern: bool = False) -> Optional[str]:
        if p is None:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    def _safe_lower(self, s: str) -> str:
        return (s or "").strip().lower()

    def _uniq(self, xs: List[str]) -> List[str]:
        out: List[str] = []
        for x in xs:
            if x not in out:
                out.append(x)
        return out

    def _is_datetime_like(self, series: pd.Series, col_name: Optional[str] = None) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if series.dtype != object:
            return False

        col_l = self._safe_lower(col_name or "")

        # 1) 컬럼명 기준 즉시 제외
        hard_exclude_name_keywords = [
            "link", "url", "originallink",
            "description", "title", "content", "text"
        ]
        if any(k in col_l for k in hard_exclude_name_keywords):
            return False

        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False

        hits = 0
        url_like = 0
        long_text_like = 0

        for v in sample:
            v2 = v.strip().lower()

            # 2) URL 냄새 강하면 datetime 후보에서 제외
            if any(tok in v2 for tok in ["http://", "https://", "www.", ".com", ".co.kr", ".kr/"]):
                url_like += 1
                continue

            # 3) 너무 긴 텍스트는 제외
            if len(v2) > 80:
                long_text_like += 1
                continue

            # 4) 진짜 날짜 패턴만 허용
            is_date = bool(re.search(r"^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}$", v2))
            is_datetime = bool(re.search(r"^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}[ t]\d{1,2}:\d{2}(:\d{2})?$", v2))
            is_compact = bool(re.search(r"^\d{8}$", v2))
            is_rfc822 = bool(re.search(
                r"^[a-z]{3},\s+\d{1,2}\s+[a-z]{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+[+\-]\d{4}$",
                v2
            ))

            if is_date or is_datetime or is_compact or is_rfc822:
                hits += 1

        n = max(len(sample), 1)

        # URL/긴 텍스트 비율이 높으면 datetime 아님
        if url_like / n >= 0.3:
            return False
        if long_text_like / n >= 0.5:
            return False

        return hits / n >= 0.6

    def _coerce_datetime(self, series: pd.Series) -> Tuple[pd.Series, float]:
        if series.dtype.kind == "M":
            return series, 1.0
        s = series.copy()
        if s.dtype != object:
            s = s.astype(str)
        s2 = s.replace({None: np.nan})
        parsed = pd.to_datetime(s2, errors="coerce", utc=False)
        rate = parsed.notna().mean() if len(parsed) else 0.0
        return parsed, float(rate)

    def _numeric_series(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        return pd.to_numeric(series, errors="coerce")

    def _mad_outlier_rate(self, x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 30:
            return 0.0
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0:
            return 0.0
        z = (x - med).abs() / (1.4826 * mad)
        return float((z > 3.5).mean())

    def _heavy_tail_score(self, x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 50:
            return 0.0
        k = float(pd.Series(x).kurtosis())
        if k > 1.5:
            k_excess = k - 3.0
        else:
            k_excess = k
        return float(np.clip(k_excess / 10.0, 0, 1))

    def _sparsity_ratio(self, df: pd.DataFrame) -> float:
        nums = df.select_dtypes(include=[np.number])
        if nums.empty:
            return 0.0
        arr = nums.to_numpy()
        mask = ~np.isnan(arr)
        denom = mask.sum()
        if denom == 0:
            return 0.0
        return float((arr[mask] == 0).sum() / denom)

    def _multicollinearity_avg(self, df: pd.DataFrame) -> float:
        nums = df.select_dtypes(include=[np.number])
        if nums.shape[1] < 3:
            return 0.0
        corr = nums.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        vals = upper.stack().values
        if len(vals) == 0:
            return 0.0
        return float(np.nanmean(vals))