from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.agent_base import BaseAgent
from core.artifacts import ArtifactPaths
from core.context import RunContext
from core.result import StageResult


@dataclass
class CandidateScoringArtifacts:
    run_dir: Path
    output_dir: Path
    result_path: Path
    all_candidates_path: Path
    top10_path: Path
    manifest_path: Path
    output_files: List[str]


class CandidateScoringAgent(BaseAgent):
    """
    Candidate Scoring Agent

    기존 ModelInferenceAgent 대체 버전.

    목적:
    - return_20d, rank_20d, top_quantile_20d 같은 미래 라벨을 만들지 않는다.
    - 머신러닝 모델을 학습하지 않는다.
    - 최신 기준일(as_of_date)의 종목 데이터를 기준으로 시장 흐름 기반 후보 점수를 계산한다.
    - MarketFlowAgent 결과의 deterministic_snapshot과 candidate_scoring_hints를 점수에 반영한다.
    - 최종 리포트에 들어갈 TOP10 후보 종목과 점수 근거를 생성한다.
    """

    stage = "candidate_scoring"
    version = "1.3-top100-liquidity-name-aware-score"

    def __init__(self, encoding: str = "utf-8", top_k: int = 10):
        self.encoding = encoding
        self.top_k = top_k

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = CandidateScoringArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            result_path=output_dir / "candidate_scoring_result.json",
            all_candidates_path=output_dir / "candidate_scoring_all_candidates.csv",
            top10_path=output_dir / "candidate_scoring_top10.csv",
            manifest_path=output_dir / "manifest.json",
            output_files=[],
        )

        ctx.logger.info("[CandidateScoringAgent] Starting candidate scoring.")

        planner_dir = run_dir / "report_target_planner"

        paths = {
            "report_target_spec": planner_dir / "report_target_spec.json",
            "report_feature_contract": planner_dir / "report_feature_contract.json",
            "candidate_scoring_plan": planner_dir / "candidate_scoring_plan.json",

            "market_flow_result": self._find_single_file(
                run_dir,
                "market_flow_result.json",
                ctx,
            ),

            "ohlcv": self._find_single_file(
                run_dir,
                "preprocessed__*__price__ohlcv_last365.csv",
                ctx,
            ),
            "news_feat": self._find_single_file(
                run_dir,
                "preprocessed__*__news__news_features_by_stock.csv",
                ctx,
            ),
            "foreign": self._find_single_file(
                run_dir,
                "preprocessed__*__price__foreign_snapshot_today.csv",
                ctx,
            ),
            "finance": self._find_single_file(
                run_dir,
                "preprocessed__*__finance__financial_features.csv",
                ctx,
            ),
            "user": self._find_single_file(
                run_dir,
                "preprocessed__*__user__user_snapshot.csv",
                ctx,
            ),
            # IngestAgent가 생성한 유동성 TOP100 파일만 종목명 매핑 파일로 사용한다.
            # 예: top100_liquidity_20260617.csv
            "stock_master": self._find_single_file(
                run_dir,
                "top100_liquidity_*.csv",
                ctx,
            ),

            "market_json": self._find_single_file(
                run_dir,
                "market_analysis_result.json",
                ctx,
            ),
            "news_json_rag": self._find_single_file(
                run_dir,
                "news_invest_rag_result.json",
                ctx,
            ),
            "news_json": self._find_single_file(
                run_dir,
                "news_invest_result.json",
                ctx,
            ),
            "risk_json": self._find_single_file(
                run_dir,
                "risk_score_result.json",
                ctx,
            ),
        }

        self._require_files(
            [
                paths["report_target_spec"],
                paths["report_feature_contract"],
                paths["candidate_scoring_plan"],
                paths["ohlcv"],
            ]
        )

        report_target_spec = self._read_json(paths["report_target_spec"])
        report_feature_contract = self._read_json(paths["report_feature_contract"])
        candidate_scoring_plan = self._read_json(paths["candidate_scoring_plan"])

        market_flow_result = self._safe_read_json(paths["market_flow_result"], ctx)
        market_json = self._safe_read_json(paths["market_json"], ctx)

        news_json_path = paths["news_json_rag"] or paths["news_json"]
        news_json = self._safe_read_json(news_json_path, ctx)

        risk_json = self._safe_read_json(paths["risk_json"], ctx)

        candidates = self._build_candidate_table(
            ctx=ctx,
            paths=paths,
            report_feature_contract=report_feature_contract,
            market_json=market_json,
            news_json=news_json,
            risk_json=risk_json,
        )

        if candidates.empty:
            raise ValueError("Candidate universe is empty after building candidate table.")

        scored = self._score_candidates(
            df=candidates,
            candidate_scoring_plan=candidate_scoring_plan,
            market_flow_result=market_flow_result,
            ctx=ctx,
        )

        scored = self._add_explanations(scored)

        scored = scored.sort_values(
            "market_flow_candidate_score",
            ascending=False,
        ).reset_index(drop=True)

        scored["rank"] = np.arange(1, len(scored) + 1)

        top_k = int(
            candidate_scoring_plan
            .get("selection_policy", {})
            .get("top_k", self.top_k)
        )

        top10 = scored.head(top_k).copy()

        self._save_csv(scored, artifacts.all_candidates_path)
        self._save_csv(top10, artifacts.top10_path)

        result = self._build_result_json(
            scored=scored,
            top10=top10,
            report_target_spec=report_target_spec,
            candidate_scoring_plan=candidate_scoring_plan,
            market_flow_result=market_flow_result,
            paths=paths,
        )

        self._save_json(result, artifacts.result_path)

        artifacts.output_files = [
            str(artifacts.result_path),
            str(artifacts.all_candidates_path),
            str(artifacts.top10_path),
            str(artifacts.manifest_path),
        ]

        manifest = {
            "stage": self.stage,
            "version": self.version,
            "created_at_utc": self._now_utc(),
            "input_files": {
                k: str(v) if v is not None else None
                for k, v in paths.items()
            },
            "output_files": artifacts.output_files,
            "report_type": report_target_spec.get("report_type"),
            "ranking_target": report_target_spec.get("ranking_target"),
            "prediction_horizon": report_target_spec.get("prediction_horizon"),
            "ranking_scope": report_target_spec.get("ranking_scope"),
            "as_of_date": result.get("as_of_date"),
            "n_candidates": int(len(scored)),
            "top_k": int(len(top10)),
            "notes": [
                "This stage replaces ModelInferenceAgent for report generation.",
                "This stage does not train ML models.",
                "This stage does not generate return_20d or rank_20d labels.",
                "market_flow_candidate_score is a current-context candidate score, not a future return prediction.",
                "MarketFlowAgent deterministic_snapshot is used for additional score adjustments.",
                "Percentile scoring direction is fixed so higher_better features receive higher scores.",
                "High-risk candidates are capped by risk_level/overall_risk_score.",
                "News momentum is adjusted by direct article relevance to the stock.",
                "Company names are mapped only from top100_liquidity_*.csv generated by IngestAgent.",
            ],
        }

        self._save_json(manifest, artifacts.manifest_path)

        outputs = {
            "output_dir": str(output_dir),
            "candidate_scoring_result": str(artifacts.result_path),
            "all_candidates": str(artifacts.all_candidates_path),
            "top10_candidates": str(artifacts.top10_path),
            "manifest": str(artifacts.manifest_path),
        }

        metrics = {
            "n_candidates": int(len(scored)),
            "top_k": int(len(top10)),
            "as_of_date": result.get("as_of_date"),
            "score_min": self._safe_float(scored["market_flow_candidate_score"].min()),
            "score_max": self._safe_float(scored["market_flow_candidate_score"].max()),
            "score_mean": self._safe_float(scored["market_flow_candidate_score"].mean()),
            "has_market_flow_result": market_flow_result is not None,
            "has_market_json": market_json is not None,
            "has_news_json": news_json is not None,
            "has_risk_json": risk_json is not None,
        }

        ctx.logger.info("[CandidateScoringAgent] Completed candidate scoring.")

        return self._make_stage_result(
            status="success",
            message="Candidate scoring completed.",
            metrics=metrics,
            outputs=outputs,
        )

    # =========================================================
    # Candidate table builder
    # =========================================================

    def _build_candidate_table(
        self,
        ctx: RunContext,
        paths: Dict[str, Optional[Path]],
        report_feature_contract: Dict[str, Any],
        market_json: Optional[Dict[str, Any]],
        news_json: Optional[Dict[str, Any]],
        risk_json: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        ohlcv = self._read_csv(paths["ohlcv"])

        ticker_col = self._infer_col(ohlcv, ["종목코드", "ticker", "code", "symbol"])
        date_col = self._infer_col(ohlcv, ["date", "datetime", "dt", "일자", "날짜"])
        close_col = self._infer_col(ohlcv, ["close", "Close", "종가"])

        if ticker_col is None or date_col is None or close_col is None:
            raise ValueError("OHLCV must include ticker/date/close columns.")

        ohlcv = ohlcv.copy()
        ohlcv[ticker_col] = ohlcv[ticker_col].apply(self._zfill6)
        ohlcv[date_col] = pd.to_datetime(ohlcv[date_col], errors="coerce")
        ohlcv = ohlcv.dropna(subset=[date_col])
        ohlcv = ohlcv.sort_values([ticker_col, date_col]).reset_index(drop=True)

        latest_date = ohlcv[date_col].max()

        latest = ohlcv[ohlcv[date_col] == latest_date].copy()
        latest = (
            latest.sort_values([ticker_col, date_col])
            .groupby(ticker_col, as_index=False)
            .tail(1)
            .copy()
        )

        latest = latest.rename(
            columns={
                ticker_col: "ticker",
                date_col: "as_of_date",
            }
        )

        latest["ticker"] = latest["ticker"].apply(self._zfill6)
        latest["as_of_date"] = pd.to_datetime(latest["as_of_date"], errors="coerce")

        if close_col in latest.columns and "latest_close" not in latest.columns:
            latest["latest_close"] = pd.to_numeric(latest[close_col], errors="coerce")

        latest = self._ensure_company_name(latest)

        ctx.logger.info(
            f"[CandidateScoringAgent] Latest candidate base: "
            f"date={latest_date.date() if pd.notna(latest_date) else None}, "
            f"rows={len(latest)}"
        )

        latest = self._join_latest_csv(
            base=latest,
            path=paths["news_feat"],
            source_name="news_features",
            ctx=ctx,
            latest_by_date=False,
            latest_by_year=False,
            conflict_prefix="news",
        )

        latest = self._join_latest_csv(
            base=latest,
            path=paths["foreign"],
            source_name="foreign_snapshot",
            ctx=ctx,
            latest_by_date=True,
            latest_by_year=False,
            conflict_prefix="foreign",
        )

        latest = self._join_latest_csv(
            base=latest,
            path=paths["finance"],
            source_name="finance_features",
            ctx=ctx,
            latest_by_date=False,
            latest_by_year=True,
            conflict_prefix="finance",
        )

        latest = self._merge_stock_master_names(
            base=latest,
            path=paths.get("stock_master"),
            ctx=ctx,
        )

        market_features = self._extract_market_json_features(market_json)
        latest = self._merge_optional_features(
            base=latest,
            features=market_features,
            source_name="market_analysis_json",
            ctx=ctx,
        )

        news_features = self._extract_news_json_features(news_json)
        latest = self._merge_optional_features(
            base=latest,
            features=news_features,
            source_name="news_invest_json",
            ctx=ctx,
        )

        risk_features = self._extract_risk_json_features(risk_json)
        latest = self._merge_optional_features(
            base=latest,
            features=risk_features,
            source_name="risk_score_json",
            ctx=ctx,
        )

        latest = self._ensure_company_name(latest)
        latest = self._compute_missing_derived_features(latest)

        return latest

    def _join_latest_csv(
        self,
        base: pd.DataFrame,
        path: Optional[Path],
        source_name: str,
        ctx: RunContext,
        latest_by_date: bool,
        latest_by_year: bool,
        conflict_prefix: str,
    ) -> pd.DataFrame:
        if path is None or not path.exists():
            ctx.logger.info(f"[CandidateScoringAgent] SKIP {source_name}: missing")
            return base

        src = self._read_csv(path)

        if src.empty:
            ctx.logger.info(f"[CandidateScoringAgent] SKIP {source_name}: empty")
            return base

        ticker_col = self._infer_col(src, ["종목코드", "ticker", "code", "symbol"])

        if ticker_col is None:
            ctx.logger.info(f"[CandidateScoringAgent] SKIP {source_name}: no ticker column")
            return base

        src = src.copy()
        src[ticker_col] = src[ticker_col].apply(self._zfill6)

        if latest_by_date:
            date_col = self._infer_col(src, ["date", "datetime", "dt", "일자", "날짜"])

            if date_col:
                src[date_col] = pd.to_datetime(src[date_col], errors="coerce")
                src = (
                    src.dropna(subset=[date_col])
                    .sort_values([ticker_col, date_col])
                    .groupby(ticker_col, as_index=False)
                    .tail(1)
                    .copy()
                )

        if latest_by_year:
            year_col = self._infer_col(src, ["year", "년도"])

            if year_col:
                src[year_col] = pd.to_numeric(src[year_col], errors="coerce")
                src = (
                    src.dropna(subset=[year_col])
                    .sort_values([ticker_col, year_col])
                    .groupby(ticker_col, as_index=False)
                    .tail(1)
                    .copy()
                )

        rename_map = {}

        for col in src.columns:
            if col == ticker_col:
                continue

            if col in base.columns:
                rename_map[col] = f"{conflict_prefix}_{col}"

        src = src.rename(columns=rename_map)

        before_cols = set(base.columns)

        merged = base.merge(
            src,
            left_on="ticker",
            right_on=ticker_col,
            how="left",
        )

        if ticker_col != "ticker":
            merged = merged.drop(columns=[ticker_col], errors="ignore")

        after_cols = set(merged.columns)

        ctx.logger.info(
            f"[CandidateScoringAgent] JOIN {source_name}: "
            f"+{len(after_cols - before_cols)} cols"
        )

        return merged

    def _merge_optional_features(
        self,
        base: pd.DataFrame,
        features: Optional[pd.DataFrame],
        source_name: str,
        ctx: RunContext,
    ) -> pd.DataFrame:
        if features is None or features.empty:
            ctx.logger.info(f"[CandidateScoringAgent] SKIP {source_name}: empty")
            return base

        if "ticker" not in features.columns:
            ctx.logger.info(f"[CandidateScoringAgent] SKIP {source_name}: no ticker")
            return base

        features = features.copy()
        features["ticker"] = features["ticker"].apply(self._zfill6)

        rename_map = {}

        for col in features.columns:
            if col == "ticker":
                continue

            if col in base.columns:
                rename_map[col] = f"{source_name}_{col}"

        features = features.rename(columns=rename_map)

        before_cols = set(base.columns)
        merged = base.merge(features, on="ticker", how="left")
        after_cols = set(merged.columns)

        ctx.logger.info(
            f"[CandidateScoringAgent] JOIN {source_name}: "
            f"+{len(after_cols - before_cols)} cols"
        )

        return merged

    def _merge_stock_master_names(
        self,
        base: pd.DataFrame,
        path: Optional[Path],
        ctx: RunContext,
    ) -> pd.DataFrame:
        """
        종목명 매핑 보강.

        기존 결과에서 company_name이 종목코드로만 남는 문제를 줄이기 위해
        IngestAgent가 생성한 top100_liquidity_*.csv가 있으면 ticker 기준으로 종목명을 붙인다.
        파일이 없으면 스킵하며, 기존 파이프라인을 깨지 않는다.
        """
        if path is None or not path.exists():
            ctx.logger.info("[CandidateScoringAgent] SKIP top100_liquidity: missing")
            return base

        src = self._read_csv(path)

        if src.empty:
            ctx.logger.info("[CandidateScoringAgent] SKIP top100_liquidity: empty")
            return base

        ticker_col = self._infer_col(
            src,
            [
                "ticker",
                "종목코드",
                "code",
                "symbol",
                "단축코드",
                "종목번호",
                "stock_code",
            ],
        )
        name_col = self._infer_col(
            src,
            [
                "company_name",
                "종목명",
                "stock_name",
                "name",
                "corp_name",
                "기업명",
                "한글 종목명",
                "한글명",
                "회사명",
                "Name",
            ],
        )

        if ticker_col is None or name_col is None:
            ctx.logger.info(
                "[CandidateScoringAgent] SKIP top100_liquidity: no ticker/name columns"
            )
            return base

        src = src[[ticker_col, name_col]].copy()
        src[ticker_col] = src[ticker_col].apply(self._zfill6)
        src[name_col] = src[name_col].astype(str).str.strip()
        src = src.drop_duplicates(subset=[ticker_col], keep="last")
        src = src.rename(
            columns={
                ticker_col: "ticker",
                name_col: "stock_master_company_name",
            }
        )

        merged = base.merge(src, on="ticker", how="left")
        ctx.logger.info(
            "[CandidateScoringAgent] JOIN top100_liquidity: "
            f"matched={merged['stock_master_company_name'].notna().sum()}"
        )

        return merged

    def _compute_missing_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "foreign_net_flow_ratio" not in df.columns:
            frgn_col = self._first_existing_col(
                df,
                [
                    "frgn_ntby_qty",
                    "foreign_frgn_ntby_qty",
                    "foreign_net_buy_qty",
                    "net_buy_qty",
                ],
            )

            volume_col = self._first_existing_col(
                df,
                [
                    "foreign_volume",
                    "volume",
                    "거래량",
                ],
            )

            if frgn_col and volume_col:
                numerator = pd.to_numeric(df[frgn_col], errors="coerce")
                denominator = pd.to_numeric(df[volume_col], errors="coerce").replace(0, np.nan)
                df["foreign_net_flow_ratio"] = numerator / denominator
            else:
                df["foreign_net_flow_ratio"] = np.nan

        if "overall_risk_score" not in df.columns:
            for col in [
                "risk_scores.overall_risk_score",
                "risk_overall_risk_score",
            ]:
                if col in df.columns:
                    df["overall_risk_score"] = df[col]
                    break

        if "risk_level" not in df.columns:
            for col in [
                "risk_scores.risk_level",
                "risk_risk_level",
            ]:
                if col in df.columns:
                    df["risk_level"] = df[col]
                    break

        return df

    def _ensure_company_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        company_name 보강.

        기존 구현은 company_name 컬럼이 이미 있으면 그 값이 종목코드여도 그대로 사용했다.
        그래서 결과 리포트에서 POSCO홀딩스 대신 005490처럼 종목코드만 보이는 문제가 생겼다.
        이 함수는 stock_master_company_name 등 실제 종목명 후보를 우선 사용하고,
        종목코드처럼 보이는 값은 이름 후보로 낮게 평가한다.
        """
        df = df.copy()

        if "ticker" not in df.columns:
            return df

        possible_cols = [
            "stock_master_company_name",
            "company_name",
            "종목명",
            "stock_name",
            "name",
            "company",
            "corp_name",
            "기업명",
            "한글 종목명",
            "한글명",
            "회사명",
            "market_analysis_json_company_name",
            "news_invest_json_company_name",
        ]

        company_like_cols = [
            col for col in df.columns
            if "company_name" in str(col) or "종목명" in str(col)
        ]

        for col in company_like_cols:
            if col not in possible_cols:
                possible_cols.append(col)

        best_col = None
        best_score = -1

        for col in possible_cols:
            if col not in df.columns:
                continue

            s = df[col].astype(str).str.strip()
            valid = s[(s != "") & (~s.str.lower().isin(["nan", "none", "null"]))]

            if valid.empty:
                continue

            non_code_count = valid.apply(lambda x: not self._looks_like_ticker(x)).sum()
            score = int(non_code_count)

            # stock_master에서 온 이름은 가장 신뢰도가 높으므로 우선권을 준다.
            if col == "stock_master_company_name":
                score += 10_000

            if score > best_score:
                best_score = score
                best_col = col

        ticker_as_name = df["ticker"].astype(str).apply(self._zfill6)
        current_name = df["company_name"] if "company_name" in df.columns else ticker_as_name
        current_name = current_name.astype(str).str.strip()

        if best_col is not None:
            candidate_name = df[best_col].astype(str).str.strip()
            candidate_name = candidate_name.mask(
                candidate_name.str.lower().isin(["", "nan", "none", "null"]),
                np.nan,
            )

            current_is_missing_or_code = (
                current_name.str.lower().isin(["", "nan", "none", "null"])
                | current_name.apply(self._looks_like_ticker)
                | (current_name == ticker_as_name)
            )

            df["company_name"] = current_name.mask(
                current_is_missing_or_code,
                candidate_name,
            )
            df["company_name"] = df["company_name"].replace("", np.nan).fillna(ticker_as_name)
        else:
            df["company_name"] = current_name.replace("", np.nan).fillna(ticker_as_name)

        return df

    # =========================================================
    # JSON feature extraction
    # =========================================================

    def _extract_market_json_features(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        if not obj:
            return None

        tickers = obj.get("tickers", [])

        if not isinstance(tickers, list):
            return None

        records = []

        for row in tickers:
            if not isinstance(row, dict):
                continue

            ticker = row.get("ticker") or row.get("종목코드") or row.get("code")

            if not ticker:
                continue

            compact = row.get("compact_signals", {}) or {}

            rec = {
                "ticker": self._zfill6(ticker),
                "market_ticker_type": row.get("ticker_type"),
                "market_phase_ticker": row.get("phase"),
                "market_tone": row.get("tone"),
                "market_rsi_state": row.get("rsi_state"),
                "market_return_state": row.get("return_state"),
                "market_return_20d": compact.get("return_20d"),
                "market_return_60d": compact.get("return_60d"),
                "market_return_252d": compact.get("return_252d"),
                "market_dist_to_ma20": compact.get("dist_to_ma20"),
                "market_dist_to_ma60": compact.get("dist_to_ma60"),
                "market_rsi_14": compact.get("rsi_14"),
                "market_ann_volatility_252d": compact.get("ann_volatility_252d"),
                "market_max_drawdown_252d": compact.get("max_drawdown_252d"),
                "market_foreign_ownership_level": compact.get("foreign_ownership_level"),
                "market_revenue_yoy": compact.get("revenue_yoy"),
                "market_operating_income_yoy": compact.get("operating_income_yoy"),
                "market_roe": compact.get("roe"),
                "market_debt_ratio": compact.get("debt_ratio"),
            }

            for name_col in [
                "company_name",
                "종목명",
                "stock_name",
                "name",
                "corp_name",
            ]:
                if row.get(name_col):
                    rec["company_name"] = row.get(name_col)
                    break

            records.append(rec)

        if not records:
            return None

        return pd.DataFrame(records)

    def _extract_news_json_features(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        if not obj:
            return None

        tickers = obj.get("tickers", [])

        if not isinstance(tickers, list):
            return None

        records = []

        for row in tickers:
            if not isinstance(row, dict):
                continue

            ticker = row.get("ticker") or row.get("종목코드") or row.get("code")

            if not ticker:
                continue

            summary = row.get("news_summary", {}) or {}

            rec = {
                "ticker": self._zfill6(ticker),
                "news_signal_score": row.get("news_signal_score"),
                "news_confidence_level": row.get("confidence_level"),
                "news_verdict": row.get("verdict"),
                "news_reasons": self._safe_json_cell(row.get("reasons")),
                "news_n_articles_7d_unique": summary.get("n_articles_7d_unique"),
                "news_n_articles_30d_unique": summary.get("n_articles_30d_unique"),
                "news_noise_ratio_0_1": summary.get("noise_ratio_0_1"),
                "news_high_impact_ratio_0_1": summary.get("high_impact_ratio_0_1"),
                "news_article_quality_score_0_1": summary.get("article_quality_score_0_1"),
                "top_articles": self._safe_json_cell(row.get("top_articles")),
            }

            for name_col in [
                "company_name",
                "종목명",
                "stock_name",
                "name",
                "corp_name",
            ]:
                if row.get(name_col):
                    rec["company_name"] = row.get(name_col)
                    break

            records.append(rec)

        if not records:
            return None

        return pd.DataFrame(records)

    def _extract_risk_json_features(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        if not obj:
            return None

        tickers = obj.get("tickers", [])

        if not isinstance(tickers, list):
            return None

        records = []

        for row in tickers:
            if not isinstance(row, dict):
                continue

            ticker = row.get("ticker") or row.get("종목코드") or row.get("code")

            if not ticker:
                continue

            scores = row.get("risk_scores", {}) or {}

            rec = {
                "ticker": self._zfill6(ticker),
                "price_overheat_risk": scores.get("price_overheat_risk"),
                "downside_risk": scores.get("downside_risk"),
                "financial_risk": scores.get("financial_risk"),
                "news_event_risk": scores.get("news_event_risk"),
                "uncertainty_risk": scores.get("uncertainty_risk"),
                "overall_risk_score": scores.get("overall_risk_score"),
                "risk_level": scores.get("risk_level"),
                "dominant_risk_factors": self._safe_json_cell(row.get("dominant_risk_factors")),
                "risk_evidence": self._safe_json_cell(row.get("evidence")),
            }

            records.append(rec)

        if not records:
            return None

        return pd.DataFrame(records)

    # =========================================================
    # Scoring
    # =========================================================

    def _score_candidates(
        self,
        df: pd.DataFrame,
        candidate_scoring_plan: Dict[str, Any],
        market_flow_result: Optional[Dict[str, Any]],
        ctx: RunContext,
    ) -> pd.DataFrame:
        df = df.copy()

        weights = self._get_component_weights(candidate_scoring_plan)

        df["market_flow_alignment_score"] = self._score_market_flow_alignment(
            df=df,
            market_flow_result=market_flow_result,
        )
        df["news_direct_relevance_score"] = self._score_news_direct_relevance(df)
        df["news_momentum_score"] = self._score_news_momentum(df)
        df["price_volume_momentum_score"] = self._score_price_volume_momentum(df)
        df["foreign_flow_score"] = self._score_foreign_flow(df)
        df["fundamental_score"] = self._score_fundamental(df)
        df["risk_penalty_score"] = self._score_risk_penalty(df)
        df["user_interest_boost"] = self._score_user_interest_boost(df)

        score = pd.Series(0.0, index=df.index)

        for component, weight in weights.items():
            if component not in df.columns:
                continue

            score = score + float(weight) * pd.to_numeric(
                df[component],
                errors="coerce",
            ).fillna(50.0)

        raw_score = score.clip(0, 100)
        capped_score, risk_cap_notes = self._apply_risk_score_caps(df, raw_score)

        df["score_before_risk_cap"] = raw_score.round(3)
        df["risk_score_cap_note"] = risk_cap_notes
        df["market_flow_candidate_score"] = capped_score.clip(0, 100).round(3)

        component_cols = [
            "market_flow_alignment_score",
            "news_direct_relevance_score",
            "news_momentum_score",
            "price_volume_momentum_score",
            "foreign_flow_score",
            "fundamental_score",
            "risk_penalty_score",
            "user_interest_boost",
        ]

        for col in component_cols:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(50.0)
                .clip(0, 100)
                .round(3)
            )

        ctx.logger.info(
            "[CandidateScoringAgent] Score calculated: "
            f"min={df['market_flow_candidate_score'].min():.2f}, "
            f"max={df['market_flow_candidate_score'].max():.2f}, "
            f"mean={df['market_flow_candidate_score'].mean():.2f}"
        )

        return df

    def _apply_risk_score_caps(
        self,
        df: pd.DataFrame,
        score: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        risk_level이 high/critical인 종목은 TOP10에 그대로 올라오지 않도록
        최종 후보 점수에 상한을 적용한다.

        - critical/위험: 최대 35점
        - high/높음: 최대 42점
        - overall_risk_score >= 75: 최대 40점

        이 처리는 risk_penalty_score 감점과 별도로 작동한다.
        즉, 고위험 종목이 단순히 모멘텀 하나만으로 상위 후보가 되는 것을 막는다.
        """
        capped = pd.to_numeric(score, errors="coerce").fillna(50.0).copy()
        notes = pd.Series("", index=df.index, dtype="object")

        risk_level = self._first_existing_series(
            df,
            ["risk_level", "risk_scores.risk_level"],
        ).astype(str).str.lower()

        overall_risk = pd.to_numeric(
            self._first_existing_series(
                df,
                ["overall_risk_score", "risk_scores.overall_risk_score"],
            ),
            errors="coerce",
        )

        critical_mask = risk_level.str.contains("critical|위험", na=False)
        high_mask = risk_level.str.contains("high|높음", na=False)
        numeric_high_mask = overall_risk >= 75

        capped.loc[critical_mask] = np.minimum(capped.loc[critical_mask], 35.0)
        notes.loc[critical_mask] = "risk_level이 critical/위험이라 최종 점수 상한 35점을 적용했습니다."

        high_only_mask = high_mask & ~critical_mask
        capped.loc[high_only_mask] = np.minimum(capped.loc[high_only_mask], 42.0)
        notes.loc[high_only_mask] = "risk_level이 high라 최종 점수 상한 42점을 적용했습니다."

        numeric_high_only_mask = numeric_high_mask & ~(critical_mask | high_only_mask)
        capped.loc[numeric_high_only_mask] = np.minimum(
            capped.loc[numeric_high_only_mask],
            40.0,
        )
        notes.loc[numeric_high_only_mask] = "overall_risk_score가 높아 최종 점수 상한 40점을 적용했습니다."

        return capped, notes

    def _get_component_weights(
        self,
        candidate_scoring_plan: Dict[str, Any],
    ) -> Dict[str, float]:
        default = {
            "market_flow_alignment_score": 0.25,
            "news_momentum_score": 0.20,
            "price_volume_momentum_score": 0.20,
            "foreign_flow_score": 0.15,
            "fundamental_score": 0.10,
            "risk_penalty_score": -0.10,
            "user_interest_boost": 0.10,
        }

        components = candidate_scoring_plan.get("score_components", [])

        if not isinstance(components, list):
            return default

        for item in components:
            if not isinstance(item, dict):
                continue

            name = item.get("component")
            weight = item.get("weight")

            if name in default and isinstance(weight, (int, float)):
                default[name] = float(weight)

        return default

    def _score_market_flow_alignment(
        self,
        df: pd.DataFrame,
        market_flow_result: Optional[Dict[str, Any]],
    ) -> pd.Series:
        tone = self._text_score(
            self._first_existing_series(
                df,
                ["market_tone", "tone"],
            ),
            mapping={
                "positive": 80,
                "neutral-positive": 70,
                "neutral": 55,
                "mixed": 50,
                "negative": 30,
                "bull": 80,
                "bear": 30,
                "상승": 80,
                "강세": 80,
                "혼조": 50,
                "하락": 30,
                "약세": 30,
            },
            default=50,
        )

        return_state = self._text_score(
            self._first_existing_series(
                df,
                ["market_return_state", "return_state", "market_regime_20"],
            ),
            mapping={
                "up": 75,
                "side": 50,
                "down": 30,
                "strong": 80,
                "weak": 30,
                "상승": 75,
                "강세": 75,
                "횡보": 50,
                "하락": 30,
                "약세": 30,
            },
            default=50,
        )

        rsi_state = self._text_score(
            self._first_existing_series(
                df,
                ["market_rsi_state", "rsi_state"],
            ),
            mapping={
                "neutral": 60,
                "normal": 60,
                "oversold": 65,
                "overbought": 40,
                "과매도": 65,
                "중립": 60,
                "과열": 40,
                "침체": 65,
                "경미한 과열": 45,
                "강한 과열": 30,
            },
            default=55,
        )

        daily_return_score = self._percentile_score(
            self._first_existing_series(
                df,
                ["daily_return", "market_return_20d"],
            ),
            higher_better=True,
            neutral=50,
        )

        price_to_ma20_score = self._percentile_score(
            self._first_existing_series(
                df,
                ["price_to_ma20", "market_dist_to_ma20"],
            ),
            higher_better=True,
            neutral=50,
        )

        base_score = self._weighted_series(
            [
                (tone, 0.20),
                (return_state, 0.15),
                (rsi_state, 0.10),
                (daily_return_score, 0.25),
                (price_to_ma20_score, 0.20),
            ],
            neutral=50,
        )

        market_flow_context_score = self._score_market_flow_context_from_result(
            df=df,
            market_flow_result=market_flow_result,
        )

        return (
            0.75 * base_score
            + 0.25 * market_flow_context_score
        ).clip(0, 100)

    def _score_market_flow_context_from_result(
        self,
        df: pd.DataFrame,
        market_flow_result: Optional[Dict[str, Any]],
    ) -> pd.Series:
        """
        MarketFlowAgent 결과의 deterministic_snapshot을 이용해
        종목별 시장 흐름 적합도 보정 점수를 만든다.

        50점을 기본값으로 두고,
        top_gainers / top_event_news / high_signal_tickers / top_foreign_buy는 가점,
        top_negative_news / top_foreign_sell / high_risk_tickers는 감점한다.
        """
        score = pd.Series(50.0, index=df.index)

        if not market_flow_result:
            return score

        snapshot = market_flow_result.get("deterministic_snapshot", {}) or {}

        price_snapshot = snapshot.get("price_market_snapshot", {}) or {}
        news_feature_overview = snapshot.get("news_feature_overview", {}) or {}
        foreign_overview = snapshot.get("foreign_flow_overview", {}) or {}
        news_overview = snapshot.get("news_overview", {}) or {}
        risk_overview = snapshot.get("risk_overview", {}) or {}

        top_gainers = self._extract_ticker_set(
            price_snapshot.get("top_gainers", [])
        )
        top_event_news = self._extract_ticker_set(
            news_feature_overview.get("top_event_news", [])
        )
        top_negative_news = self._extract_ticker_set(
            news_feature_overview.get("top_negative_news", [])
        )
        top_foreign_buy = self._extract_ticker_set(
            foreign_overview.get("top_foreign_buy", [])
        )
        top_foreign_sell = self._extract_ticker_set(
            foreign_overview.get("top_foreign_sell", [])
        )
        high_signal_tickers = self._extract_ticker_set(
            news_overview.get("high_signal_tickers", [])
        )
        high_risk_tickers = self._extract_ticker_set(
            risk_overview.get("high_risk_tickers", [])
        )

        ticker_series = df["ticker"].apply(self._zfill6)

        score = score + ticker_series.isin(top_gainers).astype(float) * 10.0
        score = score + ticker_series.isin(top_event_news).astype(float) * 8.0
        score = score + ticker_series.isin(high_signal_tickers).astype(float) * 10.0
        score = score + ticker_series.isin(top_foreign_buy).astype(float) * 8.0

        score = score - ticker_series.isin(top_negative_news).astype(float) * 8.0
        score = score - ticker_series.isin(top_foreign_sell).astype(float) * 6.0
        score = score - ticker_series.isin(high_risk_tickers).astype(float) * 12.0

        return score.clip(0, 100)

    def _score_news_momentum(self, df: pd.DataFrame) -> pd.Series:
        news_signal = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["news_signal_score"],
            ),
            neutral=50,
        )

        event_ratio = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["event_news_ratio"],
            ),
            neutral=50,
        )

        high_impact = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["news_high_impact_ratio_0_1"],
            ),
            neutral=50,
        )

        quality = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["news_article_quality_score_0_1"],
            ),
            neutral=50,
        )

        confidence = self._text_score(
            self._first_existing_series(
                df,
                ["news_confidence_level", "confidence_level"],
            ),
            mapping={
                "high": 80,
                "medium": 55,
                "low": 35,
            },
            default=50,
        )

        negative_ratio = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["negative_news_ratio"],
            ),
            neutral=0,
        )

        latest_negative = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["latest_negative_flag"],
            ),
            neutral=0,
        )

        base = self._weighted_series(
            [
                (news_signal, 0.35),
                (event_ratio, 0.15),
                (high_impact, 0.15),
                (quality, 0.15),
                (confidence, 0.20),
            ],
            neutral=50,
        )

        # 기사 직접 관련성 필터.
        # relevance=50이면 기존 점수를 유지하고,
        # 직접 관련성이 낮으면 뉴스 모멘텀을 낮추며,
        # 직접 관련성이 높으면 소폭 가점한다.
        relevance = pd.to_numeric(
            self._first_existing_series(
                df,
                ["news_direct_relevance_score"],
            ),
            errors="coerce",
        ).fillna(50.0).clip(0, 100)

        relevance_factor = pd.Series(1.0, index=df.index)
        low_mask = relevance < 50
        high_mask = relevance >= 50

        relevance_factor.loc[low_mask] = 0.60 + 0.40 * (relevance.loc[low_mask] / 50.0)
        relevance_factor.loc[high_mask] = 1.00 + 0.10 * ((relevance.loc[high_mask] - 50.0) / 50.0)

        base = (base * relevance_factor).clip(0, 100)

        penalty = 0.15 * negative_ratio + 0.10 * latest_negative

        return (base - penalty).clip(0, 100)

    def _score_news_direct_relevance(self, df: pd.DataFrame) -> pd.Series:
        """
        뉴스가 실제로 해당 종목과 직접 관련 있는지 점수화한다.

        기존 뉴스 점수는 기사 제목/본문에 종목명이 스치듯 포함되어도
        높은 article_score가 들어갈 수 있었다. 이 함수는 top_articles의
        title/description에 회사명이 직접 등장하는지, 투자 관련 키워드가
        함께 등장하는지를 확인해 뉴스 모멘텀 점수를 보정한다.
        """
        scores = []

        for _, row in df.iterrows():
            articles = self._maybe_parse_json_cell(row.get("top_articles"))

            if not isinstance(articles, list) or not articles:
                scores.append(50.0)
                continue

            aliases = self._company_aliases_from_row(row)

            if not aliases:
                scores.append(50.0)
                continue

            article_scores = []

            for article in articles[:5]:
                if not isinstance(article, dict):
                    continue

                title = str(article.get("title", ""))
                description = str(article.get("description", ""))
                text = f"{title} {description}"

                title_direct = any(alias in title for alias in aliases)
                desc_direct = any(alias in description for alias in aliases)
                direct = title_direct or desc_direct
                keyword_count = self._investment_keyword_count(text)

                if direct:
                    score = 35.0
                    if title_direct:
                        score += 35.0
                    if desc_direct:
                        score += 15.0
                    score += min(keyword_count, 3) * 5.0
                else:
                    # 종목명이 직접 등장하지 않는 기사는 노이즈 가능성이 높으므로 강하게 제한한다.
                    score = 20.0 + min(keyword_count, 2) * 5.0
                    score = min(score, 35.0)

                article_scores.append(score)

            if not article_scores:
                scores.append(50.0)
            else:
                scores.append(float(np.mean(article_scores)))

        return pd.Series(scores, index=df.index).clip(0, 100)

    def _score_price_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        daily_return = self._percentile_score(
            self._first_existing_series(
                df,
                ["daily_return"],
            ),
            higher_better=True,
            neutral=50,
        )

        price_to_ma20 = self._percentile_score(
            self._first_existing_series(
                df,
                ["price_to_ma20"],
            ),
            higher_better=True,
            neutral=50,
        )

        ma20_slope = self._percentile_score(
            self._first_existing_series(
                df,
                ["ma20_slope"],
            ),
            higher_better=True,
            neutral=50,
        )

        volume_z = self._percentile_score(
            self._first_existing_series(
                df,
                ["volume_zscore_20d"],
            ),
            higher_better=True,
            neutral=50,
        )

        liquidity = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["liquidity_percentile_20d"],
            ),
            neutral=50,
        )

        base = self._weighted_series(
            [
                (daily_return, 0.25),
                (price_to_ma20, 0.20),
                (ma20_slope, 0.15),
                (volume_z, 0.25),
                (liquidity, 0.15),
            ],
            neutral=50,
        )

        rsi = pd.to_numeric(
            self._first_existing_series(
                df,
                ["rsi_14", "market_rsi_14"],
            ),
            errors="coerce",
        )

        overheat_penalty = pd.Series(0.0, index=df.index)
        overheat_penalty = overheat_penalty.mask(rsi >= 80, 12.0)
        overheat_penalty = overheat_penalty.mask((rsi >= 70) & (rsi < 80), 6.0)

        return (base - overheat_penalty).clip(0, 100)

    def _score_foreign_flow(self, df: pd.DataFrame) -> pd.Series:
        flow_ratio = self._first_existing_series(
            df,
            ["foreign_net_flow_ratio"],
        )

        flow_score = self._percentile_score(
            flow_ratio,
            higher_better=True,
            neutral=50,
        )

        ownership_col = self._first_existing_series(
            df,
            [
                "foreign_ownership_level",
                "market_foreign_ownership_level",
                "hts_frgn_ehrt",
                "foreign_hts_frgn_ehrt",
            ],
        )

        ownership_numeric = pd.to_numeric(ownership_col, errors="coerce")

        if ownership_numeric.notna().sum() > 0:
            ownership_score = self._percentile_score(
                ownership_numeric,
                higher_better=True,
                neutral=50,
            )
        else:
            ownership_score = self._text_score(
                ownership_col,
                mapping={
                    "high": 75,
                    "medium": 55,
                    "low": 35,
                    "높음": 75,
                    "보통": 55,
                    "낮음": 35,
                },
                default=50,
            )

        return self._weighted_series(
            [
                (flow_score, 0.70),
                (ownership_score, 0.30),
            ],
            neutral=50,
        )

    def _score_fundamental(self, df: pd.DataFrame) -> pd.Series:
        revenue = self._percentile_score(
            self._first_existing_series(
                df,
                ["revenue_yoy", "market_revenue_yoy"],
            ),
            higher_better=True,
            neutral=50,
        )

        operating_income = self._percentile_score(
            self._first_existing_series(
                df,
                ["operating_income_yoy", "market_operating_income_yoy"],
            ),
            higher_better=True,
            neutral=50,
        )

        roe = self._percentile_score(
            self._first_existing_series(
                df,
                ["roe", "market_roe"],
            ),
            higher_better=True,
            neutral=50,
        )

        debt = self._percentile_score(
            self._first_existing_series(
                df,
                ["debt_ratio", "market_debt_ratio"],
            ),
            higher_better=False,
            neutral=50,
        )

        current_ratio = self._percentile_score(
            self._first_existing_series(
                df,
                ["current_ratio"],
            ),
            higher_better=True,
            neutral=50,
        )

        return self._weighted_series(
            [
                (revenue, 0.25),
                (operating_income, 0.25),
                (roe, 0.25),
                (debt, 0.15),
                (current_ratio, 0.10),
            ],
            neutral=50,
        )

    def _score_risk_penalty(self, df: pd.DataFrame) -> pd.Series:
        risk_score = self._score_scale_0_100(
            self._first_existing_series(
                df,
                ["overall_risk_score", "risk_scores.overall_risk_score"],
            ),
            neutral=np.nan,
        )

        risk_level_score = self._text_score(
            self._first_existing_series(
                df,
                ["risk_level", "risk_scores.risk_level"],
            ),
            mapping={
                "low": 20,
                "medium": 45,
                "high": 75,
                "critical": 95,
                "낮음": 20,
                "보통": 45,
                "높음": 75,
                "위험": 95,
            },
            default=np.nan,
        )

        if risk_score.notna().sum() == 0 and risk_level_score.notna().sum() == 0:
            return pd.Series(50.0, index=df.index)

        return self._weighted_series(
            [
                (risk_score, 0.75),
                (risk_level_score, 0.25),
            ],
            neutral=50,
        )

    def _score_user_interest_boost(self, df: pd.DataFrame) -> pd.Series:
        for col in [
            "user_interest_score",
            "interest_score",
            "watchlist_flag",
            "is_interested",
            "favorite_flag",
        ]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")

                if s.notna().sum() > 0:
                    return self._score_scale_0_100(s, neutral=0).fillna(0).clip(0, 100)

        return pd.Series(0.0, index=df.index)

    # =========================================================
    # Explanations
    # =========================================================

    def _add_explanations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        positive_evidence = []
        negative_evidence = []
        risk_notes = []
        reasons = []
        component_scores = []

        for _, row in df.iterrows():
            comp = {
                "market_flow_alignment_score": self._safe_float(row.get("market_flow_alignment_score")),
                "news_direct_relevance_score": self._safe_float(row.get("news_direct_relevance_score")),
                "news_momentum_score": self._safe_float(row.get("news_momentum_score")),
                "price_volume_momentum_score": self._safe_float(row.get("price_volume_momentum_score")),
                "foreign_flow_score": self._safe_float(row.get("foreign_flow_score")),
                "fundamental_score": self._safe_float(row.get("fundamental_score")),
                "risk_penalty_score": self._safe_float(row.get("risk_penalty_score")),
                "user_interest_boost": self._safe_float(row.get("user_interest_boost")),
            }

            pos = []
            neg = []
            risk = []

            if (comp["market_flow_alignment_score"] or 0) >= 70:
                pos.append("시장 흐름과 종목 신호의 방향성이 비교적 잘 맞습니다.")

            if (comp["news_momentum_score"] or 0) >= 70:
                pos.append("뉴스 모멘텀 점수가 높아 최근 관심 신호가 있습니다.")

            if (comp.get("news_direct_relevance_score") or 0) >= 70:
                pos.append("상위 뉴스가 해당 종목과 직접 관련된 기사로 확인됩니다.")

            if (comp.get("news_direct_relevance_score") or 50) <= 35:
                neg.append("뉴스 기사와 종목의 직접 관련성이 낮아 뉴스 점수를 보수적으로 반영했습니다.")

            if (comp["price_volume_momentum_score"] or 0) >= 70:
                pos.append("가격·거래량 흐름이 상대적으로 양호합니다.")

            if (comp["foreign_flow_score"] or 0) >= 70:
                pos.append("외국인 수급 신호가 상대적으로 긍정적입니다.")

            if (comp["fundamental_score"] or 0) >= 70:
                pos.append("실적·재무 지표가 상대적으로 양호합니다.")

            daily_return = self._safe_float(row.get("daily_return"))

            if daily_return is not None and daily_return > 0:
                pos.append(f"최근 일간 수익률이 양호합니다. daily_return={daily_return:.4f}")

            foreign_ratio = self._safe_float(row.get("foreign_net_flow_ratio"))

            if foreign_ratio is not None and foreign_ratio > 0:
                pos.append(f"외국인 순매수 비율이 양수입니다. foreign_net_flow_ratio={foreign_ratio:.4f}")

            if (comp["risk_penalty_score"] or 0) >= 75:
                neg.append("리스크 점수가 높아 후보 선정 시 감점되었습니다.")

            if (comp["news_momentum_score"] or 0) <= 35:
                neg.append("뉴스 모멘텀 신호가 약하거나 부정 신호가 포함될 수 있습니다.")

            if (comp["price_volume_momentum_score"] or 0) <= 35:
                neg.append("가격·거래량 모멘텀이 약한 편입니다.")

            risk_level = str(row.get("risk_level", "")).lower()
            overall_risk = self._safe_float(row.get("overall_risk_score"))
            risk_cap_note = row.get("risk_score_cap_note")

            if "high" in risk_level or "critical" in risk_level or "높음" in risk_level or "위험" in risk_level:
                neg.append("risk_level이 높아 최종 점수 상한 또는 강한 주의 문구가 적용되었습니다.")

            if risk_cap_note is not None and str(risk_cap_note).strip() not in ["", "nan", "None"]:
                risk.append(str(risk_cap_note))

            if risk_level and risk_level not in ["nan", "none", ""]:
                risk.append(f"risk_level={row.get('risk_level')}")

            if overall_risk is not None:
                risk.append(f"overall_risk_score={overall_risk:.2f}")

            dominant = row.get("dominant_risk_factors")

            if dominant is not None and str(dominant).strip() not in ["", "nan", "None"]:
                risk.append(f"주요 리스크 요인: {dominant}")

            if not pos:
                pos.append("일부 데이터 기준에서 평균 수준의 후보 신호가 확인되었습니다.")

            if not neg:
                neg.append("큰 감점 요인은 제한적이지만, 세부 데이터 확인은 필요합니다.")

            if not risk:
                risk.append("확인된 리스크 정보가 제한적입니다.")

            reason = self._make_ranking_reason(row, comp)

            positive_evidence.append(pos[:5])
            negative_evidence.append(neg[:5])
            risk_notes.append(risk[:5])
            reasons.append(reason)
            component_scores.append(comp)

        df["component_scores"] = component_scores
        df["positive_evidence"] = positive_evidence
        df["negative_evidence"] = negative_evidence
        df["risk_notes"] = risk_notes
        df["ranking_reason_short"] = reasons

        return df

    def _make_ranking_reason(
        self,
        row: pd.Series,
        comp: Dict[str, Optional[float]],
    ) -> str:
        name_map = {
            "market_flow_alignment_score": "시장 흐름 연관도",
            "news_momentum_score": "뉴스 모멘텀",
            "price_volume_momentum_score": "가격·거래량 모멘텀",
            "foreign_flow_score": "외국인 수급",
            "fundamental_score": "실적·재무",
            "user_interest_boost": "사용자 관심도",
        }

        positive_components = {
            k: v
            for k, v in comp.items()
            if k in name_map and v is not None
        }

        top_components = sorted(
            positive_components.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:2]

        if top_components:
            top_text = ", ".join(
                [f"{name_map[k]}({v:.1f})" for k, v in top_components]
            )
        else:
            top_text = "복합 지표"

        risk_score = comp.get("risk_penalty_score")
        risk_text = ""
        risk_level_text = str(row.get("risk_level", "")).lower()
        risk_cap_note = row.get("risk_score_cap_note")

        if risk_cap_note is not None and str(risk_cap_note).strip() not in ["", "nan", "None"]:
            risk_text = " 다만 고위험 종목으로 분류되어 점수 상한이 적용되었습니다."
        elif "high" in risk_level_text or "critical" in risk_level_text or "높음" in risk_level_text or "위험" in risk_level_text:
            risk_text = " 다만 risk_level이 높아 후보 검토 시 주의가 필요합니다."
        elif risk_score is not None:
            if risk_score >= 75:
                risk_text = " 다만 리스크 점수가 높아 주의가 필요합니다."
            elif risk_score <= 35:
                risk_text = " 리스크 부담은 상대적으로 낮은 편입니다."
            else:
                risk_text = " 리스크는 중간 수준으로 점검이 필요합니다."

        return (
            f"{top_text} 신호가 상대적으로 높아 시장 흐름 기반 후보로 선정되었습니다."
            f"{risk_text}"
        )

    # =========================================================
    # Result JSON
    # =========================================================

    def _build_result_json(
        self,
        scored: pd.DataFrame,
        top10: pd.DataFrame,
        report_target_spec: Dict[str, Any],
        candidate_scoring_plan: Dict[str, Any],
        market_flow_result: Optional[Dict[str, Any]],
        paths: Dict[str, Optional[Path]],
    ) -> Dict[str, Any]:
        as_of_date = None

        if "as_of_date" in scored.columns:
            dates = pd.to_datetime(scored["as_of_date"], errors="coerce").dropna()

            if not dates.empty:
                as_of_date = str(dates.max().date())

        component_cols = [
            "market_flow_alignment_score",
            "news_direct_relevance_score",
            "news_momentum_score",
            "price_volume_momentum_score",
            "foreign_flow_score",
            "fundamental_score",
            "risk_penalty_score",
            "user_interest_boost",
        ]

        score_summary = {
            "n_candidates": int(len(scored)),
            "top_k": int(len(top10)),
            "score_min": self._safe_float(scored["market_flow_candidate_score"].min()),
            "score_max": self._safe_float(scored["market_flow_candidate_score"].max()),
            "score_mean": self._safe_float(scored["market_flow_candidate_score"].mean()),
            "component_means": {
                col: self._safe_float(scored[col].mean())
                for col in component_cols
                if col in scored.columns
            },
        }

        top_candidates = []

        for _, row in top10.iterrows():
            item = {
                "rank": int(row.get("rank")),
                "ticker": str(row.get("ticker")),
                "company_name": str(row.get("company_name")),
                "as_of_date": self._date_to_str(row.get("as_of_date")),
                "market_flow_candidate_score": self._safe_float(
                    row.get("market_flow_candidate_score")
                ),
                "score_before_risk_cap": self._safe_float(row.get("score_before_risk_cap")),
                "risk_score_cap_note": row.get("risk_score_cap_note"),
                "component_scores": row.get("component_scores"),
                "positive_evidence": row.get("positive_evidence"),
                "negative_evidence": row.get("negative_evidence"),
                "risk_notes": row.get("risk_notes"),
                "ranking_reason_short": row.get("ranking_reason_short"),
                "actual_data": {
                    "latest_close": self._safe_float(row.get("latest_close")),
                    "daily_return": self._safe_float(row.get("daily_return")),
                    "volume": self._safe_float(row.get("volume")),
                    "foreign_net_flow_ratio": self._safe_float(row.get("foreign_net_flow_ratio")),
                    "roe": self._safe_float(
                        row.get("roe") if "roe" in row else row.get("market_roe")
                    ),
                    "revenue_yoy": self._safe_float(
                        row.get("revenue_yoy") if "revenue_yoy" in row else row.get("market_revenue_yoy")
                    ),
                    "operating_income_yoy": self._safe_float(
                        row.get("operating_income_yoy")
                        if "operating_income_yoy" in row
                        else row.get("market_operating_income_yoy")
                    ),
                    "overall_risk_score": self._safe_float(row.get("overall_risk_score")),
                    "risk_level": None if pd.isna(row.get("risk_level")) else row.get("risk_level"),
                    "news_direct_relevance_score": self._safe_float(row.get("news_direct_relevance_score")),
                },
                "top_articles": self._filter_top_articles_for_output(row),
            }

            top_candidates.append(self._json_safe_obj(item))

        market_flow_summary = self._extract_market_flow_summary(market_flow_result)

        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "purpose": "market_flow_based_candidate_scoring",
            },
            "input_files": {
                k: str(v) if v is not None else None
                for k, v in paths.items()
            },
            "report_context": {
                "report_type": report_target_spec.get("report_type"),
                "report_title": report_target_spec.get("report_title"),
                "main_objective": report_target_spec.get("main_objective"),
                "ranking_target": report_target_spec.get("ranking_target"),
                "prediction_horizon": report_target_spec.get("prediction_horizon"),
                "ranking_scope": report_target_spec.get("ranking_scope"),
                "candidate_definition": report_target_spec.get("candidate_definition"),
            },
            "as_of_date": as_of_date,
            "scoring_target": candidate_scoring_plan.get("scoring_target", {}),
            "score_formula": candidate_scoring_plan.get("formula"),
            "market_flow_summary": market_flow_summary,
            "score_summary": score_summary,
            "top10_candidates": top_candidates,
            "limitations": [
                "이 점수는 미래 수익률 예측값이 아니라 현재 시장 흐름과 데이터 기반 후보 점수입니다.",
                "top100_liquidity_*.csv에서 종목명을 찾지 못할 경우 company_name은 ticker로 대체됩니다.",
                "외국인 순매수 비율 컬럼이 없을 경우 frgn_ntby_qty / volume으로 계산합니다.",
                "MarketFlowAgent의 deterministic_snapshot은 후보 점수 보정에 사용됩니다.",
                "market_flow_result.json이 없으면 market_analysis/news/risk 기반의 fallback 점수로 계산됩니다.",
                "risk_level이 high/critical인 종목은 최종 점수 상한을 적용합니다.",
                "뉴스 모멘텀은 기사 제목/본문의 종목 직접 관련성으로 보정합니다.",
            ],
        }

    def _extract_market_flow_summary(
        self,
        market_flow_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not market_flow_result:
            return {
                "exists": False,
                "market_one_line_summary": "",
                "market_brief": "",
                "market_keywords": [],
                "candidate_scoring_hints": {},
                "core_market_drivers": [],
            }

        flow = market_flow_result.get("market_flow", market_flow_result)

        if not isinstance(flow, dict):
            return {
                "exists": False,
                "market_one_line_summary": "",
                "market_brief": "",
                "market_keywords": [],
                "candidate_scoring_hints": {},
                "core_market_drivers": [],
            }

        return {
            "exists": True,
            "as_of_date": flow.get("as_of_date"),
            "market_one_line_summary": flow.get("market_one_line_summary"),
            "market_brief": flow.get("market_brief"),
            "market_keywords": flow.get("market_keywords", []),
            "core_market_drivers": flow.get("core_market_drivers", []),
            "candidate_scoring_hints": flow.get("candidate_scoring_hints", {}),
            "report_summary_box": flow.get("report_summary_box", {}),
            "limitations": flow.get("limitations", []),
            "generation_provider": flow.get("generation_provider"),
            "generation_model": flow.get("generation_model"),
        }

    # =========================================================
    # Score helper functions
    # =========================================================

    def _weighted_series(
        self,
        items: List[Tuple[pd.Series, float]],
        neutral: float = 50.0,
    ) -> pd.Series:
        if not items:
            return pd.Series(dtype=float)

        index = items[0][0].index
        total = pd.Series(0.0, index=index)
        weight_sum = pd.Series(0.0, index=index)

        for series, weight in items:
            if series is None:
                continue

            s = pd.to_numeric(series, errors="coerce")
            valid = s.notna()

            total = total + s.fillna(0.0) * float(weight)
            weight_sum = weight_sum + valid.astype(float) * float(weight)

        out = total / weight_sum.replace(0, np.nan)
        out = out.fillna(neutral)

        return out.clip(0, 100)

    def _percentile_score(
        self,
        series: pd.Series,
        higher_better: bool = True,
        neutral: float = 50.0,
    ) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)

        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(neutral, index=s.index, dtype=float)

        valid = s.dropna()

        if valid.empty:
            return out

        if valid.nunique() <= 1:
            out.loc[valid.index] = neutral
            return out

        out.loc[valid.index] = valid.rank(
            pct=True,
            ascending=higher_better,
        ) * 100.0

        return out.clip(0, 100)

    def _score_scale_0_100(
        self,
        series: pd.Series,
        neutral: float = 50.0,
    ) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)

        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(neutral, index=s.index, dtype=float)

        valid = s.dropna()

        if valid.empty:
            return out

        min_v = valid.min()
        max_v = valid.max()

        if min_v >= 0 and max_v <= 1.5:
            out.loc[valid.index] = valid * 100.0
        elif min_v >= 0 and max_v <= 100:
            out.loc[valid.index] = valid
        else:
            out.loc[valid.index] = self._percentile_score(
                valid,
                higher_better=True,
                neutral=neutral,
            )

        return out.clip(0, 100)

    def _text_score(
        self,
        series: pd.Series,
        mapping: Dict[str, float],
        default: float = 50.0,
    ) -> pd.Series:
        if series is None:
            return pd.Series(dtype=float)

        out = pd.Series(default, index=series.index, dtype=float)

        for idx, value in series.items():
            if value is None:
                continue

            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass

            text = str(value).lower()
            matched = False

            for key, score in mapping.items():
                if str(key).lower() in text:
                    out.loc[idx] = float(score)
                    matched = True
                    break

            if not matched:
                out.loc[idx] = default

        return out.clip(0, 100)

    def _first_existing_series(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.Series:
        for col in cols:
            if col in df.columns:
                return df[col]

        return pd.Series(np.nan, index=df.index)

    def _first_existing_col(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> Optional[str]:
        for col in cols:
            if col in df.columns:
                return col

        return None

    def _extract_ticker_set(self, records: Any) -> set[str]:
        if not isinstance(records, list):
            return set()

        tickers = set()

        for row in records:
            if not isinstance(row, dict):
                continue

            ticker = (
                row.get("ticker")
                or row.get("종목코드")
                or row.get("code")
                or row.get("symbol")
            )

            if ticker:
                tickers.add(self._zfill6(ticker))

        return tickers

    def _company_aliases_from_row(self, row: pd.Series) -> List[str]:
        aliases = []

        for col in [
            "company_name",
            "stock_master_company_name",
            "종목명",
            "stock_name",
            "name",
            "corp_name",
            "기업명",
            "market_analysis_json_company_name",
            "news_invest_json_company_name",
        ]:
            if col not in row:
                continue

            value = row.get(col)

            if value is None:
                continue

            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass

            text = str(value).strip()

            if not text or text.lower() in ["nan", "none", "null"]:
                continue

            if self._looks_like_ticker(text):
                continue

            aliases.append(text)

            # 보통 기사에는 '주식회사', '(주)' 같은 표기가 빠져 있으므로 간단한 별칭도 추가한다.
            simplified = (
                text.replace("주식회사", "")
                .replace("(주)", "")
                .replace("㈜", "")
                .strip()
            )
            if simplified and simplified != text:
                aliases.append(simplified)

        # 중복 제거. 긴 이름을 먼저 검사해야 부분 문자열 오탐이 줄어든다.
        unique_aliases = sorted(set(aliases), key=len, reverse=True)

        return unique_aliases

    def _investment_keyword_count(self, text: str) -> int:
        keywords = [
            "실적",
            "영업이익",
            "매출",
            "수주",
            "공급",
            "계약",
            "증설",
            "투자",
            "인수",
            "합병",
            "M&A",
            "배당",
            "자사주",
            "목표가",
            "리포트",
            "외국인",
            "기관",
            "순매수",
            "순매도",
            "공시",
            "특징주",
            "상승",
            "하락",
            "급등",
            "급락",
            "정책",
            "규제",
            "테마",
            "반도체",
            "AI",
            "전력",
            "방산",
            "조선",
            "배터리",
        ]

        return sum(1 for keyword in keywords if keyword.lower() in text.lower())

    def _filter_top_articles_for_output(self, row: pd.Series) -> Any:
        articles = self._maybe_parse_json_cell(row.get("top_articles"))

        if not isinstance(articles, list) or not articles:
            return articles

        aliases = self._company_aliases_from_row(row)

        if not aliases:
            return articles

        filtered = []

        for article in articles:
            if not isinstance(article, dict):
                continue

            title = str(article.get("title", ""))
            description = str(article.get("description", ""))
            text = f"{title} {description}"

            direct = any(alias in text for alias in aliases)
            has_investment_keyword = self._investment_keyword_count(text) > 0

            if direct and has_investment_keyword:
                filtered.append(article)

        # 너무 엄격하게 걸러서 비어버리면 원문을 유지하되 점수에서만 감점한다.
        return filtered if filtered else articles

    # =========================================================
    # File helpers
    # =========================================================

    def _find_single_file(
        self,
        run_dir: Path,
        pattern: str,
        ctx: RunContext,
    ) -> Optional[Path]:
        matches = sorted(run_dir.rglob(pattern))

        if not matches:
            ctx.logger.info(f"[CandidateScoringAgent] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            ctx.logger.warning(
                f"[CandidateScoringAgent] Multiple files matched: {pattern}. "
                f"Using first: {matches[0]}"
            )

        return matches[0]

    def _require_files(self, paths: List[Optional[Path]]) -> None:
        missing = [str(p) for p in paths if p is None or not p.exists()]

        if missing:
            raise FileNotFoundError(f"Missing required files: {missing}")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)

    def _safe_read_json(
        self,
        path: Optional[Path],
        ctx: RunContext,
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            return None

        if not path.exists():
            ctx.logger.info(f"[CandidateScoringAgent] JSON missing: {path}")
            return None

        try:
            return self._read_json(path)
        except Exception as e:
            ctx.logger.warning(
                f"[CandidateScoringAgent] Failed to read JSON: {path} ({e})"
            )
            return None

    def _read_csv(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    def _save_csv(self, df: pd.DataFrame, path: Path) -> None:
        out = df.copy()

        for col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].apply(self._safe_json_cell_for_csv)

        out.to_csv(path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # =========================================================
    # Utility helpers
    # =========================================================

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _zfill6(x: Any) -> str:
        s = str(x).replace(".0", "")
        return s.zfill(6)

    @staticmethod
    def _looks_like_ticker(x: Any) -> bool:
        text = str(x).strip().replace(".0", "")
        return text.isdigit() and len(text) <= 6

    @staticmethod
    def _infer_col(
        df: pd.DataFrame,
        candidates: List[str],
    ) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col

        return None

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None

            if pd.isna(x):
                return None

            value = float(x)

            if math.isnan(value) or math.isinf(value):
                return None

            return value
        except Exception:
            return None

    @staticmethod
    def _date_to_str(x: Any) -> Optional[str]:
        try:
            if x is None or pd.isna(x):
                return None

            return str(pd.to_datetime(x).date())
        except Exception:
            return str(x)

    @staticmethod
    def _safe_json_cell(value: Any) -> str:
        if value is None:
            return ""

        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)

        return str(value)

    @staticmethod
    def _safe_json_cell_for_csv(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)

        return value

    @staticmethod
    def _maybe_parse_json_cell(value: Any) -> Any:
        if value is None:
            return []

        try:
            if pd.isna(value):
                return []
        except Exception:
            pass

        if isinstance(value, (list, dict)):
            return value

        text = str(value).strip()

        if not text:
            return []

        try:
            return json.loads(text)
        except Exception:
            return text

    def _json_safe_obj(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._json_safe_obj(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._json_safe_obj(v) for v in obj]

        if isinstance(obj, pd.Timestamp):
            return str(obj.date())

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None

            return float(obj)

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None

            return obj

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        return obj

    def _make_stage_result(
        self,
        status: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        sig = inspect.signature(StageResult)

        candidate_kwargs = {
            "stage": self.stage,
            "status": status,
            "message": message,
            "metrics": metrics or {},
            "outputs": outputs or {},
            "artifacts": outputs or {},
        }

        kwargs = {
            k: v
            for k, v in candidate_kwargs.items()
            if k in sig.parameters
        }

        return StageResult(**kwargs)