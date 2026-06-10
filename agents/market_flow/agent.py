from __future__ import annotations

import inspect
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


@dataclass
class MarketFlowArtifacts:
    run_dir: Path
    output_dir: Path
    result_path: Path
    summary_md_path: Path
    manifest_path: Path
    output_files: List[str]


class MarketFlowAgent(BaseAgent):
    """
    Market Flow Agent

    목적:
    - 전처리된 시세/뉴스/외국인 수급/재무 데이터와
      market/news/risk agent 결과를 종합한다.
    - 최종 리포트의 상단에 들어갈 "오늘 시장 한 줄 요약"을 생성한다.
    - Candidate Scoring Agent가 사용할 수 있도록
      시장 키워드, 긍정 신호, 부정 신호, 리스크 필터를 구조화한다.

    생성 순서:
    1. Gemini 시도
    2. Gemini 실패 시 OpenAI 시도
    3. OpenAI도 실패하면 deterministic fallback 생성
    """

    stage = "market_flow"
    version = "1.1-gemini-openai-fallback-compact"

    def __init__(
        self,
        gemini_model_name: str = "gemini-2.5-flash",
        openai_model_name: str = "gpt-4.1-mini",
        encoding: str = "utf-8",
        max_rows_for_snapshot: int = 10,
        model_name: Optional[str] = None,
    ):
        if model_name is not None:
            gemini_model_name = model_name

        self.gemini_model_name = gemini_model_name
        self.openai_model_name = openai_model_name
        self.encoding = encoding
        self.max_rows_for_snapshot = max_rows_for_snapshot

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        load_dotenv()

        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = MarketFlowArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            result_path=output_dir / "market_flow_result.json",
            summary_md_path=output_dir / "market_flow_summary.md",
            manifest_path=output_dir / "manifest.json",
            output_files=[],
        )

        ctx.logger.info("[MarketFlowAgent] Starting market flow analysis.")

        paths = {
            # preprocessed files
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

            # upstream agent outputs
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

        ohlcv = self._safe_read_csv(paths["ohlcv"], ctx)
        news_feat = self._safe_read_csv(paths["news_feat"], ctx)
        foreign = self._safe_read_csv(paths["foreign"], ctx)
        finance = self._safe_read_csv(paths["finance"], ctx)
        user = self._safe_read_csv(paths["user"], ctx)

        market_json = self._safe_read_json(paths["market_json"], ctx)

        # RAG 버전이 있으면 우선 사용, 없으면 일반 news_invest_result 사용
        news_json_path = paths["news_json_rag"] or paths["news_json"]
        news_json = self._safe_read_json(news_json_path, ctx)

        risk_json = self._safe_read_json(paths["risk_json"], ctx)

        deterministic_snapshot = self._build_deterministic_snapshot(
            ohlcv=ohlcv,
            news_feat=news_feat,
            foreign=foreign,
            finance=finance,
            user=user,
            market_json=market_json,
            news_json=news_json,
            risk_json=risk_json,
            ctx=ctx,
        )

        snapshot_text = json.dumps(deterministic_snapshot, ensure_ascii=False)
        ctx.logger.info(
            f"[MarketFlowAgent] Compact snapshot chars={len(snapshot_text):,}"
        )

        llm_result = self._generate_market_flow(
            snapshot=deterministic_snapshot,
            ctx=ctx,
        )

        result = {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "purpose": "market_flow_based_candidate_report_context",
                "llm_providers": {
                    "primary": "gemini",
                    "gemini_model_name": self.gemini_model_name,
                    "fallback": "openai",
                    "openai_model_name": self.openai_model_name,
                },
                "compact_snapshot_chars": len(snapshot_text),
            },
            "input_files": {
                k: str(v) if v is not None else None
                for k, v in paths.items()
            },
            "deterministic_snapshot": deterministic_snapshot,
            "market_flow": llm_result,
        }

        self._save_json(result, artifacts.result_path)
        self._save_markdown(llm_result, artifacts.summary_md_path)

        artifacts.output_files = [
            str(artifacts.result_path),
            str(artifacts.summary_md_path),
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
            "llm_provider_used": llm_result.get("generation_provider"),
            "llm_model_used": llm_result.get("generation_model"),
            "compact_snapshot_chars": len(snapshot_text),
        }
        self._save_json(manifest, artifacts.manifest_path)

        metrics = {
            "has_ohlcv": ohlcv is not None,
            "has_news_features": news_feat is not None,
            "has_foreign": foreign is not None,
            "has_finance": finance is not None,
            "has_user": user is not None,
            "has_market_json": market_json is not None,
            "has_news_json": news_json is not None,
            "has_risk_json": risk_json is not None,
            "compact_snapshot_chars": len(snapshot_text),
            "n_core_drivers": len(llm_result.get("core_market_drivers", [])),
            "n_keywords": len(llm_result.get("market_keywords", [])),
            "generation_provider": llm_result.get("generation_provider"),
        }

        outputs = {
            "output_dir": str(output_dir),
            "market_flow_result": str(artifacts.result_path),
            "market_flow_summary": str(artifacts.summary_md_path),
            "manifest": str(artifacts.manifest_path),
        }

        ctx.logger.info("[MarketFlowAgent] Completed market flow analysis.")

        return self._make_stage_result(
            status="success",
            message="Market flow analysis completed.",
            metrics=metrics,
            outputs=outputs,
        )

    # =========================================================
    # Snapshot builder: compact deterministic summary
    # =========================================================

    def _build_deterministic_snapshot(
        self,
        ohlcv: Optional[pd.DataFrame],
        news_feat: Optional[pd.DataFrame],
        foreign: Optional[pd.DataFrame],
        finance: Optional[pd.DataFrame],
        user: Optional[pd.DataFrame],
        market_json: Optional[Dict[str, Any]],
        news_json: Optional[Dict[str, Any]],
        risk_json: Optional[Dict[str, Any]],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        """
        LLM에 전체 JSON/CSV를 넘기지 않고,
        시장 흐름 판단에 필요한 요약 정보만 만든다.

        핵심:
        - market_analysis_result 전체 X -> market_overview 일부만
        - news_invest_result 전체 X -> 뉴스 신호 상위 10개만
        - risk_score_result 전체 X -> 리스크 요약 + high risk 상위 10개만
        - finance 전체 연도 X -> 최신 연도 기준 평균 요약만
        """
        return {
            "as_of_date": self._infer_as_of_date(ohlcv),

            # CSV 기반 compact 요약
            "price_market_snapshot": self._summarize_ohlcv_compact(ohlcv, ctx),
            "news_feature_overview": self._summarize_news_features_compact(news_feat, ctx),
            "foreign_flow_overview": self._summarize_foreign_flow_compact(foreign, ctx),
            "finance_overview": self._summarize_finance_compact(finance, ctx),
            "user_overview": self._summarize_user_compact(user),

            # 기존 Agent 결과 compact 요약
            "market_overview": self._extract_market_overview(market_json),
            "news_overview": self._extract_news_overview(news_json),
            "risk_overview": self._extract_risk_overview(risk_json),
        }

    def _summarize_ohlcv_compact(
        self,
        df: Optional[pd.DataFrame],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"exists": False}

        ticker_col = self._infer_col(df, ["종목코드", "ticker", "code", "symbol"])
        date_col = self._infer_col(df, ["date", "datetime", "dt", "일자", "날짜"])
        close_col = self._infer_col(df, ["close", "Close", "종가"])
        volume_col = self._infer_col(df, ["volume", "Volume", "거래량"])

        if ticker_col is None or date_col is None or close_col is None:
            return {
                "exists": True,
                "warning": "ticker/date/close column not found",
                "n_rows": int(len(df)),
                "columns": list(df.columns)[:50],
            }

        work = df.copy()
        work[ticker_col] = work[ticker_col].apply(self._zfill6)
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work = work.sort_values([ticker_col, date_col])

        latest_date = work[date_col].max()

        last_two = work.groupby(ticker_col).tail(2).copy()
        last_two["prev_close"] = last_two.groupby(ticker_col)[close_col].shift(1)

        last_two["daily_return"] = (
            pd.to_numeric(last_two[close_col], errors="coerce")
            / pd.to_numeric(last_two["prev_close"], errors="coerce")
            - 1.0
        )

        latest_ret = last_two[last_two[date_col] == latest_date].copy()

        keep_cols = [ticker_col, close_col, "daily_return"]
        if volume_col:
            keep_cols.append(volume_col)

        top_gainers = (
            latest_ret.dropna(subset=["daily_return"])
            .sort_values("daily_return", ascending=False)
            .head(self.max_rows_for_snapshot)[keep_cols]
            .to_dict(orient="records")
        )

        top_losers = (
            latest_ret.dropna(subset=["daily_return"])
            .sort_values("daily_return", ascending=True)
            .head(self.max_rows_for_snapshot)[keep_cols]
            .to_dict(orient="records")
        )

        return {
            "exists": True,
            "latest_date": str(latest_date.date()) if pd.notna(latest_date) else None,
            "n_latest_tickers": int(len(latest_ret)),
            "advancers_ratio": self._safe_float((latest_ret["daily_return"] > 0).mean()),
            "average_daily_return": self._safe_float(latest_ret["daily_return"].mean()),
            "median_daily_return": self._safe_float(latest_ret["daily_return"].median()),
            "top_gainers": self._json_safe_records(top_gainers),
            "top_losers": self._json_safe_records(top_losers),
        }

    def _summarize_news_features_compact(
        self,
        df: Optional[pd.DataFrame],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"exists": False}

        ticker_col = self._infer_col(df, ["종목코드", "ticker", "code", "symbol"])
        if ticker_col is None:
            return {
                "exists": True,
                "warning": "ticker column not found",
                "n_rows": int(len(df)),
                "columns": list(df.columns)[:50],
            }

        work = df.copy()
        work[ticker_col] = work[ticker_col].apply(self._zfill6)

        numeric_targets = [
            "recent_news_ratio",
            "event_news_ratio",
            "negative_news_ratio",
            "time_concentration_score",
            "latest_negative_flag",
        ]

        for col in numeric_targets:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        keep_cols = [
            c for c in [
                ticker_col,
                "recent_news_ratio",
                "event_news_ratio",
                "negative_news_ratio",
                "time_concentration_score",
                "latest_negative_flag",
            ]
            if c in work.columns
        ]

        top_event_news = []
        if "event_news_ratio" in work.columns:
            top_event_news = (
                work.sort_values("event_news_ratio", ascending=False)
                .head(self.max_rows_for_snapshot)[keep_cols]
                .to_dict(orient="records")
            )

        top_negative_news = []
        if "negative_news_ratio" in work.columns:
            top_negative_news = (
                work.sort_values("negative_news_ratio", ascending=False)
                .head(self.max_rows_for_snapshot)[keep_cols]
                .to_dict(orient="records")
            )

        return {
            "exists": True,
            "n_rows": int(len(work)),
            "avg_event_news_ratio": self._safe_float(work["event_news_ratio"].mean())
            if "event_news_ratio" in work.columns else None,
            "avg_negative_news_ratio": self._safe_float(work["negative_news_ratio"].mean())
            if "negative_news_ratio" in work.columns else None,
            "top_event_news": self._json_safe_records(top_event_news),
            "top_negative_news": self._json_safe_records(top_negative_news),
        }

    def _summarize_foreign_flow_compact(
        self,
        df: Optional[pd.DataFrame],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"exists": False}

        ticker_col = self._infer_col(df, ["종목코드", "ticker", "code", "symbol"])
        date_col = self._infer_col(df, ["date", "datetime", "dt", "일자", "날짜"])

        if ticker_col is None:
            return {
                "exists": True,
                "warning": "ticker column not found",
                "n_rows": int(len(df)),
                "columns": list(df.columns)[:50],
            }

        work = df.copy()
        work[ticker_col] = work[ticker_col].apply(self._zfill6)

        if date_col:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            latest_date = work[date_col].max()
            work = work[work[date_col] == latest_date].copy()
        else:
            latest_date = None

        for col in [
            "foreign_net_flow_ratio",
            "foreign_ownership_level",
            "frgn_ntby_qty",
            "volume",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        if "foreign_net_flow_ratio" not in work.columns:
            return {
                "exists": True,
                "latest_date": str(latest_date.date()) if pd.notna(latest_date) else None,
                "warning": "foreign_net_flow_ratio column not found",
            }

        keep_cols = [
            c for c in [
                ticker_col,
                "foreign_net_flow_ratio",
                "foreign_ownership_level",
                "frgn_ntby_qty",
                "volume",
            ]
            if c in work.columns
        ]

        top_buy = (
            work.sort_values("foreign_net_flow_ratio", ascending=False)
            .head(self.max_rows_for_snapshot)[keep_cols]
            .to_dict(orient="records")
        )

        top_sell = (
            work.sort_values("foreign_net_flow_ratio", ascending=True)
            .head(self.max_rows_for_snapshot)[keep_cols]
            .to_dict(orient="records")
        )

        return {
            "exists": True,
            "latest_date": str(latest_date.date()) if pd.notna(latest_date) else None,
            "avg_foreign_net_flow_ratio": self._safe_float(
                work["foreign_net_flow_ratio"].mean()
            ),
            "top_foreign_buy": self._json_safe_records(top_buy),
            "top_foreign_sell": self._json_safe_records(top_sell),
        }

    def _summarize_finance_compact(
        self,
        df: Optional[pd.DataFrame],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"exists": False}

        ticker_col = self._infer_col(df, ["종목코드", "ticker", "code", "symbol"])
        year_col = self._infer_col(df, ["year", "년도"])

        if ticker_col is None:
            return {
                "exists": True,
                "warning": "ticker column not found",
                "n_rows": int(len(df)),
                "columns": list(df.columns)[:50],
            }

        work = df.copy()
        work[ticker_col] = work[ticker_col].apply(self._zfill6)

        if year_col:
            work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
            work = (
                work.dropna(subset=[year_col])
                .sort_values([ticker_col, year_col])
                .groupby(ticker_col, as_index=False)
                .tail(1)
                .copy()
            )

        for col in [
            "revenue_yoy",
            "operating_income_yoy",
            "roe",
            "debt_ratio",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        return {
            "exists": True,
            "n_latest_tickers": int(len(work)),
            "latest_year_min": self._safe_float(work[year_col].min())
            if year_col and year_col in work.columns else None,
            "latest_year_max": self._safe_float(work[year_col].max())
            if year_col and year_col in work.columns else None,
            "avg_revenue_yoy": self._safe_float(work["revenue_yoy"].mean())
            if "revenue_yoy" in work.columns else None,
            "avg_operating_income_yoy": self._safe_float(work["operating_income_yoy"].mean())
            if "operating_income_yoy" in work.columns else None,
            "avg_roe": self._safe_float(work["roe"].mean())
            if "roe" in work.columns else None,
            "avg_debt_ratio": self._safe_float(work["debt_ratio"].mean())
            if "debt_ratio" in work.columns else None,
        }

    def _summarize_user_compact(
        self,
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"exists": False}

        out = {
            "exists": True,
            "n_users": int(len(df)),
        }

        for col in [
            "risk_score",
            "investment_horizon_score",
            "avg_dwell_time",
            "event_frequency_7d",
            "high_risk_action_ratio",
            "exploration_ratio",
            "session_completion_rate",
        ]:
            if col in df.columns:
                out[f"avg_{col}"] = self._safe_float(
                    pd.to_numeric(df[col], errors="coerce").mean()
                )

        return out

    def _extract_market_overview(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not obj:
            return {"exists": False}

        market_overview = obj.get("market_overview", {}) or {}
        diagnostics = obj.get("diagnostics", {}) or {}

        return {
            "exists": True,
            "as_of_date": market_overview.get("as_of_date"),
            "market_phase": market_overview.get("market_phase"),
            "market_tone": market_overview.get("market_tone"),
            "market_rsi_state": market_overview.get("market_rsi_state"),
            "summary": market_overview.get("summary"),
            "evidence": self._safe_list(market_overview.get("evidence"), limit=5),
            "risk_notes": self._safe_list(market_overview.get("risk_notes"), limit=5),
            "regime_distribution": market_overview.get("regime_distribution", {}),
            "aggregate_metrics": market_overview.get("aggregate_metrics", {}),
            "ticker_type_distribution": diagnostics.get("ticker_type_distribution", {}),
            "rsi_state_distribution": diagnostics.get("rsi_state_distribution", {}),
            "return_20d_state_distribution": diagnostics.get("return_20d_state_distribution", {}),
            "return_252d_state_distribution": diagnostics.get("return_252d_state_distribution", {}),
        }

    def _extract_news_overview(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not obj:
            return {"exists": False}

        universe_summary = obj.get("universe_summary", {}) or {}
        tickers = obj.get("tickers", []) or []

        if not isinstance(tickers, list):
            tickers = []

        sorted_tickers = sorted(
            tickers,
            key=lambda x: x.get("news_signal_score", 0) or 0,
            reverse=True,
        )

        high_signal = []

        for row in sorted_tickers[:self.max_rows_for_snapshot]:
            if not isinstance(row, dict):
                continue

            high_signal.append({
                "ticker": row.get("ticker"),
                "as_of_date": row.get("as_of_date"),
                "news_signal_score": row.get("news_signal_score"),
                "confidence_level": row.get("confidence_level"),
                "verdict": row.get("verdict"),
                "reasons": self._safe_list(row.get("reasons"), limit=3),
                "top_articles": self._compact_top_articles(row.get("top_articles", [])),
            })

        return {
            "exists": True,
            "universe_summary": universe_summary,
            "high_signal_tickers": high_signal,
        }

    def _extract_risk_overview(
        self,
        obj: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not obj:
            return {"exists": False}

        universe_summary = obj.get("universe_summary", {}) or {}
        tickers = obj.get("tickers", []) or []

        if not isinstance(tickers, list):
            tickers = []

        def get_risk_score(row: Dict[str, Any]) -> float:
            scores = row.get("risk_scores", {}) or {}
            try:
                return float(scores.get("overall_risk_score") or 0)
            except Exception:
                return 0.0

        sorted_tickers = sorted(
            [x for x in tickers if isinstance(x, dict)],
            key=get_risk_score,
            reverse=True,
        )

        high_risk = []

        for row in sorted_tickers[:self.max_rows_for_snapshot]:
            scores = row.get("risk_scores", {}) or {}

            high_risk.append({
                "ticker": row.get("ticker"),
                "as_of_date": row.get("as_of_date"),
                "overall_risk_score": scores.get("overall_risk_score"),
                "risk_level": scores.get("risk_level"),
                "dominant_risk_factors": self._safe_list(
                    row.get("dominant_risk_factors"),
                    limit=3,
                ),
                "evidence": self._safe_list(row.get("evidence"), limit=3),
            })

        return {
            "exists": True,
            "universe_summary": universe_summary,
            "high_risk_tickers": high_risk,
        }

    def _compact_top_articles(
        self,
        articles: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(articles, list):
            return []

        out = []

        for article in articles[:3]:
            if isinstance(article, dict):
                out.append({
                    "title": article.get("title") or article.get("headline"),
                    "source": article.get("source"),
                    "published_at": article.get("published_at") or article.get("date"),
                    "url": article.get("url"),
                })
            else:
                out.append({
                    "title": str(article)[:200],
                })

        return out

    # =========================================================
    # LLM generation: Gemini -> OpenAI -> fallback
    # =========================================================

    def _generate_market_flow(
        self,
        snapshot: Dict[str, Any],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        gemini_error = None
        openai_error = None

        # 1) Gemini first
        try:
            result = self._generate_market_flow_with_gemini(
                snapshot=snapshot,
                ctx=ctx,
            )
            result = self._validate_market_flow_result(result)
            result["generation_provider"] = "gemini"
            result["generation_model"] = self.gemini_model_name
            return result

        except Exception as e:
            gemini_error = str(e)
            ctx.logger.warning(
                f"[MarketFlowAgent] Gemini failed. Trying OpenAI next. Error: {gemini_error}"
            )

        # 2) OpenAI fallback
        try:
            result = self._generate_market_flow_with_openai(
                snapshot=snapshot,
                ctx=ctx,
            )
            result = self._validate_market_flow_result(result)
            result["generation_provider"] = "openai"
            result["generation_model"] = self.openai_model_name
            result["gemini_error"] = gemini_error
            return result

        except Exception as e:
            openai_error = str(e)
            ctx.logger.warning(
                f"[MarketFlowAgent] OpenAI failed. Using deterministic fallback. Error: {openai_error}"
            )

        # 3) deterministic fallback
        return self._fallback_market_flow(
            snapshot=snapshot,
            error=f"Gemini failed: {gemini_error} | OpenAI failed: {openai_error}",
        )

    def _generate_market_flow_with_gemini(
        self,
        snapshot: Dict[str, Any],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is missing.")

        client = genai.Client(api_key=api_key)
        prompt = self._build_prompt(snapshot)

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                ctx.logger.info(
                    f"[MarketFlowAgent] Gemini attempt {attempt}/{max_retries}"
                )

                response = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )

                text = response.text or ""
                parsed = self._parse_json_response(text)
                return self._validate_market_flow_result(parsed)

            except Exception as e:
                err_msg = str(e)
                retryable = self._is_retryable_error(err_msg)

                ctx.logger.warning(
                    f"[MarketFlowAgent] Gemini attempt {attempt} failed: {err_msg}"
                )

                if not retryable or attempt == max_retries:
                    raise

                sleep_sec = min(30, (2 ** attempt) + random.uniform(0, 2))
                ctx.logger.info(
                    f"[MarketFlowAgent] Retrying Gemini after {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)

        raise RuntimeError("Gemini generation failed after retries.")

    def _generate_market_flow_with_openai(
        self,
        snapshot: Dict[str, Any],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is missing.")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK is not installed. Run: pip install openai"
            ) from e

        client = OpenAI(api_key=api_key)
        prompt = self._build_prompt(snapshot)
        schema = self._market_flow_json_schema()

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                ctx.logger.info(
                    f"[MarketFlowAgent] OpenAI attempt {attempt}/{max_retries}"
                )

                response = client.responses.create(
                    model=self.openai_model_name,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "You are a market-flow analysis agent for a Korean "
                                "beginner-friendly stock report. Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "market_flow_result",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                    temperature=0.2,
                )

                text = response.output_text or ""
                parsed = self._parse_json_response(text)
                return self._validate_market_flow_result(parsed)

            except Exception as e:
                err_msg = str(e)
                retryable = self._is_retryable_error(err_msg)

                ctx.logger.warning(
                    f"[MarketFlowAgent] OpenAI attempt {attempt} failed: {err_msg}"
                )

                if not retryable or attempt == max_retries:
                    raise

                sleep_sec = min(30, (2 ** attempt) + random.uniform(0, 2))
                ctx.logger.info(
                    f"[MarketFlowAgent] Retrying OpenAI after {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)

        raise RuntimeError("OpenAI generation failed after retries.")

    def _build_prompt(self, snapshot: Dict[str, Any]) -> str:
        snapshot_text = json.dumps(snapshot, ensure_ascii=False, indent=2)

        return f"""
당신은 한국 주식 초보 투자자를 위한 금융 리포트의 Market Flow Agent입니다.

목표:
- 전처리된 시세/뉴스/외국인 수급/재무 데이터와 기존 market/news/risk agent 결과 요약을 종합합니다.
- 최종 리포트의 맨 위에 들어갈 "오늘 시장 한 줄 요약"을 생성합니다.
- 이후 Candidate Scoring Agent가 사용할 수 있도록 시장 핵심 키워드, 긍정 신호, 부정 신호, 관련 업종/테마 힌트를 구조화합니다.
- 결과는 초보 투자자가 이해할 수 있는 쉬운 한국어로 작성합니다.
- 단, 과장된 투자 권유 표현은 피하고, "후보", "주목", "확인 필요" 정도의 표현을 사용합니다.

중요:
- 반드시 입력 데이터에 근거해서만 작성하세요.
- 없는 사실을 지어내지 마세요.
- 특정 종목 매수/매도 권유처럼 쓰지 마세요.
- 보고서 제목의 방향은 "시장 흐름 기반 유망 종목 후보 리포트"입니다.
- 출력은 반드시 JSON만 반환하세요.

출력 JSON 스키마:
{{
  "as_of_date": "YYYY-MM-DD 또는 빈 문자열",
  "market_one_line_summary": "초보 투자자용 한 줄 요약",
  "market_brief": "2~3문장 시장 흐름 설명",
  "market_keywords": ["키워드1", "키워드2", "키워드3"],
  "core_market_drivers": [
    {{
      "driver": "핵심 시장 동인",
      "direction": "positive | negative | neutral",
      "evidence": "입력 데이터 기반 근거",
      "beginner_explanation": "초보 투자자용 쉬운 설명"
    }}
  ],
  "candidate_scoring_hints": {{
    "positive_signals": ["후보 선정 시 가점 신호"],
    "negative_signals": ["후보 선정 시 감점 신호"],
    "preferred_sectors_or_themes": ["관련 업종 또는 테마"],
    "risk_filters": ["주의해야 할 리스크 필터"]
  }},
  "report_summary_box": {{
    "title": "오늘 시장 한 줄 요약",
    "body": "리포트 상단 요약 박스에 들어갈 문장",
    "source_note": "뉴스, 시세 데이터 기반 추출"
  }},
  "limitations": ["데이터 한계 또는 해석 시 주의점"]
}}

입력 데이터 스냅샷:
{snapshot_text}
""".strip()

    def _market_flow_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "as_of_date": {
                    "type": "string",
                },
                "market_one_line_summary": {
                    "type": "string",
                },
                "market_brief": {
                    "type": "string",
                },
                "market_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "core_market_drivers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "driver": {"type": "string"},
                            "direction": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                            },
                            "evidence": {"type": "string"},
                            "beginner_explanation": {"type": "string"},
                        },
                        "required": [
                            "driver",
                            "direction",
                            "evidence",
                            "beginner_explanation",
                        ],
                    },
                },
                "candidate_scoring_hints": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "positive_signals": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "negative_signals": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "preferred_sectors_or_themes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "risk_filters": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "positive_signals",
                        "negative_signals",
                        "preferred_sectors_or_themes",
                        "risk_filters",
                    ],
                },
                "report_summary_box": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "source_note": {"type": "string"},
                    },
                    "required": [
                        "title",
                        "body",
                        "source_note",
                    ],
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "as_of_date",
                "market_one_line_summary",
                "market_brief",
                "market_keywords",
                "core_market_drivers",
                "candidate_scoring_hints",
                "report_summary_box",
                "limitations",
            ],
        }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            return json.loads(candidate)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)

        raise ValueError("Failed to parse LLM response as JSON.")

    def _validate_market_flow_result(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            obj = {}

        defaults = {
            "as_of_date": "",
            "market_one_line_summary": "",
            "market_brief": "",
            "market_keywords": [],
            "core_market_drivers": [],
            "candidate_scoring_hints": {
                "positive_signals": [],
                "negative_signals": [],
                "preferred_sectors_or_themes": [],
                "risk_filters": [],
            },
            "report_summary_box": {
                "title": "오늘 시장 한 줄 요약",
                "body": "",
                "source_note": "뉴스, 시세 데이터 기반 추출",
            },
            "limitations": [],
        }

        for key, default_value in defaults.items():
            if key not in obj or obj[key] is None:
                obj[key] = default_value

        hints = obj.get("candidate_scoring_hints")
        if not isinstance(hints, dict):
            hints = {}
        for k in [
            "positive_signals",
            "negative_signals",
            "preferred_sectors_or_themes",
            "risk_filters",
        ]:
            if k not in hints or not isinstance(hints[k], list):
                hints[k] = []
        obj["candidate_scoring_hints"] = hints

        if not isinstance(obj.get("market_keywords"), list):
            obj["market_keywords"] = []

        if not isinstance(obj.get("core_market_drivers"), list):
            obj["core_market_drivers"] = []

        if not isinstance(obj.get("limitations"), list):
            obj["limitations"] = []

        box = obj.get("report_summary_box")
        if not isinstance(box, dict):
            box = {}
        box.setdefault("title", "오늘 시장 한 줄 요약")
        box.setdefault("body", obj.get("market_one_line_summary", ""))
        box.setdefault("source_note", "뉴스, 시세 데이터 기반 추출")
        obj["report_summary_box"] = box

        return obj

    def _fallback_market_flow(
        self,
        snapshot: Dict[str, Any],
        error: str,
    ) -> Dict[str, Any]:
        as_of_date = snapshot.get("as_of_date") or ""

        price = snapshot.get("price_market_snapshot", {}) or {}
        market_overview = snapshot.get("market_overview", {}) or {}
        news_overview = snapshot.get("news_overview", {}) or {}
        risk_overview = snapshot.get("risk_overview", {}) or {}
        foreign_overview = snapshot.get("foreign_flow_overview", {}) or {}

        market_phase = market_overview.get("market_phase") or "혼재 국면"
        market_tone = market_overview.get("market_tone") or "mixed"
        market_summary = market_overview.get("summary") or ""

        advancers_ratio = price.get("advancers_ratio")
        avg_return = price.get("average_daily_return")
        median_return = price.get("median_daily_return")

        risk_counts = (
            risk_overview.get("universe_summary", {})
            .get("risk_level_counts", {})
        )

        high_news = news_overview.get("high_signal_tickers", [])
        top_foreign_buy = foreign_overview.get("top_foreign_buy", [])

        preferred_themes = []

        if high_news:
            preferred_themes.append("뉴스 신호 강한 종목군")

        if top_foreign_buy:
            preferred_themes.append("외국인 순매수 상위 종목군")

        brief_parts = [
            f"시장 분석 결과, 현재 시장은 '{market_phase}'으로 분류됩니다."
        ]

        if market_summary:
            brief_parts.append(market_summary)

        if advancers_ratio is not None:
            brief_parts.append(
                f"분석 대상 종목 중 상승 종목 비율은 약 {advancers_ratio * 100:.1f}%입니다."
            )

        if avg_return is not None and median_return is not None:
            brief_parts.append(
                f"평균 일간 수익률은 {avg_return * 100:.2f}%, "
                f"중앙값 수익률은 {median_return * 100:.2f}%입니다."
            )

        if risk_counts:
            brief_parts.append(
                f"리스크 분포는 low {risk_counts.get('low', 0)}개, "
                f"medium {risk_counts.get('medium', 0)}개, "
                f"high {risk_counts.get('high', 0)}개입니다."
            )

        market_brief = " ".join(brief_parts)

        if market_tone == "positive":
            direction = "positive"
        elif market_tone == "negative":
            direction = "negative"
        else:
            direction = "neutral"

        summary_body = (
            f"시장 분석 결과, 현재 시장은 {market_phase}으로 나타났고 "
            "종목별 차별화와 리스크 점검이 중요한 흐름입니다."
        )

        return {
            "as_of_date": as_of_date,
            "market_one_line_summary": summary_body,
            "market_brief": market_brief,
            "market_keywords": [
                market_phase,
                "종목별 차별화",
                "뉴스 신호",
                "외국인 수급",
                "리스크 점검",
            ],
            "core_market_drivers": [
                {
                    "driver": market_phase,
                    "direction": direction,
                    "evidence": market_summary or "Market Analysis Agent의 시장 국면 결과 기반",
                    "beginner_explanation": (
                        "시장 전체가 한 방향으로만 움직이기보다는, "
                        "종목마다 상승과 하락이 다르게 나타나는 상태예요."
                    ),
                },
                {
                    "driver": "뉴스 신호 차별화",
                    "direction": "neutral",
                    "evidence": f"뉴스 신호 상위 종목 수: {len(high_news)}",
                    "beginner_explanation": (
                        "뉴스가 많이 나오거나 중요한 이슈가 있는 종목은 "
                        "단기적으로 투자자 관심이 커질 수 있어요."
                    ),
                },
                {
                    "driver": "리스크 필터 필요",
                    "direction": "negative",
                    "evidence": (
                        f"high risk 종목 수: {risk_counts.get('high', 0)}"
                        if risk_counts else "Risk Agent 결과 기반"
                    ),
                    "beginner_explanation": (
                        "상승 가능성이 있어 보여도 리스크가 큰 종목은 "
                        "후보 선정 시 감점하거나 주의 표시가 필요해요."
                    ),
                },
            ],
            "candidate_scoring_hints": {
                "positive_signals": [
                    "시장 흐름 대비 강한 상승형 종목",
                    "뉴스 신호 점수가 높은 종목",
                    "외국인 순매수 비율이 높은 종목",
                    "거래량 또는 거래대금 증가가 확인되는 종목",
                    "리스크 레벨이 low 또는 medium인 종목",
                ],
                "negative_signals": [
                    "risk_level이 high인 종목",
                    "RSI 강한 과열 구간",
                    "부정 뉴스 비율 상승",
                    "중기 급락 또는 높은 변동성",
                    "외국인 순매도 비율이 큰 종목",
                ],
                "preferred_sectors_or_themes": preferred_themes,
                "risk_filters": [
                    "risk_level high 종목은 후보 점수에서 감점",
                    "RSI 강한 과열 종목은 단기 과열 주의 표시",
                    "negative_news_ratio가 높은 종목은 뉴스 리스크 표시",
                ],
            },
            "report_summary_box": {
                "title": "오늘 시장 한 줄 요약",
                "body": summary_body,
                "source_note": "시세, 뉴스, 외국인 수급, 리스크 데이터 기반 추출",
            },
            "limitations": [
                "Gemini와 OpenAI 생성이 모두 실패하여 deterministic snapshot 기반 fallback 요약을 사용했습니다.",
                f"LLM generation error: {error}",
            ],
            "generation_provider": "deterministic_fallback",
            "generation_model": "",
        }

    # =========================================================
    # Markdown output
    # =========================================================

    def _save_markdown(self, result: Dict[str, Any], path: Path) -> None:
        lines = []

        lines.append("# Market Flow Summary")
        lines.append("")
        lines.append(f"- 기준일: {result.get('as_of_date')}")
        lines.append(f"- 생성 방식: {result.get('generation_provider')}")
        lines.append(f"- 사용 모델: {result.get('generation_model')}")
        lines.append("")
        lines.append("## 오늘 시장 한 줄 요약")
        lines.append("")
        lines.append(result.get("market_one_line_summary", ""))
        lines.append("")
        lines.append("## 시장 설명")
        lines.append("")
        lines.append(result.get("market_brief", ""))
        lines.append("")
        lines.append("## 핵심 키워드")
        lines.append("")

        for kw in result.get("market_keywords", []):
            lines.append(f"- {kw}")

        lines.append("")
        lines.append("## 핵심 시장 동인")
        lines.append("")

        for item in result.get("core_market_drivers", []):
            lines.append(f"### {item.get('driver', '')}")
            lines.append(f"- 방향: {item.get('direction', '')}")
            lines.append(f"- 근거: {item.get('evidence', '')}")
            lines.append(f"- 쉬운 설명: {item.get('beginner_explanation', '')}")
            lines.append("")

        lines.append("## Candidate Scoring Hints")
        lines.append("")

        hints = result.get("candidate_scoring_hints", {})

        for key in [
            "positive_signals",
            "negative_signals",
            "preferred_sectors_or_themes",
            "risk_filters",
        ]:
            lines.append(f"### {key}")
            for v in hints.get(key, []):
                lines.append(f"- {v}")
            lines.append("")

        lines.append("## Limitations")
        lines.append("")
        for v in result.get("limitations", []):
            lines.append(f"- {v}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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
            ctx.logger.info(f"[MarketFlowAgent] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            ctx.logger.warning(
                f"[MarketFlowAgent] Multiple files matched: {pattern}. "
                f"Using first: {matches[0]}"
            )

        return matches[0]

    def _safe_read_csv(
        self,
        path: Optional[Path],
        ctx: RunContext,
    ) -> Optional[pd.DataFrame]:
        if path is None:
            return None

        if not path.exists():
            ctx.logger.warning(f"[MarketFlowAgent] Missing CSV: {path}")
            return None

        try:
            return pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            ctx.logger.warning(f"[MarketFlowAgent] Failed to read CSV: {path} ({e})")
            return None

    def _safe_read_json(
        self,
        path: Optional[Path],
        ctx: RunContext,
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            return None

        if not path.exists():
            ctx.logger.warning(f"[MarketFlowAgent] Missing JSON: {path}")
            return None

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            ctx.logger.warning(f"[MarketFlowAgent] Failed to read JSON: {path} ({e})")
            return None

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
    def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _infer_as_of_date(self, ohlcv: Optional[pd.DataFrame]) -> Optional[str]:
        if ohlcv is None or ohlcv.empty:
            return None

        date_col = self._infer_col(ohlcv, ["date", "datetime", "dt", "일자", "날짜"])
        if date_col is None:
            return None

        dates = pd.to_datetime(ohlcv[date_col], errors="coerce").dropna()
        if dates.empty:
            return None

        return str(dates.max().date())

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _safe_list(value: Any, limit: int = 5) -> List[Any]:
        if isinstance(value, list):
            return value[:limit]
        if value is None:
            return []
        return [value]

    def _json_safe_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._json_safe_obj(row) for row in records]

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
            if np.isnan(obj):
                return None
            return round(float(obj), 6)

        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            return round(obj, 6)

        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

        return obj

    @staticmethod
    def _is_retryable_error(err_msg: str) -> bool:
        lower = err_msg.lower()

        return (
            "429" in err_msg
            or "500" in err_msg
            or "502" in err_msg
            or "503" in err_msg
            or "504" in err_msg
            or "unavailable" in lower
            or "resource_exhausted" in lower
            or "rate" in lower
            or "timeout" in lower
            or "temporarily" in lower
        )

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