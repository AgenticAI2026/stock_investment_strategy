from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


# ── 수준 레이블 ────────────────────────────────────────────────────────────────
LEVEL_LABELS = {
    "beginner":  "초보",
    "intermediate": "중급",
    "advanced":  "고급",
}

TERM_GLOSSARY = {
    "SpearmanRankIC": "종목 순위 예측 정확도(Spearman IC)",
    "NDCG@10":        "상위 10개 종목 선별 품질(NDCG@10)",
    "AUC":            "하락 예측 정확도(AUC)",
    "F1":             "하락 감지 균형 성능(F1)",
    "pred_rank_score_20d":      "20일 후 예상 순위 점수",
    "pred_rank_percentile_20d": "20일 후 상위 몇 % 예측",
    "proba_top_quantile_20d":   "상위 20% 진입 확률",
    "proba_downside_20d":       "20일 내 하락 확률",
    "blend_score":              "종합 투자 매력도",
}

RISK_LEVEL_LABEL = {
    "low":      "낮음",
    "medium":   "중간",
    "high":     "높음",
    "critical": "매우 높음",
}

RISK_LEVEL_ICON = {
    "low":      "🟢",
    "medium":   "🟡",
    "high":     "🔴",
    "critical": "🚨",
}


@dataclass
class ReportArtifacts:
    run_dir: Path
    output_dir: Path
    report_json_path: Path
    report_md_path: Path
    manifest_path: Path
    output_files: List[str]


class ReportGenerativeAgent(BaseAgent):
    stage = "report_gen"
    version = "1.3-survey-enhanced"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    # ═══════════════════════════════════════════════════════════════════════════
    # [수정 1] run() — user_id / user_level 파라미터 추가
    # ═══════════════════════════════════════════════════════════════════════════
    def run(
        self,
        ctx: RunContext,
        ap: ArtifactPaths,
        user_id: Optional[str] = None,
        user_level: str = "beginner",   # "beginner" | "intermediate" | "advanced"
    ) -> StageResult:
        """
        Parameters
        ----------
        user_id : str, optional
            personalized_scores.csv 에서 사용할 사용자 ID (예: "U001").
            None 이면 전체 평균으로 fallback.
        user_level : str
            "beginner" / "intermediate" / "advanced" — 설문 결과 반영:
            초보(81% 응답자) 는 용어 설명 병기, 핵심 요약 우선 표시.
        """
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = ReportArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            report_json_path=output_dir / "report_summary.json",
            report_md_path=output_dir / "report.md",
            manifest_path=output_dir / "manifest.json",
            output_files=[],
        )

        paths = {
            "metrics":            self._find_exact_file(run_dir, "model_infer_metrics.json"),
            "predictions":        self._find_exact_file(run_dir, "model_infer_predictions_20d.csv"),
            "feature_importance": self._find_exact_file(run_dir, "model_infer_feature_importance.json"),
            "personalized_scores":self._find_exact_file(run_dir, "model_infer_personalized_scores.csv"),
            "model_infer_manifest":self._find_exact_file(run_dir, "manifest.json", preferred_parent="model_infer"),
            "risk_json":          self._find_exact_file(run_dir, "risk_score_result.json"),
            "market_json":        self._find_exact_file(run_dir, "market_analysis_result.json"),
            "news_json":          self._find_exact_file(run_dir, "news_invest_rag_result.json"),
        }

        required = ["metrics", "predictions", "feature_importance", "personalized_scores"]
        missing = [k for k in required if paths[k] is None]
        if missing:
            raise FileNotFoundError(f"Missing required report inputs: {missing}")

        metrics           = self._read_json(paths["metrics"])
        feature_importance= self._read_json(paths["feature_importance"])
        pred              = pd.read_csv(paths["predictions"], encoding=self.encoding)
        personalized      = pd.read_csv(paths["personalized_scores"], encoding=self.encoding)

        self._normalize_ticker_date(pred)
        self._normalize_ticker_date(personalized)

        selected_models = metrics.get("selected_models") or {}
        rank_model      = selected_models.get("rank_20d")
        rank_pct_model  = selected_models.get("rank_percentile_20d")
        topq_model      = selected_models.get("top_quantile_20d")
        downside_model  = selected_models.get("downside_20d")

        pred_rank_col = self._resolve_col(
            pred,
            [f"pred_rank_score_20d__{rank_model}" if rank_model else None,
             f"pred_rank_20d__{rank_model}" if rank_model else None],
            prefixes=["pred_rank_score_20d__", "pred_rank_20d__", "pred_rank__"],
            fallback_contains=["rank"],
        )
        pred_rank_pct_col = self._resolve_col(
            pred,
            [f"pred_rank_percentile_20d__{rank_pct_model}" if rank_pct_model else None,
             f"pred_rank_pct_20d__{rank_pct_model}" if rank_pct_model else None],
            prefixes=["pred_rank_percentile_20d__", "pred_rank_pct_20d__"],
            fallback_contains=["percentile"],
            required=False,
        )
        topq_col = self._resolve_col(
            pred,
            [f"proba_top_quantile_20d__{topq_model}" if topq_model else None,
             f"pred_top_quantile_20d__{topq_model}" if topq_model else None],
            prefixes=["proba_top_quantile_20d__", "pred_top_quantile_20d__", "proba_top_quantile__"],
            fallback_contains=["top_quantile"],
            required=False,
        )
        downside_col = self._resolve_col(
            pred,
            [f"proba_downside_20d__{downside_model}" if downside_model else None,
             f"pred_downside_20d__{downside_model}" if downside_model else None],
            prefixes=["proba_downside_20d__", "pred_downside_20d__", "proba_downside__"],
            fallback_contains=["downside"],
        )

        date_col    = "date" if "date" in pred.columns else None
        latest_date = pd.to_datetime(pred[date_col]).max() if date_col else None

        if date_col:
            latest_pred = pred[pd.to_datetime(pred[date_col]) == latest_date].copy()
        else:
            latest_pred = pred.copy()

        latest_pred["pred_rank_score_20d"] = pd.to_numeric(latest_pred[pred_rank_col], errors="coerce")

        if pred_rank_pct_col:
            latest_pred["pred_rank_percentile_20d"] = pd.to_numeric(latest_pred[pred_rank_pct_col], errors="coerce")
        else:
            latest_pred["pred_rank_percentile_20d"] = None

        if topq_col:
            latest_pred["proba_top_quantile_20d"] = pd.to_numeric(latest_pred[topq_col], errors="coerce")
        else:
            latest_pred["proba_top_quantile_20d"] = None

        latest_pred["proba_downside_20d"] = pd.to_numeric(latest_pred[downside_col], errors="coerce")

        # ── [수정 1] 개인화 점수: user_id 있으면 해당 user만, 없으면 전체 평균 ──
        score_col = self._resolve_col(
            personalized,
            ["personalized_ranking_score", "personalized_score"],
            prefixes=["personalized_ranking_score", "personalized_score"],
            fallback_contains=["personalized"],
            required=False,
        )

        if score_col:
            if "as_of_date" in personalized.columns:
                personalized["as_of_date"] = pd.to_datetime(personalized["as_of_date"], errors="coerce")
                personalized_last = personalized[
                    personalized["as_of_date"] == personalized["as_of_date"].max()
                ].copy()
            else:
                personalized_last = personalized.copy()

            # user_id 필터링
            if user_id and "user_id" in personalized_last.columns:
                user_subset = personalized_last[personalized_last["user_id"] == user_id]
                if user_subset.empty:
                    print(f"[WARN] user_id={user_id} not found. falling back to all-user average.")
                    user_subset = personalized_last
            else:
                user_subset = personalized_last

            pers_agg = (
                user_subset.groupby("ticker", as_index=False)[score_col]
                .mean()
                .rename(columns={score_col: "personalized_score_avg"})
            )
            latest_pred = latest_pred.merge(pers_agg, on="ticker", how="left")
        else:
            latest_pred["personalized_score_avg"] = None

        risk_map   = self._load_optional_ticker_map(paths["risk_json"])
        market_map = self._load_optional_ticker_map(paths["market_json"],
                                                    list_key="ticker_analyses")
        news_map   = self._load_optional_ticker_map(paths["news_json"])

        risk_level_counts = None
        if paths["risk_json"] is not None:
            risk_json = self._read_json(paths["risk_json"])
            risk_level_counts = (risk_json.get("universe_summary") or {}).get("risk_level_counts")

        # ── [수정 1] blend_score: 개인화 점수를 가중치로 반영 ──
        pers_w = 0.20 if (score_col and user_id) else 0.0
        rank_w = 1.0 - pers_w

        latest_pred["blend_score"] = (
            rank_w * latest_pred["pred_rank_score_20d"].fillna(0)
            + 0.30  * latest_pred["proba_top_quantile_20d"].fillna(0)
            - 0.50  * latest_pred["proba_downside_20d"].fillna(0)
            + pers_w * latest_pred["personalized_score_avg"].fillna(0)
        )

        if risk_map:
            latest_pred["risk_level"] = latest_pred["ticker"].map(
                lambda t: (risk_map.get(t, {}).get("risk_scores") or {}).get("risk_level")
            )
            latest_pred = latest_pred[
                latest_pred["risk_level"].fillna("unknown").astype(str).str.lower().ne("critical")
            ].copy()

        top5 = (
            latest_pred.sort_values("blend_score", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

        # ── [수정 5] market_overview 로드 ──
        market_overview = None
        if paths["market_json"] is not None:
            raw_market = self._read_json(paths["market_json"])
            market_overview = raw_market.get("market_overview")

        report = {
            "meta": {
                "agent":          self.__class__.__name__,
                "stage":          self.stage,
                "version":        self.version,
                "created_at_utc": self._now_utc(),
                "inputs":         {k: str(v) if v is not None else None for k, v in paths.items()},
                "as_of_date":     str(pd.to_datetime(latest_date).date()) if pd.notna(latest_date) else None,
                "selected_models":selected_models,
                "user_id":        user_id,
                "user_level":     user_level,
                "prediction_columns": {
                    "rank":           pred_rank_col,
                    "rank_percentile":pred_rank_pct_col,
                    "top_quantile":   topq_col,
                    "downside":       downside_col,
                },
            },
            "performance":              self._build_performance_summary(metrics, selected_models),
            "feature_importance_top10": self._build_feature_importance_summary(feature_importance, selected_models),
            "risk_universe_summary":    {"risk_level_counts": risk_level_counts},
            "market_overview":          market_overview,   # [수정 5]
            "recommendations_top5":     [],
        }

        for _, row in top5.iterrows():
            ticker = row["ticker"]
            item = {
                "ticker":                  ticker,
                "as_of_date":              str(pd.to_datetime(row[date_col]).date())
                                           if date_col and pd.notna(row.get(date_col))
                                           else report["meta"]["as_of_date"],
                "pred_rank_score_20d":     self._safe_float(row.get("pred_rank_score_20d")),
                "pred_rank_percentile_20d":self._safe_float(row.get("pred_rank_percentile_20d")),
                "proba_top_quantile_20d":  self._safe_float(row.get("proba_top_quantile_20d")),
                "proba_downside_20d":      self._safe_float(row.get("proba_downside_20d")),
                "personalized_score_avg":  self._safe_float(row.get("personalized_score_avg")),
                "blend_score":             self._safe_float(row.get("blend_score")),
                "risk":                    self._get_risk_brief(ticker, risk_map) if risk_map else None,
                "market_brief":            self._get_market_brief(ticker, market_map) if market_map else None,  # [수정 2]
                "news_brief":              self._get_news_brief(ticker, news_map) if news_map else None,        # [수정 3]
            }
            report["recommendations_top5"].append(item)

        self._save_json(report, artifacts.report_json_path)

        md = self._build_markdown_report(report, user_level=user_level)
        artifacts.report_md_path.write_text(md, encoding="utf-8")

        artifacts.output_files = [
            str(artifacts.report_json_path),
            str(artifacts.report_md_path),
            str(artifacts.manifest_path),
        ]

        manifest = {
            "stage":           self.stage,
            "version":         self.version,
            "created_at_utc":  self._now_utc(),
            "input_files":     {k: str(v) if v is not None else None for k, v in paths.items()},
            "output_files":    artifacts.output_files,
            "as_of_date":      report["meta"]["as_of_date"],
            "user_id":         user_id,
            "user_level":      user_level,
            "top5_tickers":    [x["ticker"] for x in report["recommendations_top5"]],
        }

        self._save_json(manifest, artifacts.manifest_path)

        return self._make_stage_result(
            status="success",
            message="Report generation completed.",
            metrics={
                "top5_count":       len(report["recommendations_top5"]),
                "has_market_json":  paths["market_json"] is not None,
                "has_news_json":    paths["news_json"] is not None,
                "has_risk_json":    paths["risk_json"] is not None,
                "user_id":          user_id,
                "user_level":       user_level,
            },
            outputs={
                "output_dir":    str(output_dir),
                "report_summary":str(artifacts.report_json_path),
                "report_md":     str(artifacts.report_md_path),
                "manifest":      str(artifacts.manifest_path),
            },
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # File helpers (기존 유지)
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_exact_file(
        self,
        run_dir: Path,
        filename: str,
        preferred_parent: Optional[str] = None,
    ) -> Optional[Path]:
        matches = [p for p in run_dir.rglob(filename) if p.is_file() and p.name == filename]
        if preferred_parent:
            preferred = [p for p in matches if p.parent.name == preferred_parent]
            if preferred:
                return sorted(preferred)[0]
        if not matches:
            print(f"[WARN] No exact file matched: {filename}")
            return None
        if len(matches) > 1:
            print(f"[WARN] Multiple exact files matched: {filename}")
            for p in sorted(matches):
                print(f" - {p}")
        return sorted(matches)[0]

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding=self.encoding) as f:
            return json.load(f)

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # StageResult helper (기존 유지)
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_stage_result(self, status, message, metrics=None, outputs=None) -> StageResult:
        sig = inspect.signature(StageResult)
        candidate_kwargs = {
            "stage":    self.stage,
            "status":   status,
            "message":  message,
            "metrics":  metrics or {},
            "outputs":  outputs or {},
            "artifacts":outputs or {},
        }
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
        return StageResult(**kwargs)

    # ═══════════════════════════════════════════════════════════════════════════
    # Basic utils (기존 유지)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _zfill6(x: Any) -> str:
        return str(x).replace(".0", "").zfill(6)

    @staticmethod
    def _safe_float(x: Any, default: Any = None) -> Optional[float]:
        try:
            if x is None:
                return default
            v = float(x)
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default

    @classmethod
    def _fmt_num(cls, x: Any, nd: int = 4) -> str:
        v = cls._safe_float(x)
        if v is None:
            return "N/A"
        return f"{v:.{nd}f}"

    @classmethod
    def _fmt_pct(cls, x: Any, nd: int = 2) -> str:
        v = cls._safe_float(x)
        if v is None:
            return "N/A"
        return f"{v * 100:.{nd}f}%"

    def _normalize_ticker_date(self, df: pd.DataFrame) -> None:
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].apply(self._zfill6)
        elif "종목코드" in df.columns:
            df["ticker"] = df["종목코드"].apply(self._zfill6)
        else:
            raise ValueError("Input CSV must contain 'ticker' or '종목코드' column.")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "as_of_date" in df.columns:
            df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")

    def _resolve_col(
        self,
        df: pd.DataFrame,
        exact_candidates: List[Optional[str]],
        prefixes: List[str],
        fallback_contains: List[str],
        required: bool = True,
    ) -> Optional[str]:
        for col in exact_candidates:
            if col and col in df.columns:
                return col
        for prefix in prefixes:
            matches = [c for c in df.columns if str(c).startswith(prefix)]
            if matches:
                return matches[0]
        lowered = {c: str(c).lower() for c in df.columns}
        for keyword in fallback_contains:
            matches = [c for c, lc in lowered.items() if keyword.lower() in lc]
            if matches:
                return matches[0]
        if required:
            raise ValueError(
                f"Could not resolve required prediction column. "
                f"keywords={fallback_contains}, columns={list(df.columns)}"
            )
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Model/metric helpers (기존 유지)
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_performance_summary(self, metrics, selected_models):
        out = {}
        for target, model_id in selected_models.items():
            target_metrics = metrics.get(target, {})
            if isinstance(target_metrics, dict):
                out[target] = {
                    "selected_model": model_id,
                    "metrics": target_metrics.get(model_id, {}),
                }
        return out

    def _build_feature_importance_summary(self, feature_importance, selected_models, k=10):
        out = {}
        for target, model_id in selected_models.items():
            rows = feature_importance.get(model_id, [])
            parsed = []
            for r in rows:
                feat = r.get("feature")
                imp  = self._safe_float(r.get("importance"))
                if feat is None:
                    continue
                parsed.append({"feature": feat, "importance": imp})
            parsed = sorted(parsed, key=lambda x: abs(x["importance"]) if x["importance"] is not None else -1, reverse=True)
            out[target] = {"selected_model": model_id, "top_features": parsed[:k]}
        return out

    # ═══════════════════════════════════════════════════════════════════════════
    # Optional upstream JSON helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_optional_ticker_map(
        self,
        path: Optional[Path],
        key: Optional[str] = None,
        list_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        list_key : 최우선 탐색 키 (market_analysis → "ticker_analyses")
        key      : 차순위 탐색 키
        """
        if path is None or not path.exists():
            return {}
        obj = self._read_json(path)

        # 탐색 순서: list_key → key → 자동 탐색
        rows = None
        if list_key:
            rows = obj.get(list_key)
        if rows is None and key:
            rows = obj.get(key)
        if rows is None:
            rows = (
                obj.get("tickers")
                or obj.get("ticker_analyses")
                or obj.get("results")
                or []
            )

        out = {}
        for row in rows:
            ticker = row.get("ticker") or row.get("종목코드") or row.get("code")
            if ticker:
                out[self._zfill6(ticker)] = row
        return out

    def _get_risk_brief(self, ticker, risk_map) -> Dict[str, Any]:
        row    = risk_map.get(ticker, {})
        scores = row.get("risk_scores") or {}
        return {
            "risk_level":            scores.get("risk_level"),
            "overall_risk_score":    scores.get("overall_risk_score"),
            "dominant_risk_factors": (row.get("dominant_risk_factors") or [])[:3],
            "evidence":              (row.get("evidence") or [])[:3],
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # [수정 2] _get_market_brief — interpretation, positive_factors, risk_factors 추가
    # ═══════════════════════════════════════════════════════════════════════════
    def _get_market_brief(self, ticker, market_map) -> Dict[str, Any]:
        row            = market_map.get(ticker, {})
        compact        = row.get("compact_signals") or {}
        price_summary  = row.get("price_summary") or {}
        scores         = row.get("scores") or {}
        return_state   = row.get("return_state") or {}

        return {
            "ticker_type":          row.get("ticker_type"),
            "phase":                row.get("phase"),
            "tone":                 row.get("tone"),
            "rsi_state":            row.get("rsi_state"),
            "market_context":       row.get("market_context"),
            "interpretation":       row.get("interpretation"),
            "return_20d":           compact.get("return_20d") or price_summary.get("cum_return_20d"),
            "return_60d":           compact.get("return_60d") or price_summary.get("cum_return_60d"),
            "ann_volatility_252d":  compact.get("ann_volatility_252d") or price_summary.get("ann_volatility_252d"),
            "max_drawdown_252d":    compact.get("max_drawdown_252d") or price_summary.get("max_drawdown_252d"),
            "return_20d_state":     return_state.get("return_20d_state"),
            "return_60d_state":     return_state.get("return_60d_state"),
            "overall_score":        scores.get("overall_score"),
            "positive_factors":     (row.get("positive_factors") or [])[:3],
            "risk_factors":         (row.get("risk_factors") or [])[:2],
            "key_findings":         (row.get("key_findings") or [])[:4],
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # [수정 3] _get_news_brief — reasons, top_articles headline 추가
    # ═══════════════════════════════════════════════════════════════════════════
    def _get_news_brief(self, ticker, news_map) -> Dict[str, Any]:
        row     = news_map.get(ticker, {})
        summary = row.get("news_summary") or {}

        top_articles_raw = row.get("top_articles") or []
        top_headline = None
        if top_articles_raw:
            a = top_articles_raw[0]
            top_headline = {
                "date":  a.get("date"),
                "title": a.get("title"),
                "press": a.get("press"),
                "url":   a.get("url"),
            }

        return {
            "news_signal_score":       row.get("news_signal_score"),
            "confidence_level":        row.get("confidence_level"),
            "verdict":                 row.get("verdict"),
            "n_articles_30d_unique":   summary.get("n_articles_30d_unique"),
            "noise_ratio_0_1":         summary.get("noise_ratio_0_1"),
            "high_impact_ratio_0_1":   summary.get("high_impact_ratio_0_1"),
            "reasons":                 (row.get("reasons") or [])[:2],     # 신호 근거
            "top_headline":            top_headline,                        # 대표 기사 헤드라인
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Markdown helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _term(self, key: str, user_level: str) -> str:
        """초보 레벨에서는 용어 설명을 괄호 안에 병기."""
        label = TERM_GLOSSARY.get(key, key)
        if user_level == "beginner":
            return label
        return key   # 중급/고급은 원래 용어 그대로

    def _build_selection_reason(self, item: Dict[str, Any]) -> List[str]:
        reasons = []
        rank_score = self._safe_float(item.get("pred_rank_score_20d"))
        topq       = self._safe_float(item.get("proba_top_quantile_20d"))
        downside   = self._safe_float(item.get("proba_downside_20d"))
        blend      = self._safe_float(item.get("blend_score"))

        if rank_score is not None:
            reasons.append("20일 후 종목 순위 예측 점수가 상위권입니다.")
        if topq is not None and downside is not None:
            reasons.append("상위 수익 구간 진입 가능성과 하락 리스크를 함께 고려했습니다.")
        elif downside is not None:
            reasons.append("하락 리스크를 반영해 위험조정 관점에서 선별했습니다.")
        if blend is not None:
            if blend > 0:
                reasons.append("위험 대비 종합 점수(블렌드)가 양수입니다.")
            else:
                reasons.append("블렌드 점수는 낮지만 유니버스 내 상대 비교에서 상위권입니다.")

        risk   = item.get("risk") or {}
        rl     = str(risk.get("risk_level") or "").lower()
        if rl in ["low", "medium"]:
            reasons.append(f"리스크 등급이 {RISK_LEVEL_LABEL.get(rl, rl)} 수준입니다.")
        elif rl == "high":
            reasons.append("리스크는 높지만 기대 순위와 위험 균형에서 상위로 평가됐습니다.")

        news        = item.get("news_brief") or {}
        confidence  = str(news.get("confidence_level") or "").lower()
        if confidence in ["high", "medium"]:
            reasons.append("뉴스 신뢰도가 중간 이상이라 이벤트 기반 신호를 참고할 수 있습니다.")
        return reasons[:3]

    # ═══════════════════════════════════════════════════════════════════════════
    # [수정 4+5] _build_markdown_report
    # — user_level 분기 / 한 줄 요약 우선 / 시장 개요 섹션 / 용어 병기
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_markdown_report(self, report: Dict[str, Any], user_level: str = "beginner") -> str:
        lines  = []
        meta   = report["meta"]
        perf   = report.get("performance") or {}
        selected = meta.get("selected_models") or {}

        rank_model     = selected.get("rank_20d")
        downside_model = selected.get("downside_20d")

        rank_metrics     = (perf.get("rank_20d")    or {}).get("metrics") or {}
        downside_metrics = (perf.get("downside_20d") or {}).get("metrics") or {}

        is_beginner  = user_level == "beginner"
        is_advanced  = user_level == "advanced"
        level_label  = LEVEL_LABELS.get(user_level, user_level)
        uid          = meta.get("user_id") or "전체 평균"

        # ── 헤더 ──
        lines.append(f"# AI 투자 리포트 (as of {meta.get('as_of_date')})")
        lines.append("")
        lines.append(f"- 생성 시각(UTC): {meta.get('created_at_utc')}")
        lines.append(f"- 사용자: {uid}  |  리포트 수준: {level_label}")
        lines.append(f"- 랭킹 모델: `{rank_model}`  /  하락 리스크 모델: `{downside_model}`")
        lines.append("")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [수정 4] 초보 모드: 한 줄 핵심 요약을 가장 먼저 보여줌 (설문 75% 요구)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        mo = report.get("market_overview") or {}
        mo_summary = mo.get("summary", "")

        if is_beginner and mo_summary:
            lines.append("## 📌 오늘의 핵심 한 줄")
            lines.append("")
            lines.append(f"> {mo_summary}")
            lines.append("")

        # ── 1) 요약 ──
        lines.append("## 1) 요약")
        lines.append("")

        if is_beginner:
            # 초보: 숫자 대신 평이한 설명
            ic  = self._safe_float(rank_metrics.get("SpearmanRankIC"))
            auc = self._safe_float(downside_metrics.get("AUC"))
            ic_desc  = "보통" if ic is None else ("좋음" if ic > 0.05 else "보통")
            auc_desc = "보통" if auc is None else ("좋음" if auc > 0.6 else "보통")
            lines.append(f"- **종목 순위 예측 품질**: {ic_desc}")
            lines.append(f"- **하락 종목 감지 능력**: {auc_desc}")
            lines.append(f"- **분석 종목 수**: {self._safe_metric_int(rank_metrics, 'test_rows')}개")
        else:
            lines.append(f"- {self._term('SpearmanRankIC', user_level)}: {self._fmt_num(rank_metrics.get('SpearmanRankIC'), 4)}")
            lines.append(f"- {self._term('NDCG@10', user_level)}: {self._fmt_num(rank_metrics.get('NDCG@10'), 4)}")
            lines.append(f"- {self._term('AUC', user_level)}: {self._fmt_num(downside_metrics.get('AUC'), 4)}")
            lines.append(f"- {self._term('F1', user_level)}: {self._fmt_num(downside_metrics.get('F1'), 4)}")

        lines.append("")
        lines.append("본 리포트는 확률/기대값 기반 예측이며, 실현 수익을 보장하지 않습니다.")
        lines.append("")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # [수정 5] 시장 개요 섹션 (market_overview 활용)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if mo:
            lines.append("## 2) 시장 개요")
            lines.append("")
            lines.append(f"- **시장 국면**: {mo.get('market_phase', 'N/A')}")
            lines.append(f"- **시장 톤**: {mo.get('market_tone', 'N/A')}")
            lines.append(f"- **RSI 상태**: {mo.get('market_rsi_state', 'N/A')}")

            # 수익률 상태
            ret_state = mo.get("market_return_state") or {}
            if ret_state:
                lines.append(
                    f"- **수익률 흐름**: 20일 {ret_state.get('return_20d_state','N/A')} / "
                    f"60일 {ret_state.get('return_60d_state','N/A')}"
                )

            # 집계 지표 (중급 이상만)
            if not is_beginner:
                agg = mo.get("aggregate_metrics") or {}
                if agg:
                    lines.append(f"- 평균 20D 수익률: {self._fmt_pct(agg.get('avg_return_20d'))}")
                    lines.append(f"- 평균 연환산 변동성: {self._fmt_pct(agg.get('avg_ann_volatility_252d'))}")
                    lines.append(f"- 평균 최대낙폭: {self._fmt_pct(agg.get('avg_max_drawdown_252d'))}")

            lines.append("")
            ev_list = (mo.get("evidence") or [])[:3]
            if ev_list:
                lines.append("**시장 근거**")
                for ev in ev_list:
                    lines.append(f"- {ev}")
                lines.append("")

            rn_list = (mo.get("risk_notes") or [])[:2]
            if rn_list:
                lines.append("**시장 리스크 노트**")
                for rn in rn_list:
                    lines.append(f"- {rn}")
                lines.append("")

        # ── 3) 모델 성능 (중급/고급만) ──
        if not is_beginner:
            lines.append("## 3) 모델 성능")
            lines.append("")
            for target, obj in perf.items():
                lines.append(f"### {target}")
                lines.append(f"- 선택 모델: `{obj.get('selected_model')}`")
                metric_obj = obj.get("metrics") or {}
                for k, v in metric_obj.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"- {k}: {self._fmt_num(v, 4)}")
                lines.append("")

        # ── 유니버스 리스크 개요 ──
        sec_num = 4 if not is_beginner else 3
        lines.append(f"## {sec_num}) 유니버스 리스크 개요")
        lines.append("")
        risk_counts = (report.get("risk_universe_summary") or {}).get("risk_level_counts")
        if risk_counts:
            for k, v in risk_counts.items():
                icon = RISK_LEVEL_ICON.get(k, "")
                label = RISK_LEVEL_LABEL.get(k, k)
                lines.append(f"- {icon} {label}: {v}개")
        else:
            lines.append("- risk_score_result.json 집계 정보 없음")
        lines.append("")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Top 5 Opportunities
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        sec_num += 1
        lines.append(f"## {sec_num}) Top 5 추천 종목")
        lines.append("")

        for rank, item in enumerate(report["recommendations_top5"], 1):
            ticker = self._zfill6(item["ticker"])
            lines.append(f"### {rank}. {ticker}")
            lines.append("")

            # ── [수정 4] 초보 모드: 한 줄 해석 먼저 (market interpretation) ──
            market = item.get("market_brief") or {}
            if is_beginner and market.get("interpretation"):
                lines.append(f"> {market['interpretation']}")
                lines.append("")

            # ── 수치 지표 ──
            if is_beginner:
                # [수정 4] 초보: 용어 설명 병기
                lines.append(f"- **{self._term('pred_rank_score_20d', 'beginner')}**: {self._fmt_num(item.get('pred_rank_score_20d'), 4)}")
                lines.append(f"- **{self._term('proba_top_quantile_20d', 'beginner')}**: {self._fmt_pct(item.get('proba_top_quantile_20d'), 2)}")
                lines.append(f"- **{self._term('proba_downside_20d', 'beginner')}**: {self._fmt_pct(item.get('proba_downside_20d'), 2)}")
                if item.get("personalized_score_avg") is not None:
                    lines.append(f"- **나를 위한 추천 점수**: {self._fmt_num(item.get('personalized_score_avg'), 4)}")
            else:
                lines.append(f"- 20일 랭킹 예측 점수: {self._fmt_num(item.get('pred_rank_score_20d'), 4)}")
                lines.append(f"- 20일 랭킹 퍼센타일: {self._fmt_num(item.get('pred_rank_percentile_20d'), 4)}")
                lines.append(f"- Top 20% 진입 확률: {self._fmt_pct(item.get('proba_top_quantile_20d'), 2)}")
                lines.append(f"- 20일 하락 리스크 확률: {self._fmt_pct(item.get('proba_downside_20d'), 2)}")
                lines.append(f"- 개인화 점수 평균: {self._fmt_num(item.get('personalized_score_avg'), 4)}")
                lines.append(f"- 블렌드 점수: {self._fmt_num(item.get('blend_score'), 4)}")
            lines.append("")

            # ── [수정 2] 리스크 요약 ──
            risk = item.get("risk") or {}
            if risk:
                rl     = str(risk.get("risk_level") or "").lower()
                icon   = RISK_LEVEL_ICON.get(rl, "")
                label  = RISK_LEVEL_LABEL.get(rl, rl)
                score  = self._fmt_num(risk.get("overall_risk_score"), 2)
                lines.append("**리스크 요약**")
                lines.append(f"- 위험 등급: {icon} {label} (종합 점수: {score})")
                factors = risk.get("dominant_risk_factors") or []
                if factors:
                    lines.append(f"- 주요 위험 요인: {', '.join(factors)}")
                for ev in risk.get("evidence") or []:
                    lines.append(f"- 근거: {ev}")
                lines.append("")

            # ── [수정 2] 시장/기술 흐름 — interpretation + positive/risk factors ──
            if market:
                lines.append("**시장/기술적 흐름**")
                if market.get("ticker_type"):
                    lines.append(f"- 종목 유형: {market['ticker_type']}")
                if market.get("phase"):
                    lines.append(f"- 시장 국면: {market['phase']}")
                if market.get("rsi_state"):
                    lines.append(f"- RSI 상태: {market['rsi_state']}")
                if market.get("return_20d") is not None:
                    state = market.get("return_20d_state", "")
                    lines.append(f"- 최근 20일 수익률: {self._fmt_pct(market['return_20d'], 2)} ({state})")
                if market.get("return_60d") is not None:
                    state = market.get("return_60d_state", "")
                    lines.append(f"- 최근 60일 수익률: {self._fmt_pct(market['return_60d'], 2)} ({state})")
                if market.get("ann_volatility_252d") is not None:
                    lines.append(f"- 연환산 변동성: {self._fmt_pct(market['ann_volatility_252d'], 2)}")
                if market.get("max_drawdown_252d") is not None:
                    lines.append(f"- 최대 낙폭: {self._fmt_pct(market['max_drawdown_252d'], 2)}")
                lines.append("")

                # 긍정 요인
                pos = market.get("positive_factors") or []
                if pos:
                    lines.append("**긍정 요인**")
                    for p in pos:
                        lines.append(f"- ✅ {p}")
                    lines.append("")

                # 리스크 요인
                rf = market.get("risk_factors") or []
                if rf:
                    lines.append("**주의 요인**")
                    for r in rf:
                        lines.append(f"- ⚠️ {r}")
                    lines.append("")

                # 시장 맥락 (중급 이상)
                if not is_beginner and market.get("market_context"):
                    lines.append(f"- 시장 맥락: {market['market_context']}")
                    lines.append("")

            # ── [수정 3] 뉴스/이슈 흐름 — reasons + headline ──
            news = item.get("news_brief") or {}
            if news:
                lines.append("**뉴스/이슈 흐름**")
                if news.get("news_signal_score") is not None:
                    lines.append(f"- 뉴스 신호 점수: {self._fmt_num(news.get('news_signal_score'), 2)}")
                if news.get("confidence_level"):
                    lines.append(f"- 뉴스 신뢰도: {news['confidence_level']}")
                if news.get("verdict"):
                    lines.append(f"- 뉴스 판정: {news['verdict']}")
                if news.get("n_articles_30d_unique") is not None:
                    lines.append(f"- 최근 30일 기사 수: {int(news['n_articles_30d_unique'])}")

                # [수정 3] 신호 근거 추가
                reasons = news.get("reasons") or []
                if reasons:
                    lines.append("- 뉴스 신호 근거:")
                    for r in reasons:
                        lines.append(f"  - {r}")

                # [수정 3] 대표 기사 헤드라인
                headline = news.get("top_headline")
                if headline and headline.get("title"):
                    lines.append(f"- 대표 기사: [{headline['title']}]({headline.get('url', '#')}) "
                                 f"({headline.get('press', '')} / {headline.get('date', '')})")
                lines.append("")

            # ── AI 선정 이유 ──
            sel_reasons = self._build_selection_reason(item)
            if sel_reasons:
                lines.append("**AI 선정 이유**")
                for r in sel_reasons:
                    lines.append(f"- {r}")
                lines.append("")

        # ── Feature Importance (고급만) ──
        if is_advanced:
            fi = report.get("feature_importance_top10") or {}
            if fi:
                lines.append(f"## {sec_num + 1}) 모델 주요 피처 (고급)")
                lines.append("")
                for target, obj in fi.items():
                    lines.append(f"### {target} — `{obj.get('selected_model')}`")
                    for feat in (obj.get("top_features") or [])[:5]:
                        lines.append(f"- {feat['feature']}: {self._fmt_num(feat['importance'], 2)}")
                    lines.append("")

        # ── Risk Note ──
        lines.append(f"## Risk Note")
        lines.append("")
        lines.append("- 본 결과는 모델 기반 확률/순위 예측이며 수익을 보장하지 않습니다.")
        lines.append("- 시장 레짐 변화, 실적, 정책, 수급 이벤트에 따라 결과가 달라질 수 있습니다.")
        lines.append("- 단일 종목 집중보다는 분산 투자, 손절 기준, 리스크 한도를 함께 고려하세요.")

        return "\n".join(lines)

    # ── 내부 유틸 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_metric_int(d: dict, key: str) -> str:
        v = d.get(key)
        if v is None:
            return "N/A"
        try:
            return str(int(v))
        except Exception:
            return str(v)