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
    version = "1.2-model-infer-report"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
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

        # =========================
        # Exact input filenames
        # =========================
        paths = {
            "metrics": self._find_exact_file(run_dir, "model_infer_metrics.json"),
            "predictions": self._find_exact_file(run_dir, "model_infer_predictions_20d.csv"),
            "feature_importance": self._find_exact_file(run_dir, "model_infer_feature_importance.json"),
            "personalized_scores": self._find_exact_file(run_dir, "model_infer_personalized_scores.csv"),
            "model_infer_manifest": self._find_exact_file(run_dir, "manifest.json", preferred_parent="model_infer"),

            # optional upstream analysis files
            "risk_json": self._find_exact_file(run_dir, "risk_score_result.json"),
            "market_json": self._find_exact_file(run_dir, "market_analysis_result.json"),
            "news_json": self._find_exact_file(run_dir, "news_invest_rag_result.json"),
        }

        required = ["metrics", "predictions", "feature_importance", "personalized_scores"]
        missing = [k for k in required if paths[k] is None]

        if missing:
            raise FileNotFoundError(f"Missing required report inputs: {missing}")

        metrics = self._read_json(paths["metrics"])
        feature_importance = self._read_json(paths["feature_importance"])
        pred = pd.read_csv(paths["predictions"], encoding=self.encoding)
        personalized = pd.read_csv(paths["personalized_scores"], encoding=self.encoding)

        self._normalize_ticker_date(pred)
        self._normalize_ticker_date(personalized)

        selected_models = metrics.get("selected_models") or {}
        rank_model = selected_models.get("rank_20d")
        rank_pct_model = selected_models.get("rank_percentile_20d")
        topq_model = selected_models.get("top_quantile_20d")
        downside_model = selected_models.get("downside_20d")

        pred_rank_col = self._resolve_col(
            pred,
            [
                f"pred_rank_score_20d__{rank_model}" if rank_model else None,
                f"pred_rank_20d__{rank_model}" if rank_model else None,
            ],
            prefixes=[
                "pred_rank_score_20d__",
                "pred_rank_20d__",
                "pred_rank__",
            ],
            fallback_contains=["rank"],
        )

        pred_rank_pct_col = self._resolve_col(
            pred,
            [
                f"pred_rank_percentile_20d__{rank_pct_model}" if rank_pct_model else None,
                f"pred_rank_pct_20d__{rank_pct_model}" if rank_pct_model else None,
            ],
            prefixes=[
                "pred_rank_percentile_20d__",
                "pred_rank_pct_20d__",
            ],
            fallback_contains=["percentile"],
            required=False,
        )

        topq_col = self._resolve_col(
            pred,
            [
                f"proba_top_quantile_20d__{topq_model}" if topq_model else None,
                f"pred_top_quantile_20d__{topq_model}" if topq_model else None,
            ],
            prefixes=[
                "proba_top_quantile_20d__",
                "pred_top_quantile_20d__",
                "proba_top_quantile__",
            ],
            fallback_contains=["top_quantile"],
            required=False,
        )

        downside_col = self._resolve_col(
            pred,
            [
                f"proba_downside_20d__{downside_model}" if downside_model else None,
                f"pred_downside_20d__{downside_model}" if downside_model else None,
            ],
            prefixes=[
                "proba_downside_20d__",
                "pred_downside_20d__",
                "proba_downside__",
            ],
            fallback_contains=["downside"],
        )

        date_col = "date" if "date" in pred.columns else None
        latest_date = pd.to_datetime(pred[date_col]).max() if date_col else None

        if date_col:
            latest_pred = pred[pd.to_datetime(pred[date_col]) == latest_date].copy()
        else:
            latest_pred = pred.copy()

        latest_pred["pred_rank_score_20d"] = pd.to_numeric(
            latest_pred[pred_rank_col],
            errors="coerce",
        )

        if pred_rank_pct_col:
            latest_pred["pred_rank_percentile_20d"] = pd.to_numeric(
                latest_pred[pred_rank_pct_col],
                errors="coerce",
            )
        else:
            latest_pred["pred_rank_percentile_20d"] = None

        if topq_col:
            latest_pred["proba_top_quantile_20d"] = pd.to_numeric(
                latest_pred[topq_col],
                errors="coerce",
            )
        else:
            latest_pred["proba_top_quantile_20d"] = None

        latest_pred["proba_downside_20d"] = pd.to_numeric(
            latest_pred[downside_col],
            errors="coerce",
        )

        # personalized score attach
        score_col = self._resolve_col(
            personalized,
            ["personalized_score"],
            prefixes=["personalized_score"],
            fallback_contains=["personalized"],
            required=False,
        )

        if score_col:
            if "as_of_date" in personalized.columns:
                personalized["as_of_date"] = pd.to_datetime(
                    personalized["as_of_date"],
                    errors="coerce",
                )
                personalized_last = personalized[
                    personalized["as_of_date"] == personalized["as_of_date"].max()
                ].copy()
            else:
                personalized_last = personalized.copy()

            pers_agg = (
                personalized_last.groupby("ticker", as_index=False)[score_col]
                .mean()
                .rename(columns={score_col: "personalized_score_avg"})
            )
            latest_pred = latest_pred.merge(pers_agg, on="ticker", how="left")
        else:
            latest_pred["personalized_score_avg"] = None

        # optional upstream maps
        risk_map = self._load_optional_ticker_map(paths["risk_json"])
        market_map = self._load_optional_ticker_map(paths["market_json"])
        news_map = self._load_optional_ticker_map(paths["news_json"])

        risk_level_counts = None
        if paths["risk_json"] is not None:
            risk_json = self._read_json(paths["risk_json"])
            risk_level_counts = (risk_json.get("universe_summary") or {}).get("risk_level_counts")

        # recommendation score
        latest_pred["blend_score"] = (
            latest_pred["pred_rank_score_20d"].fillna(0)
            + 0.30 * latest_pred["proba_top_quantile_20d"].fillna(0)
            - 0.50 * latest_pred["proba_downside_20d"].fillna(0)
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

        report = {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "inputs": {
                    k: str(v) if v is not None else None
                    for k, v in paths.items()
                },
                "as_of_date": str(pd.to_datetime(latest_date).date()) if pd.notna(latest_date) else None,
                "selected_models": selected_models,
                "prediction_columns": {
                    "rank": pred_rank_col,
                    "rank_percentile": pred_rank_pct_col,
                    "top_quantile": topq_col,
                    "downside": downside_col,
                },
            },
            "performance": self._build_performance_summary(metrics, selected_models),
            "feature_importance_top10": self._build_feature_importance_summary(
                feature_importance,
                selected_models,
            ),
            "risk_universe_summary": {
                "risk_level_counts": risk_level_counts,
            },
            "recommendations_top5": [],
        }

        for _, row in top5.iterrows():
            ticker = row["ticker"]

            item = {
                "ticker": ticker,
                "as_of_date": str(pd.to_datetime(row[date_col]).date())
                if date_col and pd.notna(row.get(date_col))
                else report["meta"]["as_of_date"],
                "pred_rank_score_20d": self._safe_float(row.get("pred_rank_score_20d")),
                "pred_rank_percentile_20d": self._safe_float(row.get("pred_rank_percentile_20d")),
                "proba_top_quantile_20d": self._safe_float(row.get("proba_top_quantile_20d")),
                "proba_downside_20d": self._safe_float(row.get("proba_downside_20d")),
                "personalized_score_avg": self._safe_float(row.get("personalized_score_avg")),
                "blend_score": self._safe_float(row.get("blend_score")),
                "risk": self._get_risk_brief(ticker, risk_map) if risk_map else None,
                "market_brief": self._get_market_brief(ticker, market_map) if market_map else None,
                "news_brief": self._get_news_brief(ticker, news_map) if news_map else None,
            }

            report["recommendations_top5"].append(item)

        self._save_json(report, artifacts.report_json_path)

        md = self._build_markdown_report(report)
        artifacts.report_md_path.write_text(md, encoding="utf-8")

        artifacts.output_files = [
            str(artifacts.report_json_path),
            str(artifacts.report_md_path),
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
            "as_of_date": report["meta"]["as_of_date"],
            "top5_tickers": [
                x["ticker"]
                for x in report["recommendations_top5"]
            ],
        }

        self._save_json(manifest, artifacts.manifest_path)

        return self._make_stage_result(
            status="success",
            message="Report generation completed.",
            metrics={
                "top5_count": len(report["recommendations_top5"]),
                "has_market_json": paths["market_json"] is not None,
                "has_news_json": paths["news_json"] is not None,
                "has_risk_json": paths["risk_json"] is not None,
            },
            outputs={
                "output_dir": str(output_dir),
                "report_summary": str(artifacts.report_json_path),
                "report_md": str(artifacts.report_md_path),
                "manifest": str(artifacts.manifest_path),
            },
        )

    # =========================
    # File helpers
    # =========================

    def _find_exact_file(
        self,
        run_dir: Path,
        filename: str,
        preferred_parent: Optional[str] = None,
    ) -> Optional[Path]:
        matches = [
            p for p in run_dir.rglob(filename)
            if p.is_file() and p.name == filename
        ]

        if preferred_parent:
            preferred = [
                p for p in matches
                if p.parent.name == preferred_parent
            ]
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

    # =========================
    # StageResult helper
    # =========================

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

    # =========================
    # Basic utils
    # =========================

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
            matches = [
                c for c in df.columns
                if str(c).startswith(prefix)
            ]
            if matches:
                return matches[0]

        lowered = {
            c: str(c).lower()
            for c in df.columns
        }

        for keyword in fallback_contains:
            matches = [
                c for c, lc in lowered.items()
                if keyword.lower() in lc
            ]
            if matches:
                return matches[0]

        if required:
            raise ValueError(
                f"Could not resolve required prediction column. "
                f"keywords={fallback_contains}, columns={list(df.columns)}"
            )

        return None

    # =========================
    # Model/metric helpers
    # =========================

    def _build_performance_summary(
        self,
        metrics: Dict[str, Any],
        selected_models: Dict[str, str],
    ) -> Dict[str, Any]:
        out = {}

        for target, model_id in selected_models.items():
            target_metrics = metrics.get(target, {})
            if isinstance(target_metrics, dict):
                out[target] = {
                    "selected_model": model_id,
                    "metrics": target_metrics.get(model_id, {}),
                }

        return out

    def _build_feature_importance_summary(
        self,
        feature_importance: Dict[str, Any],
        selected_models: Dict[str, str],
        k: int = 10,
    ) -> Dict[str, Any]:
        out = {}

        for target, model_id in selected_models.items():
            rows = feature_importance.get(model_id, [])

            parsed = []
            for r in rows:
                feat = r.get("feature")
                imp = self._safe_float(r.get("importance"))

                if feat is None:
                    continue

                parsed.append({
                    "feature": feat,
                    "importance": imp,
                })

            parsed = sorted(
                parsed,
                key=lambda x: abs(x["importance"]) if x["importance"] is not None else -1,
                reverse=True,
            )

            out[target] = {
                "selected_model": model_id,
                "top_features": parsed[:k],
            }

        return out

    # =========================
    # Optional upstream JSON helpers
    # =========================

    def _load_optional_ticker_map(
        self,
        path: Optional[Path],
        key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if path is None or not path.exists():
            return {}

        obj = self._read_json(path)

        if key is not None:
            rows = obj.get(key, [])
        else:
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

    def _get_risk_brief(
        self,
        ticker: str,
        risk_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = risk_map.get(ticker, {})
        scores = row.get("risk_scores") or {}

        return {
            "risk_level": scores.get("risk_level"),
            "overall_risk_score": scores.get("overall_risk_score"),
            "dominant_risk_factors": (row.get("dominant_risk_factors") or [])[:3],
            "evidence": (row.get("evidence") or [])[:3],
        }

    def _get_market_brief(
        self,
        ticker: str,
        market_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = market_map.get(ticker, {})
        compact = row.get("compact_signals") or {}
        price_summary = row.get("price_summary") or {}
        scores = row.get("scores") or {}

        return {
            "ticker_type": row.get("ticker_type"),
            "phase": row.get("phase"),
            "tone": row.get("tone"),
            "return_20d": compact.get("return_20d") or price_summary.get("cum_return_20d"),
            "return_60d": compact.get("return_60d") or price_summary.get("cum_return_60d"),
            "ann_volatility_252d": compact.get("ann_volatility_252d") or price_summary.get("ann_volatility_252d"),
            "max_drawdown_252d": compact.get("max_drawdown_252d") or price_summary.get("max_drawdown_252d"),
            "overall_score": scores.get("overall_score"),
            "key_findings": (row.get("key_findings") or [])[:4],
        }

    def _get_news_brief(
        self,
        ticker: str,
        news_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = news_map.get(ticker, {})
        summary = row.get("news_summary") or {}

        return {
            "news_signal_score": row.get("news_signal_score"),
            "confidence_level": row.get("confidence_level"),
            "verdict": row.get("verdict"),
            "n_articles_30d_unique": summary.get("n_articles_30d_unique"),
            "noise_ratio_0_1": summary.get("noise_ratio_0_1"),
            "high_impact_ratio_0_1": summary.get("high_impact_ratio_0_1"),
        }

    # =========================
    # Markdown
    # =========================

    def _build_selection_reason(self, item: Dict[str, Any]) -> List[str]:
        reasons = []

        rank_score = self._safe_float(item.get("pred_rank_score_20d"))
        topq = self._safe_float(item.get("proba_top_quantile_20d"))
        downside = self._safe_float(item.get("proba_downside_20d"))
        blend = self._safe_float(item.get("blend_score"))

        if rank_score is not None:
            reasons.append("20일 후 종목 순위 예측 점수가 상위권입니다.")

        if topq is not None and downside is not None:
            reasons.append("상위 수익 구간 진입 가능성과 하락 리스크를 함께 고려했습니다.")
        elif downside is not None:
            reasons.append("하락 리스크를 반영해 위험조정 관점에서 선별했습니다.")

        if blend is not None:
            if blend > 0:
                reasons.append("위험 대비 블렌드 점수가 양수입니다.")
            else:
                reasons.append("블렌드 점수는 낮지만 유니버스 내 상대 비교에서 상위권입니다.")

        risk = item.get("risk") or {}
        risk_level = str(risk.get("risk_level") or "").lower()

        if risk_level in ["low", "medium"]:
            reasons.append(f"리스크 등급이 {risk.get('risk_level')} 수준입니다.")
        elif risk_level == "high":
            reasons.append("리스크는 높지만 기대 순위와 위험 균형에서 상위로 평가됐습니다.")

        news = item.get("news_brief") or {}
        confidence = str(news.get("confidence_level") or "").lower()

        if confidence in ["high", "medium"]:
            reasons.append("뉴스 신뢰도가 중간 이상이라 이벤트 기반 신호를 참고할 수 있습니다.")

        return reasons[:3]

    def _build_markdown_report(self, report: Dict[str, Any]) -> str:
        lines = []

        meta = report["meta"]
        perf = report.get("performance") or {}
        selected = meta.get("selected_models") or {}

        rank_model = selected.get("rank_20d")
        downside_model = selected.get("downside_20d")

        rank_metrics = (perf.get("rank_20d") or {}).get("metrics") or {}
        downside_metrics = (perf.get("downside_20d") or {}).get("metrics") or {}

        lines.append(f"# AI 투자 리포트 (as of {meta.get('as_of_date')})")
        lines.append("")
        lines.append(f"- 생성 시각(UTC): {meta.get('created_at_utc')}")
        lines.append(f"- 랭킹 모델: `{rank_model}`")
        lines.append(f"- 하락 리스크 모델: `{downside_model}`")
        lines.append("")
        lines.append("## 1) 요약")
        lines.append("")
        lines.append(f"- 랭킹 성능 Spearman IC: {self._fmt_num(rank_metrics.get('SpearmanRankIC'), 4)}")
        lines.append(f"- 랭킹 NDCG@10: {self._fmt_num(rank_metrics.get('NDCG@10'), 4)}")
        lines.append(f"- 하락 리스크 AUC: {self._fmt_num(downside_metrics.get('AUC'), 4)}")
        lines.append(f"- 하락 리스크 F1: {self._fmt_num(downside_metrics.get('F1'), 4)}")
        lines.append("")
        lines.append("본 리포트는 확률/기대값 기반 예측이며, 실현 수익을 보장하지 않습니다.")
        lines.append("")

        lines.append("## 2) 모델 성능")
        lines.append("")
        for target, obj in perf.items():
            lines.append(f"### {target}")
            lines.append(f"- 선택 모델: `{obj.get('selected_model')}`")

            metric_obj = obj.get("metrics") or {}
            for k, v in metric_obj.items():
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {self._fmt_num(v, 4)}")

            lines.append("")

        lines.append("## 3) 유니버스 리스크 개요")
        lines.append("")
        risk_counts = (report.get("risk_universe_summary") or {}).get("risk_level_counts")

        if risk_counts:
            for k, v in risk_counts.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("- risk_score_result.json 집계 정보 없음")
        lines.append("")

        lines.append("## 4) Top 5 Opportunities")
        lines.append("")

        for rank, item in enumerate(report["recommendations_top5"], 1):
            ticker = self._zfill6(item["ticker"])

            lines.append(f"### {rank}. {ticker}")
            lines.append("")
            lines.append(f"- 20일 랭킹 예측 점수: {self._fmt_num(item.get('pred_rank_score_20d'), 4)}")
            lines.append(f"- 20일 랭킹 퍼센타일 예측: {self._fmt_num(item.get('pred_rank_percentile_20d'), 4)}")
            lines.append(f"- Top 20% 진입 확률: {self._fmt_pct(item.get('proba_top_quantile_20d'), 2)}")
            lines.append(f"- 20일 하락 리스크 확률: {self._fmt_pct(item.get('proba_downside_20d'), 2)}")
            lines.append(f"- 개인화 점수 평균: {self._fmt_num(item.get('personalized_score_avg'), 4)}")
            lines.append(f"- 블렌드 점수: {self._fmt_num(item.get('blend_score'), 4)}")
            lines.append("")

            risk = item.get("risk") or {}
            if risk:
                lines.append("**리스크 요약**")
                lines.append(
                    f"- 위험 등급: {risk.get('risk_level')} "
                    f"(종합 위험 점수: {self._fmt_num(risk.get('overall_risk_score'), 2)})"
                )

                factors = risk.get("dominant_risk_factors") or []
                if factors:
                    lines.append(f"- 주요 위험 요인: {', '.join(factors)}")

                for ev in risk.get("evidence") or []:
                    lines.append(f"- 근거: {ev}")

                lines.append("")

            market = item.get("market_brief") or {}
            if market:
                lines.append("**시장/기술적 흐름**")
                if market.get("phase") is not None:
                    lines.append(f"- 시장 국면: {market.get('phase')}")
                if market.get("tone") is not None:
                    lines.append(f"- 시장 톤: {market.get('tone')}")
                if market.get("return_20d") is not None:
                    lines.append(f"- 최근 20일 수익률: {self._fmt_pct(market.get('return_20d'), 2)}")
                if market.get("ann_volatility_252d") is not None:
                    lines.append(f"- 연환산 변동성: {self._fmt_pct(market.get('ann_volatility_252d'), 2)}")
                if market.get("max_drawdown_252d") is not None:
                    lines.append(f"- 최대 낙폭: {self._fmt_pct(market.get('max_drawdown_252d'), 2)}")

                for finding in market.get("key_findings") or []:
                    lines.append(f"- {finding}")

                lines.append("")

            news = item.get("news_brief") or {}
            if news:
                lines.append("**뉴스/이슈 흐름**")
                if news.get("news_signal_score") is not None:
                    lines.append(f"- 뉴스 신호 점수: {self._fmt_num(news.get('news_signal_score'), 2)}")
                if news.get("confidence_level") is not None:
                    lines.append(f"- 뉴스 신뢰도: {news.get('confidence_level')}")
                if news.get("verdict") is not None:
                    lines.append(f"- 뉴스 판정: {news.get('verdict')}")
                if news.get("n_articles_30d_unique") is not None:
                    lines.append(f"- 최근 30일 기사 수: {int(news.get('n_articles_30d_unique'))}")

                lines.append("")

            reasons = self._build_selection_reason(item)
            if reasons:
                lines.append("**AI 선정 이유**")
                for reason in reasons:
                    lines.append(f"- {reason}")
                lines.append("")

        lines.append("## 5) Risk Note")
        lines.append("")
        lines.append("- 본 결과는 모델 기반 확률/순위 예측이며 수익을 보장하지 않습니다.")
        lines.append("- 시장 레짐 변화, 실적, 정책, 수급 이벤트에 따라 결과가 달라질 수 있습니다.")
        lines.append("- 단일 종목 집중보다는 분산 투자, 손절 기준, 리스크 한도를 함께 고려하세요.")

        return "\n".join(lines)