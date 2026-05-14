from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.context import RunContext
from core.artifacts import ArtifactPaths
from core.result import StageResult


class RiskScoreAgent:
    stage = "risk_score"

    def __init__(self):
        self.output_dir: Optional[Path] = None
        self.market_json_path: Optional[Path] = None
        self.news_json_path: Optional[Path] = None
        self.out_json_path: Optional[Path] = None

    # ----------------------------
    # Utils
    # ----------------------------
    @staticmethod
    def clamp01(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return 0.0
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    @classmethod
    def minmax01(cls, x, lo, hi):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return 0.0
            if hi == lo:
                return 0.0
            return cls.clamp01((float(x) - lo) / (hi - lo))
        except Exception:
            return 0.0

    @staticmethod
    def safe_float(x, default=None):
        try:
            if x is None:
                return default
            v = float(x)
            if isinstance(v, float) and math.isnan(v):
                return default
            return v
        except Exception:
            return default

    @staticmethod
    def label_risk(label: str, mapping: dict, default: float = 50.0) -> float:
        return float(mapping.get(label or "", default))

    @staticmethod
    def risk_level_from_score(score_0_100: float) -> str:
        if score_0_100 < 25:
            return "low"
        if score_0_100 < 40:
            return "medium"
        if score_0_100 < 60:
            return "high"
        return "critical"

    # ----------------------------
    # IO
    # ----------------------------
    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------
    # Risk components
    # ----------------------------
    def compute_price_overheat_risk(self, m: dict) -> float:
        compact = m.get("compact_signals") or {}

        rsi = self.safe_float(compact.get("rsi_14"), 50.0)
        ret20 = self.safe_float(compact.get("return_20d"), 0.0)
        ret60 = self.safe_float(compact.get("return_60d"), 0.0)
        vol = self.safe_float(compact.get("ann_volatility_252d"), 0.0)

        rsi_risk = self.minmax01(rsi, 55, 85) * 100
        ret20_risk = self.minmax01(ret20, 0.05, 0.60) * 100
        ret60_risk = self.minmax01(ret60, 0.10, 1.00) * 100
        vol_risk = self.minmax01(vol, 0.35, 1.00) * 100

        ticker_type_risk = self.label_risk(
            m.get("ticker_type"),
            {
                "급등 과열형": 90,
                "고변동성 변곡형": 85,
                "강한 상승형": 70,
                "조정/약세형": 70,
                "혼재형": 55,
                "횡보/방향성 탐색형": 50,
                "완만한 상승형": 40,
            },
            55,
        )

        return round(
            0.30 * rsi_risk
            + 0.25 * ret20_risk
            + 0.15 * ret60_risk
            + 0.20 * vol_risk
            + 0.10 * ticker_type_risk,
            4,
        )

    def compute_downside_risk(self, m: dict) -> float:
        compact = m.get("compact_signals") or {}

        mdd = abs(self.safe_float(compact.get("max_drawdown_252d"), 0.0))
        ret20 = self.safe_float(compact.get("return_20d"), 0.0)
        ret60 = self.safe_float(compact.get("return_60d"), 0.0)
        dist_ma20 = self.safe_float(compact.get("dist_to_ma20"), 0.0)
        dist_ma60 = self.safe_float(compact.get("dist_to_ma60"), 0.0)

        regime20 = compact.get("market_regime_20d")
        regime60 = compact.get("market_regime_60d")

        mdd_risk = self.minmax01(mdd, 0.20, 0.60) * 100
        ret20_down = self.minmax01(-ret20, 0.03, 0.25) * 100
        ret60_down = self.minmax01(-ret60, 0.05, 0.30) * 100

        ma_break = max(
            self.minmax01(-dist_ma20, 0.02, 0.15),
            self.minmax01(-dist_ma60, 0.03, 0.20),
        ) * 100

        regime_risk = 0
        if regime20 == "Down":
            regime_risk += 50
        if regime60 == "Down":
            regime_risk += 50

        return round(
            0.35 * mdd_risk
            + 0.20 * ret20_down
            + 0.20 * ret60_down
            + 0.15 * ma_break
            + 0.10 * regime_risk,
            4,
        )

    def compute_financial_risk(self, m: dict) -> float:
        compact = m.get("compact_signals") or {}

        revenue_yoy = self.safe_float(compact.get("revenue_yoy"), None)
        op_yoy = self.safe_float(compact.get("operating_income_yoy"), None)
        roe = self.safe_float(compact.get("roe"), None)
        debt = self.safe_float(compact.get("debt_ratio"), None)

        missing_count = sum(v is None for v in [revenue_yoy, op_yoy, roe, debt])
        missing_risk = missing_count / 4 * 100

        revenue_risk = 50 if revenue_yoy is None else self.minmax01(-revenue_yoy, 0.00, 0.30) * 100
        op_risk = 50 if op_yoy is None else self.minmax01(-op_yoy, 0.00, 0.50) * 100
        roe_risk = 50 if roe is None else self.minmax01(-roe, 0.00, 0.20) * 100
        debt_risk = 50 if debt is None else self.minmax01(debt, 1.5, 5.0) * 100

        return round(
            0.25 * revenue_risk
            + 0.30 * op_risk
            + 0.20 * roe_risk
            + 0.15 * debt_risk
            + 0.10 * missing_risk,
            4,
        )

    def compute_news_event_risk(self, n: dict) -> float:
        news_summary = n.get("news_summary") or {}

        score = self.safe_float(n.get("news_signal_score"), 0.0)
        noise = self.safe_float(news_summary.get("noise_ratio_0_1"), 0.0)
        high_impact = self.safe_float(news_summary.get("high_impact_ratio_0_1"), 0.0)
        n30 = self.safe_float(news_summary.get("n_articles_30d_unique"), 0.0)

        signal_risk = self.minmax01(score, 45, 80) * 100
        impact_risk = high_impact * 100
        noise_risk = noise * 100
        volume_risk = self.minmax01(n30, 3, 25) * 100

        verdict_risk = self.label_risk(
            n.get("verdict"),
            {
                "high_signal": 85,
                "medium_signal": 60,
                "low_signal": 25,
            },
            30,
        )

        return round(
            0.35 * signal_risk
            + 0.25 * impact_risk
            + 0.15 * volume_risk
            + 0.15 * verdict_risk
            + 0.10 * noise_risk,
            4,
        )

    def compute_uncertainty_risk(self, m: dict, n: dict) -> float:
        compact = m.get("compact_signals") or {}
        news_summary = n.get("news_summary") or {}

        finance_keys = ["revenue_yoy", "operating_income_yoy", "roe", "debt_ratio"]
        finance_missing = sum(compact.get(k) is None for k in finance_keys) / len(finance_keys) * 100

        n30 = self.safe_float(news_summary.get("n_articles_30d_unique"), 0.0)
        news_coverage_risk = (1.0 - self.minmax01(n30, 3, 15)) * 100

        confidence_risk = self.label_risk(
            n.get("confidence_level"),
            {
                "high": 20,
                "medium": 50,
                "low": 85,
            },
            80,
        )

        return round(
            0.45 * finance_missing
            + 0.35 * news_coverage_risk
            + 0.20 * confidence_risk,
            4,
        )

    # ----------------------------
    # Main
    # ----------------------------
    def run(self) -> dict[str, Any]:
        market = self._load_json(self.market_json_path)
        news = self._load_json(self.news_json_path)

        market_rows = market.get("ticker_analyses") or []
        news_rows = news.get("tickers") or []

        market_by_ticker = {
            str(r.get("ticker")).zfill(6): r
            for r in market_rows
            if r.get("ticker")
        }

        news_by_ticker = {
            str(r.get("ticker")).zfill(6): r
            for r in news_rows
            if r.get("ticker")
        }

        tickers_union = sorted(set(market_by_ticker.keys()) | set(news_by_ticker.keys()))

        out_rows = [
            self.compute_risk_for_ticker(
                ticker=t,
                market_by_ticker=market_by_ticker,
                news_by_ticker=news_by_ticker,
                market_root=market,
            )
            for t in tickers_union
        ]

        universe_summary = {
            "n_tickers": len(out_rows),
            "risk_level_counts": {
                "low": sum(1 for r in out_rows if r["risk_scores"]["risk_level"] == "low"),
                "medium": sum(1 for r in out_rows if r["risk_scores"]["risk_level"] == "medium"),
                "high": sum(1 for r in out_rows if r["risk_scores"]["risk_level"] == "high"),
                "critical": sum(1 for r in out_rows if r["risk_scores"]["risk_level"] == "critical"),
            },
            "avg_overall_risk_score": round(
                sum(r["risk_scores"]["overall_risk_score"] for r in out_rows) / len(out_rows),
                4,
            ) if out_rows else 0.0,
        }

        out = {
            "meta": {
                "agent": "RiskScoreAgent",
                "version": "v2_market_news_structure_aligned",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {
                    "market": str(self.market_json_path),
                    "news": str(self.news_json_path),
                },
                "score_formula": {
                    "price_overheat_risk": 0.45,
                    "downside_risk": 0.25,
                    "financial_risk": 0.15,
                    "news_event_risk": 0.10,
                    "uncertainty_risk": 0.05,
                },
                "note": "market_analysis_result의 실제 구조와 news_invest_result의 실제 구조만 사용해 리스크 점수를 계산함.",
            },
            "universe_summary": universe_summary,
            "tickers": out_rows,
        }

        with open(self.out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return out

    def compute_risk_for_ticker(
        self,
        ticker: str,
        market_by_ticker: dict[str, dict],
        news_by_ticker: dict[str, dict],
        market_root: dict,
    ) -> dict[str, Any]:
        m = market_by_ticker.get(ticker, {}) or {}
        n = news_by_ticker.get(ticker, {}) or {}

        compact = m.get("compact_signals") or {}
        news_summary = n.get("news_summary") or {}

        price_overheat_risk = self.compute_price_overheat_risk(m)
        downside_risk = self.compute_downside_risk(m)
        financial_risk = self.compute_financial_risk(m)
        news_event_risk = self.compute_news_event_risk(n)
        uncertainty_risk = self.compute_uncertainty_risk(m, n)

        overall_risk_score = round(
                0.45 * price_overheat_risk
            + 0.25 * downside_risk
            + 0.15 * financial_risk
            + 0.10 * news_event_risk
            + 0.05 * uncertainty_risk,
            4
        )

        risk_level = self.risk_level_from_score(overall_risk_score)

        dominant_candidates = [
            ("price_overheat_risk", price_overheat_risk),
            ("downside_risk", downside_risk),
            ("financial_risk", financial_risk),
            ("news_event_risk", news_event_risk),
            ("uncertainty_risk", uncertainty_risk),
        ]

        dominant_risk_factors = [
            name for name, _ in sorted(
                dominant_candidates,
                key=lambda x: x[1],
                reverse=True,
            )[:3]
        ]

        evidence = []

        for rf in m.get("risk_factors", [])[:5]:
            evidence.append(rf)

        if n.get("verdict") in ["high_signal", "medium_signal"]:
            evidence.extend((n.get("reasons") or [])[:3])

        return {
            "ticker": ticker,
            "as_of_date": (
                n.get("as_of_date")
                or market_root.get("market_overview", {}).get("as_of_date")
                or market_root.get("meta", {}).get("as_of_date")
            ),
            "source_fields_used": {
                "market": [
                    "ticker_type",
                    "phase",
                    "tone",
                    "rsi_state",
                    "return_state",
                    "risk_factors",
                    "compact_signals",
                ],
                "news": [
                    "news_signal_score",
                    "verdict",
                    "confidence_level",
                    "news_summary",
                    "reasons",
                ],
            },
            "inputs_snapshot": {
                "ticker_type": m.get("ticker_type"),
                "phase": m.get("phase"),
                "tone": m.get("tone"),
                "rsi_state": m.get("rsi_state"),
                "return_state": m.get("return_state"),
                "return_20d": compact.get("return_20d"),
                "return_60d": compact.get("return_60d"),
                "return_252d": compact.get("return_252d"),
                "dist_to_ma20": compact.get("dist_to_ma20"),
                "dist_to_ma60": compact.get("dist_to_ma60"),
                "rsi_14": compact.get("rsi_14"),
                "ann_volatility_252d": compact.get("ann_volatility_252d"),
                "max_drawdown_252d": compact.get("max_drawdown_252d"),
                "market_regime_20d": compact.get("market_regime_20d"),
                "market_regime_60d": compact.get("market_regime_60d"),
                "revenue_yoy": compact.get("revenue_yoy"),
                "operating_income_yoy": compact.get("operating_income_yoy"),
                "roe": compact.get("roe"),
                "debt_ratio": compact.get("debt_ratio"),
                "news_signal_score": n.get("news_signal_score"),
                "news_verdict": n.get("verdict"),
                "confidence_level": n.get("confidence_level"),
                "n_articles_30d_unique": news_summary.get("n_articles_30d_unique"),
                "noise_ratio_0_1": news_summary.get("noise_ratio_0_1"),
                "high_impact_ratio_0_1": news_summary.get("high_impact_ratio_0_1"),
                "article_quality_score_0_1": news_summary.get("article_quality_score_0_1"),
            },
            "risk_scores": {
                "price_overheat_risk": price_overheat_risk,
                "downside_risk": downside_risk,
                "financial_risk": financial_risk,
                "news_event_risk": news_event_risk,
                "uncertainty_risk": uncertainty_risk,
                "overall_risk_score": overall_risk_score,
                "risk_level": risk_level,
            },
            "dominant_risk_factors": dominant_risk_factors,
            "evidence": evidence[:8],
        }

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        try:
            market_dir = Path(ap.stage_dir("market_analysis"))
            news_dir = Path(ap.stage_dir("news_invest"))
            self.output_dir = Path(ap.stage_dir(self.stage))
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.market_json_path = market_dir / "market_analysis_result.json"
            self.news_json_path = news_dir / "news_invest_result.json"
            self.out_json_path = self.output_dir / "risk_score_result.json"

            for required in [self.market_json_path, self.news_json_path]:
                if not required.exists():
                    raise FileNotFoundError(f"Required input file not found: {required}")

            result = self.run()

            ctx.logger.info(f"[risk_score] Saved: {self.out_json_path}")
            ctx.logger.info(
                f"[risk_score] Universe: {result.get('universe_summary', {})}"
            )

            return StageResult.success(
                stage=self.stage,
                outputs=[str(self.out_json_path)],
            )

        except Exception as e:
            ctx.logger.exception("❌ Risk Score Failed")
            return StageResult.failed(
                stage=self.stage,
                error=str(e),
            )