from __future__ import annotations

import json
import os
import glob
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


@dataclass
class MarketAnalysisInput:
    ohlcv_path: str
    foreign_path: str
    finance_path: str
    output_json_path: str


class MarketAnalysisAgent(BaseAgent):
    """
    Market Analysis Agent v4

    목적:
    - 스코어/랭킹이 아니라 시장과 종목의 현재 국면을 자연어로 해석한다.
    - 예측/추천이 아니라 리포트 생성용 근거를 만든다.

    v4 변경사항 (v3 대비):
    1. _classify_return_state: period 파라미터 추가 (20d / 60d / 252d 기준 분리)
       - 252D: 300%+ → 초급등, 100~300% → 급등, 30~100% → 상승
       - 유니버스 특성(거래대금 상위 급등주 포함)을 반영한 기준값
    2. _classify_market_phase: strong_overheat_ratio 파라미터 추가 및 dead code 제거
       - 평균 RSI 대신 종목별 강한 과열 비율(>=75) 기준으로 국면 판단
    3. _build_market_overview: 호출 순서 조정 및 period 전달 통일
    4. _build_market_stock_context: period 전달 추가
    5. _build_positive_factors: 252D 수익률 구간별 문장 분리
    6. notices: 유니버스 편향 명시 추가
    """

    stage = "market_analysis"

    def __init__(
        self,
        base_dir: str = "",
        output_dir: str = "",
        ohlcv_path: Optional[str] = None,
        foreign_path: Optional[str] = None,
        finance_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        include_raw_signals: bool = False,
    ):
        self.base_dir = base_dir or ""
        self.output_dir = output_dir or ""
        self.include_raw_signals = include_raw_signals

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.paths = {
            "ohlcv_path": self._abspath(ohlcv_path) if ohlcv_path else None,
            "foreign_path": self._abspath(foreign_path) if foreign_path else None,
            "finance_path": self._abspath(finance_path) if finance_path else None,
            "output_json_path": self._abspath(output_json_path) if output_json_path else None,
        }

    # =========================================================
    # Pipeline entry
    # =========================================================

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        try:
            runtime_agent = self if self._is_configured() else self._build_runtime_agent(ctx, ap)
            outputs = runtime_agent.run()
            return StageResult.success(
                stage=self.stage,
                outputs=[outputs["market_analysis_result"]],
            )
        except Exception as e:
            ctx.logger.exception("❌ Market Analysis Agent Failed")
            return StageResult.failed(stage=self.stage, error=str(e))

    def _is_configured(self) -> bool:
        return all(
            [
                self.paths.get("ohlcv_path"),
                self.paths.get("foreign_path"),
                self.paths.get("finance_path"),
                self.paths.get("output_json_path"),
            ]
        )

    def _build_runtime_agent(self, ctx: RunContext, ap: ArtifactPaths) -> "MarketAnalysisAgent":
        prep_apply_dir = Path(ap.stage_dir("prep_apply"))
        stage_dir = Path(ap.stage_dir(self.stage))
        stage_dir.mkdir(parents=True, exist_ok=True)

        if not prep_apply_dir.exists():
            raise FileNotFoundError(f"prep_apply 디렉토리를 찾지 못했습니다: {prep_apply_dir}")

        ohlcv_path = self._pick_latest_matching_file(
            str(prep_apply_dir / "preprocessed__*__price__ohlcv_last365.csv")
        )
        foreign_path = self._pick_latest_matching_file(
            str(prep_apply_dir / "preprocessed__*__price__foreign_snapshot_today.csv")
        )
        finance_path = self._pick_latest_matching_file(
            str(prep_apply_dir / "preprocessed__*__finance__financial_features.csv")
        )

        if not ohlcv_path:
            raise FileNotFoundError(f"ohlcv 파일을 찾지 못했습니다: {prep_apply_dir}")
        if not foreign_path:
            raise FileNotFoundError(f"foreign 파일을 찾지 못했습니다: {prep_apply_dir}")
        if not finance_path:
            raise FileNotFoundError(f"finance 파일을 찾지 못했습니다: {prep_apply_dir}")

        output_json_path = stage_dir / "market_analysis_result.json"

        ctx.logger.info("prep_apply_dir=%s", prep_apply_dir)
        ctx.logger.info("ohlcv_path=%s", ohlcv_path)
        ctx.logger.info("foreign_path=%s", foreign_path)
        ctx.logger.info("finance_path=%s", finance_path)

        return MarketAnalysisAgent(
            base_dir=self.base_dir,
            output_dir=str(stage_dir),
            ohlcv_path=str(ohlcv_path),
            foreign_path=str(foreign_path),
            finance_path=str(finance_path),
            output_json_path=str(output_json_path),
            include_raw_signals=self.include_raw_signals,
        )

    def run(self) -> Dict[str, Any]:
        inp = MarketAnalysisInput(
            ohlcv_path=self.paths["ohlcv_path"],
            foreign_path=self.paths["foreign_path"],
            finance_path=self.paths["finance_path"],
            output_json_path=self.paths["output_json_path"],
        )

        ohlcv = pd.read_csv(inp.ohlcv_path)
        foreign = pd.read_csv(inp.foreign_path)
        fin = pd.read_csv(inp.finance_path)

        ohlcv = self._normalize_ohlcv(ohlcv)
        foreign = self._normalize_foreign(foreign)
        fin = self._normalize_finance(fin)

        as_of_date = str(ohlcv["date"].max().date()) if not ohlcv.empty else None
        as_of_year = int(pd.to_datetime(as_of_date).year) if as_of_date else None

        # look-ahead 방지: 분석 기준일 기준 미래 재무 데이터 제외
        if as_of_year is not None and "year" in fin.columns:
            fin = fin[fin["year"] <= as_of_year].copy()

        fin_latest = self._latest_by_key(fin, key="종목코드", order_col="year")
        foreign_latest = self._latest_by_key(foreign, key="종목코드", order_col="date")

        tickers = sorted(ohlcv["종목코드"].dropna().unique().tolist())
        ticker_rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for ticker in tickers:
            df_t = ohlcv[ohlcv["종목코드"] == ticker].copy().sort_values("date")

            if len(df_t) < 60:
                skipped.append({"ticker": ticker, "reason": "price observations less than 60"})
                continue

            price_summary = self._compute_price_metrics(df_t)
            market = self._extract_market(df_t)
            foreign_summary = self._build_foreign_summary(ticker, foreign_latest)
            fundamentals_summary = self._build_fundamentals_summary(ticker, fin_latest)

            ticker_rows.append(
                {
                    "ticker": ticker,
                    "market": market,
                    "price_summary": price_summary,
                    "foreign_summary": foreign_summary,
                    "fundamentals_summary": fundamentals_summary,
                }
            )

        if not ticker_rows:
            result = self._empty_result(inp, as_of_date, skipped)
            self._save_json(inp.output_json_path, result)
            return {
                "market_analysis_result": inp.output_json_path,
                "market_analysis_result_json": inp.output_json_path,
            }

        market_overview = self._build_market_overview(ticker_rows, as_of_date)
        ticker_analyses = [
            self._build_ticker_analysis(row, market_overview)
            for row in ticker_rows
        ]

        result = {
            "meta": {
                "agent": "MarketAnalysisAgent",
                "version": "v4_interpretive_compact",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": as_of_date,
                "inputs": {
                    "ohlcv_last365_csv": os.path.basename(inp.ohlcv_path),
                    "foreign_snapshot_today_csv": os.path.basename(inp.foreign_path),
                    "financial_features_csv": os.path.basename(inp.finance_path),
                },
                "universe_size": len(ticker_rows),
                "skipped_count": len(skipped),
                "purpose": "전처리된 정량 지표를 바탕으로 시장과 종목의 현재 국면을 해석하고, 리포트 생성을 위한 근거를 제공한다.",
                "notices": [
                    "본 결과는 투자 추천이나 가격 예측이 아니라 해석/근거 생성 목적입니다.",
                    "수익률, 추세, 변동성, RSI, 유동성, 재무지표를 근거로 문장형 분석을 생성합니다.",
                    "재무 데이터는 분석 기준일 기준 미래 연도 데이터가 섞이지 않도록 필터링합니다.",
                    "raw_signals는 기본 output에서 제외하며, include_raw_signals=True일 때만 포함합니다.",
                    # v4 추가
                    "이 유니버스는 거래대금 상위 100 종목으로 구성되어 상승 편향이 구조적으로 존재합니다. ticker_type의 상승형 쏠림과 높은 RSI는 유니버스 특성이며 시장 전체를 대표하지 않습니다.",
                    "252D 수익률과 RSI 해석 시 거래대금 상위 종목 특성(테마주, 급등주 포함)을 감안해야 합니다. 252D 수익률 분류 기준은 이 유니버스에 맞게 조정되었습니다.",
                ],
            },
            "market_overview": market_overview,
            "ticker_analyses": ticker_analyses,
            "diagnostics": {
                "skipped": skipped[:100],
                "foreign_coverage": self._coverage_ratio(
                    [x.get("foreign_summary", {}).get("foreign_net_flow_ratio") for x in ticker_rows]
                ),
                "fundamentals_coverage": self._coverage_ratio(
                    [x.get("fundamentals_summary", {}).get("latest_year") for x in ticker_rows]
                ),
                "ticker_type_distribution": self._distribution(
                    [x.get("ticker_type") for x in ticker_analyses]
                ),
                "rsi_state_distribution": self._distribution(
                    [x.get("rsi_state") for x in ticker_analyses]
                ),
                "return_20d_state_distribution": self._distribution(
                    [x.get("return_state", {}).get("return_20d_state") for x in ticker_analyses]
                ),
                "return_252d_state_distribution": self._distribution(
                    [x.get("return_state", {}).get("return_252d_state") for x in ticker_analyses]
                ),
            },
        }

        self._save_json(inp.output_json_path, result)

        return {
            "market_analysis_result": inp.output_json_path,
            "market_analysis_result_json": inp.output_json_path,
        }

    # =========================================================
    # Normalize
    # =========================================================

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["date", "종목코드", "close"]
        self._require_columns(df, required, "ohlcv")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "종목코드", "close"])
        df["종목코드"] = df["종목코드"].apply(self._zfill6)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        if "daily_return" not in df.columns:
            df = df.sort_values(["종목코드", "date"])
            df["daily_return"] = df.groupby("종목코드")["close"].pct_change()
        else:
            df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")

        return df

    def _normalize_foreign(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df

        self._require_columns(df, ["date", "종목코드"], "foreign")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "종목코드"])
        df["종목코드"] = df["종목코드"].apply(self._zfill6)

        for c in ["frgn_ntby_qty", "foreign_net_flow_ratio", "foreign_ownership_level", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def _normalize_finance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            return df

        self._require_columns(df, ["종목코드"], "finance")
        df["종목코드"] = df["종목코드"].apply(self._zfill6)

        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        else:
            df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

        numeric_cols = [
            "revenue_yoy", "operating_income_yoy", "roe", "roa", "debt_ratio",
            "current_ratio", "asset_turnover", "revenue_cagr_3y",
            "loss_year_count_5y", "profit_volatility_3y",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    # =========================================================
    # Market overview
    # =========================================================

    def _build_market_overview(self, rows: List[Dict[str, Any]], as_of_date: Optional[str]) -> Dict[str, Any]:
        ps_list = [r.get("price_summary", {}) for r in rows]

        regime60 = [ps.get("market_regime_60d") for ps in ps_list if ps.get("market_regime_60d")]
        regime20 = [ps.get("market_regime_20d") for ps in ps_list if ps.get("market_regime_20d")]
        regime_base = regime60 if regime60 else regime20
        regime_dist = pd.Series(regime_base).value_counts().to_dict() if regime_base else {}

        n = len(rows)
        up_ratio = regime_dist.get("Up", 0) / n if n else 0
        down_ratio = regime_dist.get("Down", 0) / n if n else 0
        side_ratio = regime_dist.get("Side", 0) / n if n else 0

        ret20 = self._series_from_ps(ps_list, "cum_return_20d")
        ret60 = self._series_from_ps(ps_list, "cum_return_60d")
        ret252 = self._series_from_ps(ps_list, "cum_return_252d")
        vol252 = self._series_from_ps(ps_list, "ann_volatility_252d")
        mdd252 = self._series_from_ps(ps_list, "max_drawdown_252d")
        rsi14 = self._series_from_ps(ps_list, "rsi_14")

        avg_ret20 = self._safe_float(ret20.mean())
        avg_ret60 = self._safe_float(ret60.mean())
        avg_ret252 = self._safe_float(ret252.mean())
        avg_rsi = self._safe_float(rsi14.mean())

        # [FIX] strong_overheat_ratio를 _classify_market_phase 호출 전에 계산
        strong_overheat_ratio = float((rsi14 >= 75).mean()) if not rsi14.empty else 0.0
        mild_overheat_ratio = float(((rsi14 >= 65) & (rsi14 < 75)).mean()) if not rsi14.empty else 0.0

        # [FIX] strong_overheat_ratio를 파라미터로 전달
        phase = self._classify_market_phase(
            up_ratio=up_ratio,
            down_ratio=down_ratio,
            avg_ret20=avg_ret20,
            avg_ret60=avg_ret60,
            avg_rsi14=avg_rsi,
            avg_vol252=self._safe_float(vol252.mean()),
            strong_overheat_ratio=strong_overheat_ratio,
        )

        # [FIX] market_return_state에 period 전달
        market_return_state = {
            "return_20d_state":  self._classify_return_state(avg_ret20,  "20d"),
            "return_60d_state":  self._classify_return_state(avg_ret60,  "60d"),
            "return_252d_state": self._classify_return_state(avg_ret252, "252d"),
        }
        market_rsi_state = self._classify_rsi_state(avg_rsi)

        evidence = []
        if regime_dist:
            evidence.append(
                f"분석 대상 {n}개 종목 중 상승 레짐 비중은 {up_ratio*100:.1f}%, "
                f"횡보 레짐은 {side_ratio*100:.1f}%, 하락 레짐은 {down_ratio*100:.1f}%입니다."
            )
        if avg_ret20 is not None:
            evidence.append(f"평균 20D 수익률은 {avg_ret20*100:.1f}%로 '{market_return_state['return_20d_state']}' 구간입니다.")
        if avg_ret60 is not None:
            evidence.append(f"평균 60D 수익률은 {avg_ret60*100:.1f}%로 '{market_return_state['return_60d_state']}' 구간입니다.")
        if avg_ret252 is not None:
            evidence.append(f"평균 252D 수익률은 {avg_ret252*100:.1f}%로 '{market_return_state['return_252d_state']}' 구간입니다. (유니버스 상승 편향 반영)")
        if not vol252.empty:
            evidence.append(f"평균 연환산 변동성은 {vol252.mean()*100:.1f}%입니다.")
        if not mdd252.empty:
            evidence.append(f"평균 252D 최대낙폭은 {mdd252.mean()*100:.1f}%입니다.")
        if avg_rsi is not None:
            strong_overheated = int((rsi14 >= 75).sum())
            mild_overheated = int(((rsi14 >= 65) & (rsi14 < 75)).sum())
            depressed = int((rsi14 <= 30).sum())
            evidence.append(
                f"RSI(14) 기준 강한 과열 종목은 {strong_overheated}개({strong_overheat_ratio*100:.1f}%), "
                f"경미한 과열 종목은 {mild_overheated}개, 침체 구간 종목은 {depressed}개입니다."
            )

        risk_notes = self._build_market_risk_notes(ret20, ret60, vol252, mdd252, rsi14, up_ratio)

        return {
            "as_of_date": as_of_date,
            "market_phase": phase["label"],
            "market_tone": phase["tone"],
            "market_rsi_state": market_rsi_state,
            "market_return_state": market_return_state,
            "summary": phase["summary"],
            "evidence": evidence,
            "risk_notes": risk_notes,
            "regime_distribution": {str(k): int(v) for k, v in regime_dist.items()},
            "aggregate_metrics": {
                "avg_return_20d": avg_ret20,
                "avg_return_60d": avg_ret60,
                "avg_return_252d": avg_ret252,
                "avg_ann_volatility_252d": self._safe_float(vol252.mean()),
                "avg_max_drawdown_252d": self._safe_float(mdd252.mean()),
                "avg_rsi_14": avg_rsi,
            },
            # [FIX] RSI 과열 상세 비율 추가
            "rsi_overheat_detail": {
                "strong_overheat_ratio": round(strong_overheat_ratio, 3),
                "mild_overheat_ratio": round(mild_overheat_ratio, 3),
                "total_overheat_ratio": round(strong_overheat_ratio + mild_overheat_ratio, 3),
                "strong_overheat_count": int((rsi14 >= 75).sum()) if not rsi14.empty else 0,
            },
        }

    def _classify_market_phase(
        self,
        up_ratio: float,
        down_ratio: float,
        avg_ret20: Optional[float],
        avg_ret60: Optional[float],
        avg_rsi14: Optional[float],
        avg_vol252: Optional[float],
        strong_overheat_ratio: float = 0.0,  # [FIX] 파라미터 추가
    ) -> Dict[str, str]:
        avg_ret20 = avg_ret20 or 0
        avg_ret60 = avg_ret60 or 0
        avg_rsi14 = avg_rsi14 or 50
        avg_vol252 = avg_vol252 or 0

        if up_ratio >= 0.55 and avg_ret20 > 0 and avg_ret60 > 0:
            # [FIX] 두 번째 dead code 블록 제거 후 strong_overheat_ratio 우선 적용
            if strong_overheat_ratio >= 0.40:
                return {
                    "label": "상승 우위이나 강한 과열 주의 국면",
                    "tone": "positive_cautious",
                    "summary": (
                        f"상승 흐름이 우세하나 종목의 {strong_overheat_ratio*100:.0f}%가 "
                        "RSI 강한 과열 구간(>=75)에 있어 단기 조정 리스크가 높습니다."
                    ),
                }
            if avg_rsi14 >= 65 or avg_vol252 >= 0.6:
                return {
                    "label": "상승 우위이나 과열/변동성 주의 국면",
                    "tone": "positive_cautious",
                    "summary": "다수 종목이 상승 흐름에 있지만, 일부 과열 신호와 높은 변동성이 함께 나타나 리스크 점검이 필요한 상태입니다.",
                }
            return {
                "label": "상승 우위 국면",
                "tone": "positive",
                "summary": "시장 전반에서 상승 종목 비중이 높고 단기·중기 수익률도 양호해 상승 흐름이 우세한 상태입니다.",
            }

        if down_ratio >= 0.35 and avg_ret20 < 0 and avg_ret60 < 0:
            return {
                "label": "하락 압력 우위 국면",
                "tone": "negative",
                "summary": "하락 레짐 비중이 높고 단기·중기 수익률이 부진해 방어적 해석이 필요한 상태입니다.",
            }

        if abs(avg_ret20) < 0.03 and abs(avg_ret60) < 0.05:
            return {
                "label": "방향성 탐색/횡보 국면",
                "tone": "neutral",
                "summary": "시장 전체의 방향성이 강하지 않아 개별 종목별 모멘텀과 리스크를 분리해서 볼 필요가 있습니다.",
            }

        return {
            "label": "혼재 국면",
            "tone": "mixed",
            "summary": "상승·하락·횡보 신호가 함께 나타나고 있어 시장 전체보다 종목별 차별화가 중요한 상태입니다.",
        }

    def _build_market_risk_notes(
        self,
        ret20: pd.Series,
        ret60: pd.Series,
        vol252: pd.Series,
        mdd252: pd.Series,
        rsi14: pd.Series,
        up_ratio: float,
    ) -> List[str]:
        notes: List[str] = []

        if not rsi14.empty:
            strong_overheat_ratio = float((rsi14 >= 75).mean())
            mild_overheat_ratio = float(((rsi14 >= 65) & (rsi14 < 75)).mean())
            if strong_overheat_ratio >= 0.25:
                notes.append(
                    f"강한 RSI 과열 종목 비중이 {strong_overheat_ratio*100:.0f}%로 높아 단기 조정 가능성에 유의해야 합니다."
                )
            elif strong_overheat_ratio + mild_overheat_ratio >= 0.35:
                notes.append("경미한 과열 이상 종목 비중이 높아 상승 흐름 속 속도 부담을 점검해야 합니다.")

        if not vol252.empty and vol252.mean() >= 0.60:
            notes.append("시장 평균 변동성이 높은 편이라 상승 해석과 함께 가격 흔들림 리스크를 병기해야 합니다.")
        if not mdd252.empty and mdd252.mean() <= -0.35:
            notes.append("평균 최대낙폭이 큰 편이므로 최근 상승 흐름만으로 안정적 국면이라고 보기 어렵습니다.")
        if up_ratio >= 0.60 and not ret20.empty and ret20.mean() > 0.10:
            notes.append("상승 종목 비중과 단기 수익률이 동시에 높아 과열성 상승 가능성을 점검해야 합니다.")
        if not ret20.empty and not ret60.empty and ret20.mean() < 0 < ret60.mean():
            notes.append("중기 상승 흐름 속에서 단기 조정이 나타나는 구조일 수 있습니다.")

        return notes

    # =========================================================
    # Ticker analysis
    # =========================================================

    def _build_ticker_analysis(self, row: Dict[str, Any], market_overview: Dict[str, Any]) -> Dict[str, Any]:
        ticker = row.get("ticker")
        market = row.get("market")
        ps = row.get("price_summary", {})
        fs = row.get("foreign_summary", {})
        fu = row.get("fundamentals_summary", {})

        ticker_type = self._classify_ticker_type(ps)
        phase = self._classify_ticker_phase(ps, ticker_type)
        return_state = self._build_return_state(ps)
        rsi_state = self._classify_rsi_state(ps.get("rsi_14"))
        market_context = self._build_market_stock_context(ticker, ps, market_overview, ticker_type)

        positive_factors = self._build_positive_factors(ps, fs, fu, return_state, rsi_state)
        risk_factors = self._build_risk_factors(ps, fs, fu, return_state, rsi_state)
        evidence = self._build_evidence_items(ps, fs, fu, return_state, rsi_state)
        interpretation = self._compose_ticker_interpretation(
            ticker=ticker,
            phase=phase,
            ticker_type=ticker_type,
            market_context=market_context,
            positives=positive_factors,
            risks=risk_factors,
            return_state=return_state,
            rsi_state=rsi_state,
        )

        compact_signals = self._build_compact_signals(ps, fs, fu)

        item = {
            "ticker": ticker,
            "market": market,
            "ticker_type": ticker_type,
            "phase": phase["label"],
            "tone": phase["tone"],
            "market_context": market_context,
            "return_state": return_state,
            "rsi_state": rsi_state,
            "interpretation": interpretation,
            "positive_factors": positive_factors,
            "risk_factors": risk_factors,
            "evidence": evidence,
            "compact_signals": compact_signals,
        }

        if self.include_raw_signals:
            item["raw_signals"] = {
                "price": ps,
                "foreign": fs,
                "fundamentals": fu,
            }

        return item

    def _classify_ticker_type(self, ps: Dict[str, Any]) -> str:
        r20 = ps.get("cum_return_20d") or 0
        r60 = ps.get("cum_return_60d") or 0
        d20 = ps.get("dist_to_ma20") or 0
        d60 = ps.get("dist_to_ma60") or 0
        rsi = ps.get("rsi_14")
        vol = ps.get("ann_volatility_252d") or ps.get("ann_volatility_full") or 0
        reg20 = ps.get("market_regime_20d")
        reg60 = ps.get("market_regime_60d")

        # 1) 약세/조정 먼저
        if r60 <= -0.15 or (r20 < -0.05 and d20 < 0):
            return "조정/약세형"

        if r20 < 0 and r60 > 0 and d20 < 0:
            return "상승 후 단기 조정형"

        # 2) 급등/상승
        if r20 >= 0.50 and (rsi is not None and rsi >= 75):
            return "급등 과열형"

        if r20 >= 0.30 and r60 >= 0.30 and d20 > 0 and d60 > 0:
            return "강한 상승형"

        if r20 >= 0.10 and d20 > 0:
            return "완만한 상승형"

        # 3) 하락
        if r20 <= -0.10 and r60 <= -0.10 and d20 < 0 and d60 < 0:
            return "하락형"

        # 4) 변동성/횡보
        if vol >= 0.80 and abs(r20) >= 0.10:
            return "고변동성 변곡형"

        if reg20 == "Side" or reg60 == "Side":
            return "횡보/방향성 탐색형"

        return "혼재형"

    def _classify_ticker_phase(self, ps: Dict[str, Any], ticker_type: str) -> Dict[str, str]:
        mapping = {
            "급등 과열형":       {"label": "급등 후 강한 과열",       "tone": "positive_cautious"},
            "강한 상승형":       {"label": "상승 추세 지속",           "tone": "positive"},
            "완만한 상승형":     {"label": "완만한 상승 흐름",         "tone": "positive"},
            "상승 후 단기 조정형": {"label": "중기 상승 후 단기 조정", "tone": "mixed"},
            "하락형":           {"label": "하락 추세 우위",            "tone": "negative"},
            "고변동성 변곡형":   {"label": "고변동성 변곡 구간",       "tone": "mixed_cautious"},
            "횡보/방향성 탐색형": {"label": "방향성 탐색",             "tone": "neutral"},
            "혼재형":           {"label": "중립/혼재 흐름",            "tone": "neutral"},
        }
        return mapping.get(ticker_type, mapping["혼재형"])

    def _build_return_state(self, ps: Dict[str, Any]) -> Dict[str, Optional[str]]:
        # [FIX] period 전달
        return {
            "return_20d_state":  self._classify_return_state(ps.get("cum_return_20d"),  "20d"),
            "return_60d_state":  self._classify_return_state(ps.get("cum_return_60d"),  "60d"),
            "return_252d_state": self._classify_return_state(ps.get("cum_return_252d"), "252d"),
        }

    @staticmethod
    def _classify_return_state(value: Optional[float], period: str = "20d") -> Optional[str]:
        """
        [FIX] period별 분류 기준 분리
        - 252d: 거래대금 상위 유니버스 특성을 반영해 기준값 상향
        - 60d / 20d: 기존 대비 소폭 조정
        """
        if value is None or pd.isna(value):
            return None

        if period == "252d":
            # 유니버스 중앙값이 ~168%이므로 기준 상향
            if value >= 3.00: return "초급등"   # 300%+
            if value >= 1.00: return "급등"     # 100~300%
            if value >= 0.30: return "상승"     # 30~100%
            if value > -0.10: return "보합"
            if value > -0.30: return "하락"
            return "급락"

        elif period == "60d":
            if value >= 0.50: return "초급등"
            if value >= 0.20: return "급등"
            if value >= 0.05: return "상승"
            if value > -0.05: return "보합"
            if value > -0.20: return "하락"
            return "급락"

        else:  # 20d 기본
            if value >= 0.50: return "초급등"
            if value >= 0.20: return "급등"
            if value >= 0.05: return "상승"
            if value > -0.05: return "보합"
            if value > -0.15: return "하락"
            return "급락"

    @staticmethod
    def _classify_rsi_state(value: Optional[float]) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        if value >= 75:
            return "강한 과열"
        if value >= 65:
            return "경미한 과열"
        if value <= 30:
            return "침체"
        return "중립"

    def _build_market_stock_context(
        self,
        ticker: str,
        ps: Dict[str, Any],
        market_overview: Dict[str, Any],
        ticker_type: str,
    ) -> str:
        market_phase = market_overview.get("market_phase", "시장 국면 불명")
        market_tone = market_overview.get("market_tone", "neutral")
        r20 = ps.get("cum_return_20d")
        r60 = ps.get("cum_return_60d")
        rsi_state = self._classify_rsi_state(ps.get("rsi_14"))
        # [FIX] period 전달
        ret20_state = self._classify_return_state(r20, "20d")
        ret60_state = self._classify_return_state(r60, "60d")

        if market_tone.startswith("positive"):
            if ticker_type in ["급등 과열형", "강한 상승형"]:
                return (
                    f"시장 전체가 '{market_phase}'인 가운데, {ticker}도 단기 {ret20_state}·중기 {ret60_state} 흐름을 보여 시장 상승세와 같은 방향으로 움직이고 있습니다."
                )
            if ticker_type in ["상승 후 단기 조정형", "횡보/방향성 탐색형", "혼재형"]:
                return (
                    f"시장 전체는 '{market_phase}'이지만, {ticker}는 시장 대비 방향성이 상대적으로 약하거나 혼재되어 있어 개별 리스크 점검이 필요합니다."
                )
            if ticker_type == "하락형":
                return (
                    f"시장 전체는 '{market_phase}'임에도 {ticker}는 하락 신호가 우세해 시장 흐름 대비 상대적으로 약한 종목으로 해석됩니다."
                )

        if market_tone == "negative":
            if ticker_type in ["강한 상승형", "완만한 상승형", "급등 과열형"]:
                return (
                    f"시장 전체는 '{market_phase}'이나, {ticker}는 독립적인 상승 신호를 보여 상대적으로 강한 흐름을 보입니다."
                )
            return (
                f"시장 전체가 '{market_phase}'인 상황에서 {ticker}도 방어적 해석이 필요한 흐름을 보입니다."
            )

        if rsi_state in ["강한 과열", "경미한 과열"]:
            return (
                f"시장 전체가 '{market_phase}'인 가운데, {ticker} 역시 RSI 기준 '{rsi_state}' 구간에 있어 상승 속도 부담을 함께 확인해야 합니다."
            )

        return (
            f"시장 전체가 '{market_phase}'인 가운데, {ticker}는 '{ticker_type}'으로 분류되어 시장 흐름과 개별 신호를 함께 해석할 필요가 있습니다."
        )

    # =========================================================
    # Natural language factors
    # =========================================================

    def _build_positive_factors(
        self,
        ps: Dict[str, Any],
        fs: Dict[str, Any],
        fu: Dict[str, Any],
        return_state: Dict[str, Optional[str]],
        rsi_state: Optional[str],
    ) -> List[str]:
        factors: List[str] = []

        r20 = ps.get("cum_return_20d")
        r60 = ps.get("cum_return_60d")
        r252 = ps.get("cum_return_252d")
        d20 = ps.get("dist_to_ma20")
        d60 = ps.get("dist_to_ma60")
        l20 = ps.get("liquidity_state_20d")
        l60 = ps.get("liquidity_state_60d")
        foreign_own = fs.get("foreign_ownership_level")
        rev = fu.get("revenue_yoy")
        op = fu.get("operating_income_yoy")
        roe = fu.get("roe")

        if r20 is not None and r20 >= 0.30:
            factors.append(f"최근 20D 수익률이 {r20*100:.1f}%로 '급등' 구간에 있습니다.")
        elif r20 is not None and r20 >= 0.10:
            factors.append(f"최근 20D 수익률이 {r20*100:.1f}%로 단기 상승 흐름이 확인됩니다.")

        if r60 is not None and r60 >= 0.30:
            factors.append(f"최근 60D 수익률이 {r60*100:.1f}%로 중기 기준 '급등' 흐름입니다.")
        elif r60 is not None and r60 >= 0.10:
            factors.append(f"최근 60D 수익률이 {r60*100:.1f}%로 중기 흐름이 우호적입니다.")

        # [FIX] 252D 수익률 구간별 문장 분리 (단순 "장기 상승" 한 줄 제거)
        if r252 is not None:
            if r252 >= 3.0:
                factors.append(f"최근 252D 수익률이 {r252*100:.1f}%로 매우 높은 급등 흐름입니다. (유니버스 상위 급등주 특성)")
            elif r252 >= 1.0:
                factors.append(f"최근 252D 수익률이 {r252*100:.1f}%로 장기 기준 급등 구간입니다.")
            elif r252 >= 0.30:
                factors.append(f"최근 252D 수익률이 {r252*100:.1f}%로 장기 상승 흐름이 확인됩니다.")

        if d20 is not None and d20 > 0.03:
            factors.append(f"종가가 MA20보다 {d20*100:.1f}% 높아 단기 추세선 위에 있습니다.")
        if d60 is not None and d60 > 0.05:
            factors.append(f"종가가 MA60보다 {d60*100:.1f}% 높아 중기 추세선 위에 있습니다.")
        if l20 == "High" or l60 == "High":
            factors.append("최근 유동성 상태가 높아 거래 관심이 확대된 모습입니다.")
        if foreign_own is not None and foreign_own >= 30:
            factors.append(f"외국인 보유 비중이 {foreign_own:.1f}%로 비교적 높은 편입니다.")
        if rev is not None and rev > 0.05:
            factors.append(f"최근 재무 기준 매출 YoY가 {rev*100:.1f}%로 성장 흐름이 있습니다.")
        if op is not None and op > 0.05:
            factors.append(f"영업이익 YoY가 {op*100:.1f}%로 이익 개선 신호가 있습니다.")
        if roe is not None and roe > 0.10:
            factors.append(f"ROE가 {roe*100:.1f}%로 수익성 지표가 양호합니다.")

        return factors[:6]

    def _build_risk_factors(
        self,
        ps: Dict[str, Any],
        fs: Dict[str, Any],
        fu: Dict[str, Any],
        return_state: Dict[str, Optional[str]],
        rsi_state: Optional[str],
    ) -> List[str]:
        risks: List[str] = []

        r20 = ps.get("cum_return_20d")
        r60 = ps.get("cum_return_60d")
        d20 = ps.get("dist_to_ma20")
        d60 = ps.get("dist_to_ma60")
        rsi = ps.get("rsi_14")
        vol = ps.get("ann_volatility_252d") or ps.get("ann_volatility_full")
        mdd = ps.get("max_drawdown_252d") or ps.get("max_drawdown_full")
        d52h = ps.get("dist_to_52w_high")
        rev = fu.get("revenue_yoy")
        op = fu.get("operating_income_yoy")
        roe = fu.get("roe")
        debt = fu.get("debt_ratio")
        loss_count = fu.get("loss_year_count_5y")

        if rsi_state == "강한 과열":
            risks.append(f"RSI(14)가 {rsi:.1f}로 강한 과열 구간에 있어 단기 차익실현 가능성에 유의해야 합니다.")
        elif rsi_state == "경미한 과열":
            risks.append(f"RSI(14)가 {rsi:.1f}로 경미한 과열 구간에 있어 상승 속도 부담을 점검해야 합니다.")
        elif rsi_state == "침체":
            risks.append(f"RSI(14)가 {rsi:.1f}로 침체 구간에 있어 단기 수급 회복 여부 확인이 필요합니다.")

        if vol is not None and vol >= 0.80:
            risks.append(f"연환산 변동성이 {vol*100:.1f}%로 매우 높아 가격 흔들림이 클 수 있습니다.")
        elif vol is not None and vol >= 0.60:
            risks.append(f"연환산 변동성이 {vol*100:.1f}%로 높아 변동성 리스크가 있습니다.")

        if mdd is not None and mdd <= -0.45:
            risks.append(f"최근 1년 최대낙폭이 {mdd*100:.1f}%로 과거 낙폭 리스크가 매우 큽니다.")
        elif mdd is not None and mdd <= -0.35:
            risks.append(f"최근 1년 최대낙폭이 {mdd*100:.1f}%로 낙폭 리스크가 있습니다.")

        if d52h is not None and d52h >= -0.03:
            risks.append("52주 고점 부근에 위치해 단기 차익실현 압력이 나올 수 있습니다.")

        if r20 is not None and r20 <= -0.15:
            risks.append(f"최근 20D 수익률이 {r20*100:.1f}%로 단기 급락 구간입니다.")
        elif r20 is not None and r20 <= -0.05:
            risks.append(f"최근 20D 수익률이 {r20*100:.1f}%로 단기 하락 흐름입니다.")

        if r60 is not None and r60 <= -0.15:
            risks.append(f"최근 60D 수익률이 {r60*100:.1f}%로 중기 급락 구간입니다.")
        elif r60 is not None and r60 <= -0.05:
            risks.append(f"최근 60D 수익률이 {r60*100:.1f}%로 중기 하락 흐름입니다.")

        if d20 is not None and d20 < -0.03:
            risks.append(f"종가가 MA20보다 {abs(d20)*100:.1f}% 낮아 단기 추세선 아래에 있습니다.")
        if d60 is not None and d60 < -0.05:
            risks.append(f"종가가 MA60보다 {abs(d60)*100:.1f}% 낮아 중기 추세가 약합니다.")

        if rev is not None and rev < 0:
            risks.append(f"매출 YoY가 {rev*100:.1f}%로 감소했습니다.")
        if op is not None and op < 0:
            risks.append(f"영업이익 YoY가 {op*100:.1f}%로 감소했습니다.")
        if roe is not None and roe < 0:
            risks.append(f"ROE가 {roe*100:.1f}%로 수익성이 부진합니다.")
        if debt is not None and debt >= 2.0:
            risks.append(f"부채비율이 {debt*100:.1f}%로 재무 안정성 부담이 있습니다.")
        if loss_count is not None and loss_count >= 2:
            risks.append(f"최근 5년 적자 연수가 {loss_count:.0f}회로 이익 안정성 점검이 필요합니다.")

        return risks[:6]

    def _build_evidence_items(
        self,
        ps: Dict[str, Any],
        fs: Dict[str, Any],
        fu: Dict[str, Any],
        return_state: Dict[str, Optional[str]],
        rsi_state: Optional[str],
    ) -> List[str]:
        evidence: List[str] = []

        for label, key, state_key in [
            ("20D 수익률",  "cum_return_20d",  "return_20d_state"),
            ("60D 수익률",  "cum_return_60d",  "return_60d_state"),
            ("252D 수익률", "cum_return_252d", "return_252d_state"),
        ]:
            v = ps.get(key)
            state = return_state.get(state_key)
            if v is not None:
                evidence.append(f"{label}: {v*100:.1f}% ({state})" if state else f"{label}: {v*100:.1f}%")

        for label, key in [
            ("MA20 대비 거리",    "dist_to_ma20"),
            ("MA60 대비 거리",    "dist_to_ma60"),
            ("252D 연환산 변동성", "ann_volatility_252d"),
            ("252D MDD",          "max_drawdown_252d"),
            ("52주 고점 대비",    "dist_to_52w_high"),
            ("52주 저점 대비",    "dist_to_52w_low"),
        ]:
            v = ps.get(key)
            if v is not None:
                evidence.append(f"{label}: {v*100:.1f}%")

        rsi = ps.get("rsi_14")
        if rsi is not None:
            evidence.append(f"RSI(14): {rsi:.1f} ({rsi_state})")

        reg20 = ps.get("market_regime_20d")
        reg60 = ps.get("market_regime_60d")
        if reg20:
            evidence.append(f"20D 레짐: {reg20}")
        if reg60:
            evidence.append(f"60D 레짐: {reg60}")

        l20 = ps.get("liquidity_state_20d")
        l60 = ps.get("liquidity_state_60d")
        if l20:
            evidence.append(f"20D 유동성 상태: {l20}")
        if l60:
            evidence.append(f"60D 유동성 상태: {l60}")

        year = fu.get("latest_year")
        if year:
            for label, key in [
                ("매출 YoY",      "revenue_yoy"),
                ("영업이익 YoY",  "operating_income_yoy"),
                ("ROE",           "roe"),
                ("부채비율",      "debt_ratio"),
            ]:
                v = fu.get(key)
                if v is not None:
                    evidence.append(f"{year}년 {label}: {v*100:.1f}%")

        return evidence[:14]

    def _compose_ticker_interpretation(
        self,
        ticker: str,
        phase: Dict[str, str],
        ticker_type: str,
        market_context: str,
        positives: List[str],
        risks: List[str],
        return_state: Dict[str, Optional[str]],
        rsi_state: Optional[str],
    ) -> str:
        label = phase.get("label", "중립/혼재 흐름")
        ret20_state = return_state.get("return_20d_state")
        ret60_state = return_state.get("return_60d_state")

        positive_text = positives[0] if positives else "뚜렷한 긍정 신호는 제한적입니다."
        risk_text = risks[0] if risks else "뚜렷한 위험 신호는 제한적입니다."

        if ticker_type == "급등 과열형":
            return (
                f"{ticker}는 현재 '{label}' 구간입니다. {market_context} "
                f"단기 수익률은 '{ret20_state}', 중기 수익률은 '{ret60_state}'로 분류됩니다. "
                f"{positive_text} 다만 {risk_text} "
                "따라서 리포트에서는 상승 모멘텀보다 과열 부담과 변동성 리스크를 함께 강조하는 것이 적절합니다."
            )
        if ticker_type == "강한 상승형":
            return (
                f"{ticker}는 현재 '{label}' 구간입니다. {market_context} "
                f"{positive_text} "
                f"다만 {risk_text} "
                "전반적으로 추세는 우호적이지만, 단기 급등 여부와 변동성은 함께 점검해야 합니다."
            )
        if ticker_type == "완만한 상승형":
            return (
                f"{ticker}는 현재 '{label}'으로 해석됩니다. {market_context} "
                f"{positive_text} "
                f"위험 요인으로는 {risk_text} "
                "급등형 종목보다는 속도 부담이 낮지만, 시장 과열 국면에서는 방어적 확인이 필요합니다."
            )
        if ticker_type == "조정/약세형":
            return (
                f"{ticker}는 현재 시장 흐름 대비 약한 '조정/약세형'으로 분류됩니다. {market_context} "
                f"단기 수익률은 '{ret20_state}', 중기 수익률은 '{ret60_state}'입니다. "
                f"주요 부담은 {risk_text} "
                "따라서 리포트에서는 반등 가능성보다 조정 원인과 추세 회복 여부를 중심으로 해석하는 것이 적절합니다."
            )
        if ticker_type == "상승 후 단기 조정형":
            return (
                f"{ticker}는 중기 상승 흐름 이후 단기 조정이 나타나는 구간입니다. {market_context} "
                f"{positive_text} 반면 {risk_text} "
                "즉, 기존 상승 추세가 완전히 꺾였다기보다는 단기 피로도가 나타나는지 확인해야 합니다."
            )
        if ticker_type == "하락형":
            return (
                f"{ticker}는 현재 '{label}'으로 하락 압력이 우세합니다. {market_context} "
                f"주요 위험 요인은 {risk_text} "
                "리포트에서는 상승 근거보다 추세 약화, 재무 부담, 변동성 리스크를 우선 설명하는 것이 적절합니다."
            )
        if ticker_type == "고변동성 변곡형":
            return (
                f"{ticker}는 현재 변동성이 큰 변곡 구간에 있습니다. {market_context} "
                f"{positive_text} 그러나 {risk_text} "
                "방향성 판단보다는 변동성 확대와 추세 전환 가능성을 함께 설명하는 것이 적절합니다."
            )
        if ticker_type == "횡보/방향성 탐색형":
            return (
                f"{ticker}는 뚜렷한 방향성보다는 탐색 흐름이 강합니다. {market_context} "
                f"{positive_text} 다만 {risk_text} "
                "따라서 리포트에서는 상승/하락 단정 대신 박스권, 추세 확인, 거래량 변화를 중심으로 해석하는 것이 적절합니다."
            )
        return (
            f"{ticker}는 현재 '{ticker_type}'이며, 국면상 '{label}'로 해석됩니다. {market_context} "
            f"{positive_text} 반면 {risk_text} "
            "여러 지표가 혼재되어 있어 추가 확인이 필요합니다."
        )

    def _build_compact_signals(self, ps: Dict[str, Any], fs: Dict[str, Any], fu: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "return_20d": ps.get("cum_return_20d"),
            "return_60d": ps.get("cum_return_60d"),
            "return_252d": ps.get("cum_return_252d"),
            "dist_to_ma20": ps.get("dist_to_ma20"),
            "dist_to_ma60": ps.get("dist_to_ma60"),
            "rsi_14": ps.get("rsi_14"),
            "ann_volatility_252d": ps.get("ann_volatility_252d"),
            "max_drawdown_252d": ps.get("max_drawdown_252d"),
            "market_regime_20d": ps.get("market_regime_20d"),
            "market_regime_60d": ps.get("market_regime_60d"),
            "liquidity_state_20d": ps.get("liquidity_state_20d"),
            "liquidity_state_60d": ps.get("liquidity_state_60d"),
            "foreign_ownership_level": fs.get("foreign_ownership_level"),
            "latest_finance_year": fu.get("latest_year"),
            "revenue_yoy": fu.get("revenue_yoy"),
            "operating_income_yoy": fu.get("operating_income_yoy"),
            "roe": fu.get("roe"),
            "debt_ratio": fu.get("debt_ratio"),
        }

    # =========================================================
    # Price metrics
    # =========================================================

    def _compute_price_metrics(self, df_one: pd.DataFrame) -> Dict[str, Any]:
        df_one = df_one.sort_values("date")
        close = pd.to_numeric(df_one["close"], errors="coerce")
        ret = pd.to_numeric(df_one["daily_return"], errors="coerce")

        r20  = self._last_n_return(close, 20)
        r60  = self._last_n_return(close, 60)
        r252 = self._last_n_return(close, 252)

        ann_vol_full = None
        if len(ret.dropna()) >= 2:
            ann_vol_full = float(ret.std(ddof=1) * np.sqrt(252))

        ann_vol_20  = self._last_float(df_one, "ret_vol_ann_20d")
        ann_vol_60  = self._last_float(df_one, "ret_vol_ann_60d")
        ann_vol_252 = self._last_float(df_one, "ret_vol_ann_252d")

        mdd_full = self._compute_mdd_from_close(close)
        mdd_60   = self._min_float(df_one, "drawdown_60d")
        mdd_120  = self._min_float(df_one, "drawdown_120d")
        mdd_252  = self._min_float(df_one, "drawdown_252d") or mdd_full

        last_close = self._safe_float(close.iloc[-1])
        ma20 = self._last_float(df_one, "ma20_close")
        ma60 = self._last_float(df_one, "ma60_close")

        dist_to_ma20 = self._last_float(df_one, "price_to_ma20")
        if dist_to_ma20 is None and ma20 not in (None, 0) and last_close is not None:
            dist_to_ma20 = float(last_close / ma20 - 1)

        dist_to_ma60 = self._last_float(df_one, "price_to_ma60")
        if dist_to_ma60 is None and ma60 not in (None, 0) and last_close is not None:
            dist_to_ma60 = float(last_close / ma60 - 1)

        volz20  = self._last_float(df_one, "volume_zscore_20d")
        volz60  = self._last_float(df_one, "volume_zscore_60d")
        volz120 = self._last_float(df_one, "volume_zscore_120d")

        avg_intraday   = self._mean_float(df_one, "intraday_volatility")
        avg_liq_pct_20 = self._mean_float(df_one, "liquidity_percentile_20d")
        avg_liq_pct_60 = self._mean_float(df_one, "liquidity_percentile_60d")

        dist_52w_high = self._last_float(df_one, "dist_to_52w_high")
        dist_52w_low  = self._last_float(df_one, "dist_to_52w_low")

        rsi14 = self._last_float(df_one, "rsi_14")
        rsi28 = self._last_float(df_one, "rsi_28")

        regime20        = self._latest_non_null_str(df_one, "market_regime_20")
        regime20_counts = self._value_counts_str(df_one, "market_regime_20")
        regime60        = self._latest_non_null_str(df_one, "market_regime_60")
        regime60_counts = self._value_counts_str(df_one, "market_regime_60")

        lstate20        = self._latest_non_null_str(df_one, "liquidity_state_20d")
        lstate20_counts = self._value_counts_str(df_one, "liquidity_state_20d")
        lstate60        = self._latest_non_null_str(df_one, "liquidity_state_60d")
        lstate60_counts = self._value_counts_str(df_one, "liquidity_state_60d")

        if regime20 is None:
            regime20, regime20_counts = self._majority_onehot(df_one, "market_regime_20_")
        if regime60 is None:
            regime60, regime60_counts = self._majority_onehot(df_one, "market_regime_60_")
        if lstate20 is None:
            lstate20, lstate20_counts = self._majority_onehot(df_one, "liquidity_state_20d_")
        if lstate60 is None:
            lstate60, lstate60_counts = self._majority_onehot(df_one, "liquidity_state_60d_")

        return {
            "start_date": str(df_one["date"].iloc[0].date()),
            "end_date":   str(df_one["date"].iloc[-1].date()),
            "n_obs":      int(len(df_one)),
            "last_close": last_close,
            "ma20": ma20,
            "ma60": ma60,
            "dist_to_ma20": dist_to_ma20,
            "dist_to_ma60": dist_to_ma60,
            "cum_return_20d":  r20,
            "cum_return_60d":  r60,
            "cum_return_252d": r252,
            "ann_volatility_full":  ann_vol_full,
            "ann_volatility_20d":   ann_vol_20,
            "ann_volatility_60d":   ann_vol_60,
            "ann_volatility_252d":  ann_vol_252,
            "max_drawdown_full":  mdd_full,
            "max_drawdown_60d":   mdd_60,
            "max_drawdown_120d":  mdd_120,
            "max_drawdown_252d":  mdd_252,
            "avg_intraday_volatility":    avg_intraday,
            "latest_volume_zscore_20d":   volz20,
            "latest_volume_zscore_60d":   volz60,
            "latest_volume_zscore_120d":  volz120,
            "avg_liquidity_percentile_20d": avg_liq_pct_20,
            "avg_liquidity_percentile_60d": avg_liq_pct_60,
            "dist_to_52w_high": dist_52w_high,
            "dist_to_52w_low":  dist_52w_low,
            "rsi_14": rsi14,
            "rsi_28": rsi28,
            "market_regime_20d":        regime20,
            "market_regime_20d_counts": regime20_counts,
            "market_regime_60d":        regime60,
            "market_regime_60d_counts": regime60_counts,
            "liquidity_state_20d":        lstate20,
            "liquidity_state_20d_counts": lstate20_counts,
            "liquidity_state_60d":        lstate60,
            "liquidity_state_60d_counts": lstate60_counts,
        }

    # =========================================================
    # Foreign / Fundamentals
    # =========================================================

    def _build_foreign_summary(self, ticker: str, foreign_latest: pd.DataFrame) -> Dict[str, Any]:
        if foreign_latest.empty or ticker not in set(foreign_latest["종목코드"].astype(str)):
            return {}

        r = foreign_latest[foreign_latest["종목코드"] == ticker].iloc[-1].to_dict()
        date_val = r.get("date")
        return {
            "as_of_date": str(pd.to_datetime(date_val).date()) if date_val is not None and not pd.isna(date_val) else None,
            "frgn_ntby_qty": self._safe_float(r.get("frgn_ntby_qty")),
            "foreign_net_flow_ratio": self._safe_float(r.get("foreign_net_flow_ratio")),
            "foreign_ownership_level": self._safe_float(r.get("foreign_ownership_level")),
            "volume": self._safe_float(r.get("volume")),
        }

    def _build_fundamentals_summary(self, ticker: str, fin_latest: pd.DataFrame) -> Dict[str, Any]:
        if fin_latest.empty or ticker not in set(fin_latest["종목코드"].astype(str)):
            return {}

        r = fin_latest[fin_latest["종목코드"] == ticker].iloc[-1].to_dict()
        latest_year = None
        if r.get("year") is not None and not pd.isna(r.get("year")):
            latest_year = int(r.get("year"))

        return {
            "latest_year": latest_year,
            "revenue_yoy": self._safe_float(r.get("revenue_yoy")),
            "operating_income_yoy": self._safe_float(r.get("operating_income_yoy")),
            "roe": self._safe_float(r.get("roe")),
            "roa": self._safe_float(r.get("roa")),
            "debt_ratio": self._safe_float(r.get("debt_ratio")),
            "current_ratio": self._safe_float(r.get("current_ratio")),
            "asset_turnover": self._safe_float(r.get("asset_turnover")),
            "revenue_cagr_3y": self._safe_float(r.get("revenue_cagr_3y")),
            "loss_year_count_5y": self._safe_float(r.get("loss_year_count_5y")),
            "profit_volatility_3y": self._safe_float(r.get("profit_volatility_3y")),
        }

    # =========================================================
    # Utility helpers
    # =========================================================

    @staticmethod
    def _pick_latest_matching_file(pattern: str) -> Optional[str]:
        matches = glob.glob(pattern)
        if not matches:
            return None
        return sorted(matches, key=os.path.getmtime, reverse=True)[0]

    def _abspath(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        if self.base_dir:
            return os.path.abspath(os.path.join(self.base_dir, path))
        return os.path.abspath(path)

    @staticmethod
    def _save_json(path: str, data: Dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _zfill6(x: Any) -> str:
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s.zfill(6)

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None or pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 데이터에 필수 컬럼이 없습니다: {missing}")

    @staticmethod
    def _latest_by_key(df: pd.DataFrame, key: str, order_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        if key not in df.columns or order_col not in df.columns:
            return pd.DataFrame()
        return (
            df.dropna(subset=[key])
            .sort_values([key, order_col])
            .groupby(key, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    @staticmethod
    def _last_n_return(close: pd.Series, n: int) -> Optional[float]:
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < (n + 1):
            return None
        base = close.iloc[-(n + 1)]
        if base == 0 or pd.isna(base):
            return None
        return float(close.iloc[-1] / base - 1)

    @staticmethod
    def _compute_mdd_from_close(close: pd.Series) -> Optional[float]:
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 2:
            return None
        running_max = close.cummax()
        dd = close / running_max - 1
        return float(dd.min())

    def _last_float(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if col not in df.columns or df.empty:
            return None
        return self._safe_float(pd.to_numeric(df[col], errors="coerce").iloc[-1])

    def _min_float(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if col not in df.columns or df.empty:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return self._safe_float(s.min())

    def _mean_float(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if col not in df.columns or df.empty:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return self._safe_float(s.mean())

    def _latest_non_null_str(self, df: pd.DataFrame, col: str) -> Optional[str]:
        if col not in df.columns:
            return None
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            return None
        return s.iloc[-1]

    def _value_counts_str(self, df: pd.DataFrame, col: str) -> Dict[str, int]:
        if col not in df.columns:
            return {}
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            return {}
        return {str(k): int(v) for k, v in s.value_counts().to_dict().items()}

    @staticmethod
    def _majority_onehot(
        df_one: pd.DataFrame,
        prefix: str,
        exclude_suffixes: Tuple[str, ...] = ("nan", "__MISSING__"),
    ) -> Tuple[Optional[str], Dict[str, int]]:
        cols = [c for c in df_one.columns if c.startswith(prefix)]
        cols = [c for c in cols if all(not c.endswith(suf) for suf in exclude_suffixes)]
        if not cols:
            return None, {}

        counts: Dict[str, int] = {}
        for c in cols:
            key = c.replace(prefix, "")
            counts[key] = int(df_one[c].astype(bool).sum())

        majority = max(counts, key=counts.get) if counts else None
        return majority, counts

    @staticmethod
    def _extract_market(df_t: pd.DataFrame) -> Optional[str]:
        if "시장" in df_t.columns:
            s = df_t["시장"].dropna().astype(str).str.strip()
            if not s.empty:
                val = s.iloc[-1]
                upper = val.upper()
                return upper if upper in ("KOSPI", "KOSDAQ") else val

        for col in ["시장_KOSPI", "market_KOSPI"]:
            if col in df_t.columns and df_t[col].astype(bool).any():
                return "KOSPI"
        for col in ["시장_KOSDAQ", "market_KOSDAQ"]:
            if col in df_t.columns and df_t[col].astype(bool).any():
                return "KOSDAQ"

        return None

    @staticmethod
    def _series_from_ps(ps_list: List[Dict[str, Any]], key: str) -> pd.Series:
        vals = [ps.get(key) for ps in ps_list]
        return pd.to_numeric(pd.Series(vals), errors="coerce").dropna()

    @staticmethod
    def _coverage_ratio(values: List[Any]) -> float:
        if not values:
            return 0.0
        valid = [v for v in values if v is not None and not pd.isna(v)]
        return float(len(valid) / len(values))

    @staticmethod
    def _distribution(values: List[Any]) -> Dict[str, int]:
        clean = [str(v) for v in values if v is not None and not pd.isna(v)]
        if not clean:
            return {}
        return {str(k): int(v) for k, v in pd.Series(clean).value_counts().to_dict().items()}

    def _empty_result(self, inp: MarketAnalysisInput, as_of_date: Optional[str], skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": "MarketAnalysisAgent",
                "version": "v4_interpretive_compact",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": as_of_date,
                "inputs": {
                    "ohlcv_last365_csv": os.path.basename(inp.ohlcv_path),
                    "foreign_snapshot_today_csv": os.path.basename(inp.foreign_path),
                    "financial_features_csv": os.path.basename(inp.finance_path),
                },
                "universe_size": 0,
                "skipped_count": len(skipped),
                "purpose": "시장/종목 국면 해석",
            },
            "market_overview": {
                "market_phase": "분석 불가",
                "summary": "유효한 종목 데이터가 없어 시장 국면을 해석할 수 없습니다.",
                "evidence": [],
                "risk_notes": [],
            },
            "ticker_analyses": [],
            "diagnostics": {"skipped": skipped[:100]},
        }