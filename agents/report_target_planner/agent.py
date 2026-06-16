from __future__ import annotations

import inspect
import json
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
class ReportTargetPlannerArtifacts:
    run_dir: Path
    output_dir: Path
    report_target_spec_path: Path
    report_feature_contract_path: Path
    candidate_scoring_plan_path: Path
    evidence_builder_contract_path: Path
    report_output_contract_path: Path
    manifest_path: Path
    output_files: List[str]


class ReportTargetPlannerAgent(BaseAgent):
    """
    Report Target Planner Agent

    목적:
    - 기존 20일 뒤 수익률 예측 타겟(rank_20d)을 정의하지 않는다.
    - 최종 리포트의 목적을 "시장 흐름 기반 유망/관심 종목 후보 리포트"로 정의한다.
    - Market Flow Agent, Candidate Scoring Agent, Evidence Builder Agent,
      Report Generative Agent가 따라야 할 출력 필드와 기준을 구조화한다.
    """

    stage = "report_target_planner"
    version = "1.0-market-flow-candidate-report"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = ReportTargetPlannerArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            report_target_spec_path=output_dir / "report_target_spec.json",
            report_feature_contract_path=output_dir / "report_feature_contract.json",
            candidate_scoring_plan_path=output_dir / "candidate_scoring_plan.json",
            evidence_builder_contract_path=output_dir / "evidence_builder_contract.json",
            report_output_contract_path=output_dir / "report_output_contract.json",
            manifest_path=output_dir / "manifest.json",
            output_files=[],
        )

        ctx.logger.info("[ReportTargetPlannerAgent] Starting report target planning.")

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

        # RAG 결과가 있으면 우선 사용하고, 없으면 일반 news_invest_result 사용
        news_json_path = paths["news_json_rag"] or paths["news_json"]
        news_json = self._safe_read_json(news_json_path, ctx)

        risk_json = self._safe_read_json(paths["risk_json"], ctx)

        as_of_date = self._infer_as_of_date(ohlcv)

        report_target_spec = self._build_report_target_spec(
            paths=paths,
            as_of_date=as_of_date,
        )

        report_feature_contract = self._build_report_feature_contract(
            ohlcv=ohlcv,
            news_feat=news_feat,
            foreign=foreign,
            finance=finance,
            user=user,
            market_json=market_json,
            news_json=news_json,
            risk_json=risk_json,
        )

        candidate_scoring_plan = self._build_candidate_scoring_plan()
        evidence_builder_contract = self._build_evidence_builder_contract()
        report_output_contract = self._build_report_output_contract()

        self._save_json(report_target_spec, artifacts.report_target_spec_path)
        self._save_json(report_feature_contract, artifacts.report_feature_contract_path)
        self._save_json(candidate_scoring_plan, artifacts.candidate_scoring_plan_path)
        self._save_json(evidence_builder_contract, artifacts.evidence_builder_contract_path)
        self._save_json(report_output_contract, artifacts.report_output_contract_path)

        artifacts.output_files = [
            str(artifacts.report_target_spec_path),
            str(artifacts.report_feature_contract_path),
            str(artifacts.candidate_scoring_plan_path),
            str(artifacts.evidence_builder_contract_path),
            str(artifacts.report_output_contract_path),
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
            "report_type": "market_flow_candidate_report",
            "main_target": "market_flow_candidate_score",
            "prediction_horizon": None,
            "ranking_scope": "as_of_date_cross_section",
            "as_of_date": as_of_date,
            "notes": [
                "This stage does not define return_20d or rank_20d labels.",
                "This stage defines the report objective, candidate ranking target, scoring components, and downstream output contracts.",
                "The final report should present candidate stocks, not direct buy or sell recommendations.",
            ],
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
            "n_price_features": len(report_feature_contract["feature_groups"]["price_features"]),
            "n_news_features": len(report_feature_contract["feature_groups"]["news_features"]),
            "n_foreign_features": len(report_feature_contract["feature_groups"]["foreign_features"]),
            "n_finance_features": len(report_feature_contract["feature_groups"]["finance_features"]),
            "n_user_features": len(report_feature_contract["feature_groups"]["user_features"]),
            "n_required_report_outputs": len(report_target_spec["required_outputs"]),
        }

        outputs = {
            "output_dir": str(output_dir),
            "report_target_spec": str(artifacts.report_target_spec_path),
            "report_feature_contract": str(artifacts.report_feature_contract_path),
            "candidate_scoring_plan": str(artifacts.candidate_scoring_plan_path),
            "evidence_builder_contract": str(artifacts.evidence_builder_contract_path),
            "report_output_contract": str(artifacts.report_output_contract_path),
            "manifest": str(artifacts.manifest_path),
        }

        ctx.logger.info("[ReportTargetPlannerAgent] Completed report target planning.")

        return self._make_stage_result(
            status="success",
            message="Report target planning completed.",
            metrics=metrics,
            outputs=outputs,
        )

    # =========================================================
    # Main builders
    # =========================================================

    def _build_report_target_spec(
        self,
        paths: Dict[str, Optional[Path]],
        as_of_date: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "inputs": {
                    k: str(v) if v is not None else None
                    for k, v in paths.items()
                },
            },
            "report_type": "market_flow_candidate_report",
            "report_title": "시장 흐름 기반 유망 종목 후보 리포트",
            "main_objective": (
                "오늘의 시장 흐름, 뉴스, 시세, 수급, 재무, 리스크 데이터를 기반으로 "
                "초보 투자자가 참고할 수 있는 유망/관심 종목 후보를 선정하고, "
                "각 후보의 근거를 쉬운 문장으로 설명한다."
            ),
            "as_of_date": as_of_date,
            "ranking_target": "market_flow_candidate_score",
            "prediction_horizon": None,
            "ranking_scope": "as_of_date_cross_section",
            "candidate_definition": {
                "candidate_type": "watchlist_or_promising_candidate",
                "meaning": (
                    "후보 종목은 매수 추천 종목이 아니라, 현재 시장 흐름과 데이터상 "
                    "관심 있게 확인할 만한 종목이다."
                ),
                "not_allowed_interpretation": [
                    "20일 뒤 기대수익률 예측값",
                    "확정적인 매수 추천",
                    "수익 보장",
                    "단기 급등 예측",
                ],
            },
            "required_outputs": [
                "market_one_line_summary",
                "market_brief",
                "market_keywords",
                "top10_candidates",
                "candidate_score",
                "component_scores",
                "chart_data",
                "actual_data",
                "top_news_3",
                "ranking_reason",
                "risk_notes",
                "glossary_terms",
                "data_limitations",
            ],
            "downstream_agents": [
                {
                    "stage": "market_flow",
                    "role": "오늘 시장 흐름 요약, 시장 키워드, 후보 선정 힌트 생성",
                    "expected_input": [
                        "ohlcv",
                        "news_features",
                        "foreign_flow",
                        "finance_features",
                        "market_analysis_result",
                        "news_invest_result_or_rag_result",
                        "risk_score_result",
                    ],
                    "expected_output": "market_flow_result.json",
                },
                {
                    "stage": "candidate_scoring",
                    "role": "시장 흐름과 개별 종목 데이터를 결합해 TOP10 후보 점수 산출",
                    "expected_input": [
                        "report_target_spec",
                        "report_feature_contract",
                        "candidate_scoring_plan",
                        "market_flow_result",
                        "ohlcv",
                        "news_features",
                        "foreign_flow",
                        "finance_features",
                        "risk_score_result",
                        "user_snapshot",
                    ],
                    "expected_output": "candidate_scoring_result.json",
                },
                {
                    "stage": "report_evidence_builder",
                    "role": "TOP10 후보별 차트, 실제 데이터, 뉴스 3개, 순위 진입 이유 구조화",
                    "expected_input": [
                        "candidate_scoring_result",
                        "ohlcv",
                        "market_analysis_result",
                        "news_invest_result_or_rag_result",
                        "risk_score_result",
                    ],
                    "expected_output": "report_evidence_result.json",
                },
                {
                    "stage": "report_gen",
                    "role": "구조화된 근거를 초보 투자자용 최종 리포트 문장으로 변환",
                    "expected_input": [
                        "market_flow_result",
                        "candidate_scoring_result",
                        "report_evidence_result",
                        "report_output_contract",
                    ],
                    "expected_output": [
                        "report.md",
                        "report_summary.json",
                    ],
                },
            ],
            "anti_hallucination_policy": {
                "principle": "최종 리포트는 입력 데이터와 Evidence Builder 결과에 있는 근거만 사용한다.",
                "rules": [
                    "없는 뉴스 제목을 만들지 않는다.",
                    "없는 재무 수치를 만들지 않는다.",
                    "없는 외국인 수급 흐름을 만들지 않는다.",
                    "candidate_score를 미래 수익률 예측값처럼 설명하지 않는다.",
                    "투자 권유 표현 대신 후보, 관심, 확인 필요 표현을 사용한다.",
                ],
            },
        }

    def _build_report_feature_contract(
        self,
        ohlcv: Optional[pd.DataFrame],
        news_feat: Optional[pd.DataFrame],
        foreign: Optional[pd.DataFrame],
        finance: Optional[pd.DataFrame],
        user: Optional[pd.DataFrame],
        market_json: Optional[Dict[str, Any]],
        news_json: Optional[Dict[str, Any]],
        risk_json: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        price_feature_candidates = [
            "종목코드", "ticker", "code", "symbol",
            "date", "datetime", "dt", "일자", "날짜",
            "open", "high", "low", "close", "volume",
            "daily_return", "log_return",
            "ma20_close", "ma60_close", "ma120_close", "ma240_close",
            "price_to_ma20", "price_to_ma60", "price_to_ma120", "price_to_ma240",
            "ma20_slope", "ma60_slope", "ma120_slope", "ma240_slope",
            "intraday_volatility", "gap_return", "true_range", "atr_14",
            "ret_vol_20d", "ret_vol_60d", "ret_vol_120d", "ret_vol_252d",
            "ret_vol_ann_20d", "ret_vol_ann_60d", "ret_vol_ann_120d", "ret_vol_ann_252d",
            "drawdown_60d", "drawdown_120d", "drawdown_252d",
            "volume_zscore_20d", "volume_zscore_60d", "volume_zscore_120d",
            "liquidity_percentile_20d", "liquidity_percentile_60d",
            "dist_to_52w_high", "dist_to_52w_low",
            "rsi_14", "rsi_28",
            "market_phase",
            "market_regime_20", "market_regime_60",
            "liquidity_state_20d", "liquidity_state_60d",
        ]

        news_feature_candidates = [
            "종목코드", "ticker", "code", "symbol",
            "recent_news_ratio",
            "event_news_ratio",
            "negative_news_ratio",
            "time_concentration_score",
            "latest_negative_flag",
        ]

        foreign_feature_candidates = [
            "종목코드", "ticker", "code", "symbol",
            "date", "datetime", "dt", "일자", "날짜",
            "foreign_net_flow_ratio",
            "foreign_ownership_level",
            "frgn_ntby_qty",
            "volume",
        ]

        finance_feature_candidates = [
            "종목코드", "ticker", "code", "symbol",
            "year",
            "revenue_yoy",
            "operating_income_yoy",
            "roe",
            "roa",
            "debt_ratio",
            "current_ratio",
            "asset_turnover",
            "revenue_cagr_3y",
            "loss_year_count_5y",
            "profit_volatility_3y",
            "operating_income",
            "net_income",
        ]

        user_feature_candidates = [
            "user_id",
            "user_age",
            "risk_score",
            "investment_horizon_score",
            "leverage_preference",
            "avg_dwell_time",
            "event_frequency_7d",
            "high_risk_action_ratio",
            "exploration_ratio",
            "session_completion_rate",
        ]

        price_features = self._pick_cols(ohlcv, price_feature_candidates)
        news_features = self._pick_cols(news_feat, news_feature_candidates)
        foreign_features = self._pick_cols(foreign, foreign_feature_candidates)
        finance_features = self._pick_cols(finance, finance_feature_candidates)
        user_features = self._pick_cols(user, user_feature_candidates)

        if user is not None:
            numeric_cols = [
                col for col in user.columns
                if pd.api.types.is_numeric_dtype(user[col])
            ]
            for col in numeric_cols:
                if col not in user_features:
                    user_features.append(col)
            user_features = user_features[:60]

        market_analysis_json_fields = [
            "market_overview.as_of_date",
            "market_overview.market_phase",
            "market_overview.market_tone",
            "market_overview.market_rsi_state",
            "market_overview.summary",
            "market_overview.evidence",
            "market_overview.risk_notes",
            "market_overview.regime_distribution",
            "market_overview.aggregate_metrics",
            "tickers[].ticker",
            "tickers[].ticker_type",
            "tickers[].phase",
            "tickers[].tone",
            "tickers[].rsi_state",
            "tickers[].return_state",
            "tickers[].compact_signals.return_20d",
            "tickers[].compact_signals.return_60d",
            "tickers[].compact_signals.return_252d",
            "tickers[].compact_signals.rsi_14",
            "tickers[].compact_signals.ann_volatility_252d",
            "tickers[].compact_signals.max_drawdown_252d",
            "tickers[].compact_signals.foreign_ownership_level",
            "tickers[].compact_signals.revenue_yoy",
            "tickers[].compact_signals.operating_income_yoy",
            "tickers[].compact_signals.roe",
            "tickers[].compact_signals.debt_ratio",
        ]

        news_json_fields = [
            "universe_summary",
            "tickers[].ticker",
            "tickers[].as_of_date",
            "tickers[].news_signal_score",
            "tickers[].confidence_level",
            "tickers[].verdict",
            "tickers[].reasons",
            "tickers[].news_summary.n_articles_7d_unique",
            "tickers[].news_summary.n_articles_30d_unique",
            "tickers[].news_summary.noise_ratio_0_1",
            "tickers[].news_summary.high_impact_ratio_0_1",
            "tickers[].news_summary.article_quality_score_0_1",
            "tickers[].top_articles",
        ]

        risk_json_fields = [
            "universe_summary",
            "tickers[].ticker",
            "tickers[].as_of_date",
            "tickers[].risk_scores.price_overheat_risk",
            "tickers[].risk_scores.downside_risk",
            "tickers[].risk_scores.financial_risk",
            "tickers[].risk_scores.news_event_risk",
            "tickers[].risk_scores.uncertainty_risk",
            "tickers[].risk_scores.overall_risk_score",
            "tickers[].risk_scores.risk_level",
            "tickers[].dominant_risk_factors",
            "tickers[].evidence",
        ]

        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "notes": [
                    "Feature contract for market-flow-based candidate report.",
                    "This contract does not require future return labels.",
                    "return_20d fields from market analysis may be used only as historical momentum signals, not as prediction labels.",
                ],
            },
            "keys": {
                "ticker_key_candidates": ["종목코드", "ticker", "code", "symbol"],
                "date_key_candidates": ["date", "datetime", "dt", "일자", "날짜"],
                "ticker_normalization": "zfill6",
            },
            "sources_summary": [
                self._summarize_df(ohlcv, "ohlcv_last365"),
                self._summarize_df(news_feat, "news_features_by_stock"),
                self._summarize_df(foreign, "foreign_snapshot_today"),
                self._summarize_df(finance, "financial_features"),
                self._summarize_df(user, "user_snapshot"),
                {"name": "market_analysis_result.json", "exists": market_json is not None},
                {"name": "news_invest_result_or_rag_result.json", "exists": news_json is not None},
                {"name": "risk_score_result.json", "exists": risk_json is not None},
            ],
            "feature_groups": {
                "price_features": price_features,
                "news_features": news_features,
                "foreign_features": foreign_features,
                "finance_features": finance_features,
                "user_features": user_features,
                "agent_json_features": {
                    "market_analysis": market_analysis_json_fields,
                    "news_invest": news_json_fields,
                    "risk_score": risk_json_fields,
                },
            },
            "usage_by_downstream_agent": {
                "market_flow": {
                    "required_groups": [
                        "price_features",
                        "news_features",
                        "foreign_features",
                        "agent_json_features.market_analysis",
                        "agent_json_features.news_invest",
                        "agent_json_features.risk_score",
                    ],
                    "purpose": "시장 전체 흐름과 핵심 키워드 추출",
                },
                "candidate_scoring": {
                    "required_groups": [
                        "price_features",
                        "news_features",
                        "foreign_features",
                        "finance_features",
                        "agent_json_features.market_analysis",
                        "agent_json_features.news_invest",
                        "agent_json_features.risk_score",
                    ],
                    "optional_groups": [
                        "user_features",
                    ],
                    "purpose": "종목별 market_flow_candidate_score 계산",
                },
                "report_evidence_builder": {
                    "required_groups": [
                        "price_features",
                        "agent_json_features.news_invest",
                        "agent_json_features.risk_score",
                    ],
                    "optional_groups": [
                        "finance_features",
                        "foreign_features",
                        "agent_json_features.market_analysis",
                    ],
                    "purpose": "종목별 차트, 실제 데이터, 뉴스, 순위 진입 이유 구성",
                },
            },
            "missing_value_policy": {
                "numeric": "median_or_neutral_score",
                "categorical": "unknown",
                "json_missing": "use_empty_list_or_null_and_add_limitation_note",
            },
        }

    def _build_candidate_scoring_plan(self) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
            },
            "scoring_target": {
                "score_name": "market_flow_candidate_score",
                "score_range": [0, 100],
                "ranking_order": "higher_is_better",
                "prediction_horizon": None,
                "meaning": (
                    "현재 시장 흐름과 종목별 데이터가 얼마나 잘 맞는지를 나타내는 후보 점수다. "
                    "미래 수익률 예측값이 아니다."
                ),
            },
            "score_components": [
                {
                    "component": "market_flow_alignment_score",
                    "weight": 0.25,
                    "direction": "positive",
                    "description": "Market Flow Agent가 도출한 핵심 시장 키워드/테마와 종목 신호의 일치도",
                    "example_signals": [
                        "시장 주도 테마와 관련 있는 종목",
                        "시장 국면과 같은 방향의 가격 흐름",
                        "market_analysis tone이 positive 또는 neutral-positive",
                    ],
                },
                {
                    "component": "news_momentum_score",
                    "weight": 0.20,
                    "direction": "positive",
                    "description": "뉴스 신호 강도, 기사 품질, 이벤트성 뉴스 비율을 반영",
                    "example_signals": [
                        "news_signal_score 높음",
                        "high_impact_ratio 높음",
                        "최근 기사 수 충분",
                        "negative_news_ratio 낮음",
                    ],
                },
                {
                    "component": "price_volume_momentum_score",
                    "weight": 0.20,
                    "direction": "positive",
                    "description": "최근 가격 흐름, 이동평균, 거래량 증가, 유동성 신호를 반영",
                    "example_signals": [
                        "daily_return 양호",
                        "price_to_ma20 개선",
                        "volume_zscore_20d 상승",
                        "liquidity_percentile_20d 높음",
                    ],
                },
                {
                    "component": "foreign_flow_score",
                    "weight": 0.15,
                    "direction": "positive",
                    "description": "외국인 순매수 비율과 외국인 보유 수준을 반영",
                    "example_signals": [
                        "foreign_net_flow_ratio 높음",
                        "foreign_ownership_level 안정적",
                    ],
                },
                {
                    "component": "fundamental_score",
                    "weight": 0.10,
                    "direction": "positive",
                    "description": "실적 성장성, 수익성, 재무 안정성을 반영",
                    "example_signals": [
                        "revenue_yoy 개선",
                        "operating_income_yoy 개선",
                        "roe 양호",
                        "debt_ratio 과도하지 않음",
                    ],
                },
                {
                    "component": "risk_penalty_score",
                    "weight": -0.10,
                    "direction": "negative",
                    "description": "과열, 하방, 재무, 뉴스 이벤트, 불확실성 리스크를 차감",
                    "example_signals": [
                        "overall_risk_score 높음",
                        "risk_level high 또는 critical",
                        "dominant_risk_factors 존재",
                    ],
                },
                {
                    "component": "user_interest_boost",
                    "weight": 0.10,
                    "direction": "positive",
                    "description": "사용자 관심 종목 또는 사용자 성향과의 적합도를 보조 가점으로 반영",
                    "example_signals": [
                        "사용자 관심 종목",
                        "사용자 위험 성향과 종목 리스크 수준 일치",
                        "사용자 탐색 패턴과 관련 있는 종목",
                    ],
                },
            ],
            "formula": (
                "candidate_score = "
                "0.25 * market_flow_alignment_score "
                "+ 0.20 * news_momentum_score "
                "+ 0.20 * price_volume_momentum_score "
                "+ 0.15 * foreign_flow_score "
                "+ 0.10 * fundamental_score "
                "- 0.10 * risk_penalty_score "
                "+ 0.10 * user_interest_boost"
            ),
            "required_candidate_output_fields": [
                "rank",
                "ticker",
                "company_name",
                "market_flow_candidate_score",
                "component_scores",
                "positive_evidence",
                "negative_evidence",
                "risk_notes",
                "ranking_reason_short",
            ],
            "selection_policy": {
                "top_k": 10,
                "minimum_data_quality": "at_least_price_data_required",
                "risk_filter_policy": [
                    "risk_level critical은 기본적으로 제외하거나 별도 주의 표시",
                    "risk_level high는 후보 점수 감점 후 포함 여부 판단",
                    "뉴스 기사 수가 너무 적으면 news_momentum_score 상한 적용",
                ],
            },
        }

    def _build_evidence_builder_contract(self) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
            },
            "purpose": (
                "Candidate Scoring Agent가 선정한 TOP10 후보를 최종 리포트에 넣을 수 있도록 "
                "차트, 실제 데이터, 주요 뉴스, 순위 진입 이유, 리스크 근거를 구조화한다."
            ),
            "input_requirements": [
                "candidate_scoring_result.json",
                "ohlcv_last365.csv",
                "market_analysis_result.json",
                "news_invest_result.json 또는 news_invest_rag_result.json",
                "risk_score_result.json",
                "financial_features.csv",
                "foreign_snapshot_today.csv",
            ],
            "per_candidate_required_fields": {
                "identity": [
                    "rank",
                    "ticker",
                    "company_name",
                    "market_flow_candidate_score",
                ],
                "chart_data": [
                    "recent_price_series",
                    "recent_volume_series",
                    "moving_average_series_optional",
                    "chart_summary",
                ],
                "actual_data": [
                    "latest_close",
                    "daily_return",
                    "volume",
                    "foreign_net_flow_ratio",
                    "roe",
                    "revenue_yoy",
                    "operating_income_yoy",
                    "overall_risk_score",
                    "risk_level",
                ],
                "top_news_3": [
                    "title",
                    "source",
                    "published_at",
                    "url",
                    "summary_optional",
                ],
                "ranking_reason": [
                    "main_reason",
                    "supporting_reasons",
                    "beginner_explanation",
                    "caution_note",
                ],
                "glossary_terms": [
                    "term",
                    "plain_korean_definition",
                    "why_it_matters",
                ],
            },
            "reason_generation_rules": [
                "순위 진입 이유는 component_scores와 positive_evidence를 기반으로 작성한다.",
                "뉴스 제목, 수치, 리스크 요인은 입력 데이터에 있는 것만 사용한다.",
                "candidate_score를 기대수익률이나 확정 수익 가능성처럼 설명하지 않는다.",
                "초보 투자자용 표현을 사용하되, 근거는 구체적으로 남긴다.",
            ],
            "output_file": "report_evidence_result.json",
        }

    def _build_report_output_contract(self) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
            },
            "final_report": {
                "report_type": "market_flow_candidate_report",
                "tone": "beginner_friendly_korean",
                "investment_disclaimer_style": (
                    "투자 권유가 아니라 시장 흐름과 데이터 기반 참고용 후보 리포트임을 명시"
                ),
                "sections": [
                    {
                        "section_id": "market_summary",
                        "title": "오늘 시장 한 줄 요약",
                        "required_fields": [
                            "market_one_line_summary",
                            "market_brief",
                            "market_keywords",
                        ],
                    },
                    {
                        "section_id": "top10_candidates",
                        "title": "유망/관심 종목 후보 TOP10",
                        "required_fields": [
                            "rank",
                            "ticker",
                            "company_name",
                            "market_flow_candidate_score",
                            "ranking_reason_short",
                            "risk_level",
                        ],
                    },
                    {
                        "section_id": "candidate_detail",
                        "title": "종목별 상세 근거",
                        "repeat_for": "top10_candidates",
                        "required_fields": [
                            "chart_data",
                            "actual_data",
                            "top_news_3",
                            "ranking_reason",
                            "risk_notes",
                        ],
                    },
                    {
                        "section_id": "glossary",
                        "title": "초보 투자자를 위한 용어 풀이",
                        "required_fields": [
                            "glossary_terms",
                        ],
                    },
                    {
                        "section_id": "limitations",
                        "title": "확인해야 할 점",
                        "required_fields": [
                            "data_limitations",
                            "risk_notes",
                        ],
                    },
                ],
            },
            "report_gen_policy": {
                "role": "analysis_to_document_writer",
                "allowed": [
                    "구조화된 근거를 자연스러운 문장으로 편집",
                    "초보자용 쉬운 설명 추가",
                    "반복되는 표현 정리",
                    "용어 풀이 추가",
                ],
                "not_allowed": [
                    "새로운 뉴스나 수치 생성",
                    "없는 기업 이벤트 생성",
                    "매수/매도 추천 표현",
                    "후보 점수를 미래 수익률로 해석",
                ],
            },
            "expected_output_files": [
                "report.md",
                "report_summary.json",
            ],
        }

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
            ctx.logger.info(f"[ReportTargetPlannerAgent] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            ctx.logger.warning(
                f"[ReportTargetPlannerAgent] Multiple files matched: {pattern}. "
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
            ctx.logger.warning(f"[ReportTargetPlannerAgent] Missing CSV: {path}")
            return None

        try:
            return pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            ctx.logger.warning(
                f"[ReportTargetPlannerAgent] Failed to read CSV: {path} ({e})"
            )
            return None

    def _safe_read_json(
        self,
        path: Optional[Path],
        ctx: RunContext,
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            return None

        if not path.exists():
            ctx.logger.warning(f"[ReportTargetPlannerAgent] Missing JSON: {path}")
            return None

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            ctx.logger.warning(
                f"[ReportTargetPlannerAgent] Failed to read JSON: {path} ({e})"
            )
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
    def _pick_cols(df: Optional[pd.DataFrame], candidates: List[str]) -> List[str]:
        if df is None:
            return []

        result = []
        seen = set()

        for col in candidates:
            if col in df.columns and col not in seen:
                result.append(col)
                seen.add(col)

        return result

    @staticmethod
    def _summarize_df(df: Optional[pd.DataFrame], name: str) -> Dict[str, Any]:
        if df is None:
            return {
                "name": name,
                "exists": False,
            }

        return {
            "name": name,
            "exists": True,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns)[:200],
        }

    @staticmethod
    def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _infer_as_of_date(self, ohlcv: Optional[pd.DataFrame]) -> Optional[str]:
        if ohlcv is None or ohlcv.empty:
            return None

        date_col = self._infer_col(
            ohlcv,
            ["date", "datetime", "dt", "일자", "날짜"],
        )

        if date_col is None:
            return None

        dates = pd.to_datetime(ohlcv[date_col], errors="coerce").dropna()

        if dates.empty:
            return None

        return str(dates.max().date())

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