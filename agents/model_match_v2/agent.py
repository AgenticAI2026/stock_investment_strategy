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
class ModelTargetMatcherArtifacts:
    run_dir: Path
    output_dir: Path
    target_spec_path: Path
    feature_contract_path: Path
    model_candidates_path: Path
    evaluation_plan_path: Path
    manifest_path: Path
    output_files: List[str]


class ModelTargetMatcherAgent(BaseAgent):
    stage = "model_match_v2"
    version = "1.1-ranking-20d"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = ModelTargetMatcherArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            target_spec_path=output_dir / "target_spec.json",
            feature_contract_path=output_dir / "feature_contract_model_target.json",
            model_candidates_path=output_dir / "model_candidates.json",
            evaluation_plan_path=output_dir / "evaluation_plan.json",
            manifest_path=output_dir / "manifest.json",
            output_files=[],
        )

        paths = {
            "ohlcv": self._find_single_file(run_dir, "preprocessed__*__price__ohlcv_last365.csv"),
            "news_feat": self._find_single_file(run_dir, "preprocessed__*__news__news_features_by_stock.csv"),
            "foreign": self._find_single_file(run_dir, "preprocessed__*__price__foreign_snapshot_today.csv"),
            "finance": self._find_single_file(run_dir, "preprocessed__*__finance__financial_features.csv"),
            "user": self._find_single_file(run_dir, "preprocessed__*__user__user_snapshot.csv"),
            "market_json": self._find_single_file(run_dir, "market_analysis_result.json"),
            "news_json": self._find_single_file(run_dir, "news_invest_rag_result.json"),
            "risk_json": self._find_single_file(run_dir, "risk_score_result.json"),
        }

        ohlcv = self._safe_read_csv(paths["ohlcv"])
        news_feat = self._safe_read_csv(paths["news_feat"])
        foreign = self._safe_read_csv(paths["foreign"])
        finance = self._safe_read_csv(paths["finance"])
        user = self._safe_read_csv(paths["user"])

        market_json = self._safe_read_json(paths["market_json"])
        news_json = self._safe_read_json(paths["news_json"])
        risk_json = self._safe_read_json(paths["risk_json"])

        target_spec = self._build_target_spec(paths)
        feature_contract = self._build_feature_contract(
            ohlcv=ohlcv,
            news_feat=news_feat,
            foreign=foreign,
            finance=finance,
            user=user,
            market_json=market_json,
            news_json=news_json,
            risk_json=risk_json,
        )
        model_candidates = self._build_model_candidates()
        evaluation_plan = self._build_evaluation_plan()

        self._save_json(target_spec, artifacts.target_spec_path)
        self._save_json(feature_contract, artifacts.feature_contract_path)
        self._save_json(model_candidates, artifacts.model_candidates_path)
        self._save_json(evaluation_plan, artifacts.evaluation_plan_path)

        artifacts.output_files = [
            str(artifacts.target_spec_path),
            str(artifacts.feature_contract_path),
            str(artifacts.model_candidates_path),
            str(artifacts.evaluation_plan_path),
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
            "main_target": "rank_20d",
            "horizon": "20 trading days",
        }

        self._save_json(manifest, artifacts.manifest_path)

        metrics = {
            "price_features": len(feature_contract["feature_groups"]["price_features"]),
            "news_features": len(feature_contract["feature_groups"]["news_features"]),
            "foreign_features": len(feature_contract["feature_groups"]["foreign_features"]),
            "finance_features": len(feature_contract["feature_groups"]["finance_features"]),
            "user_features": len(feature_contract["feature_groups"]["user_features"]),
        }

        outputs = {
            "output_dir": str(output_dir),
            "target_spec": str(artifacts.target_spec_path),
            "feature_contract": str(artifacts.feature_contract_path),
            "model_candidates": str(artifacts.model_candidates_path),
            "evaluation_plan": str(artifacts.evaluation_plan_path),
            "manifest": str(artifacts.manifest_path),
        }

        return self._make_stage_result(
            status="success",
            message="Model target matching completed.",
            metrics=metrics,
            outputs=outputs,
        )

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

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

    def _find_single_file(
        self,
        run_dir: Path,
        pattern: str,
    ) -> Optional[Path]:
        matches = sorted(run_dir.rglob(pattern))

        if not matches:
            print(f"[WARN] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            print(f"[WARN] Multiple files matched: {pattern}")
            for path in matches:
                print(f" - {path}")

        return matches[0]

    def _safe_read_csv(
        self,
        path: Optional[Path],
    ) -> Optional[pd.DataFrame]:
        if path is None:
            return None

        if not path.exists():
            print(f"[WARN] Missing CSV: {path}")
            return None

        try:
            return pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] Failed to read CSV: {path} ({e})")
            return None

    def _safe_read_json(
        self,
        path: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            return None

        if not path.exists():
            print(f"[WARN] Missing JSON: {path}")
            return None

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read JSON: {path} ({e})")
            return None

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

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
            return {"name": name, "exists": False}

        return {
            "name": name,
            "exists": True,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns)[:200],
        }

    def _build_target_spec(
        self,
        paths: Dict[str, Optional[Path]],
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
                "main_goal": "Predict stock ranking based on 20-trading-day forward return.",
            },
            "base_labels": [
                {
                    "label_id": "return_20d",
                    "label_type": "continuous_auxiliary_label",
                    "definition": "return_20d = close(t+20)/close(t) - 1",
                    "time_horizon_trading_days": 20,
                    "usage": [
                        "Used to construct rank_20d.",
                        "Used to construct top_quantile_20d.",
                        "Not treated as the main prediction target in ranking setup.",
                    ],
                    "primary_key": ["ticker", "date"],
                    "requires_price_history": True,
                }
            ],
            "targets": [
                {
                    "target_id": "rank_20d",
                    "task_type": "ranking",
                    "label_definition": (
                        "rank_20d = cross-sectional rank of return_20d within the same date. "
                        "Higher return_20d receives a better rank."
                    ),
                    "base_return_label": "return_20d",
                    "time_horizon_trading_days": 20,
                    "ranking_scope": "same_date_cross_section",
                    "rank_order": "higher_return_is_better",
                    "granularity": "daily",
                    "primary_key": ["ticker", "date"],
                    "requires_price_history": True,
                },
                {
                    "target_id": "rank_percentile_20d",
                    "task_type": "ranking_regression",
                    "label_definition": (
                        "rank_percentile_20d = percentile rank of return_20d within the same date, "
                        "scaled from 0 to 1, where 1 means highest future return group."
                    ),
                    "base_return_label": "return_20d",
                    "time_horizon_trading_days": 20,
                    "ranking_scope": "same_date_cross_section",
                    "primary_key": ["ticker", "date"],
                    "requires_price_history": True,
                },
                {
                    "target_id": "top_quantile_20d",
                    "task_type": "classification",
                    "label_definition": (
                        "top_quantile_20d = 1 if stock is in the top 20% by return_20d "
                        "within the same date; otherwise 0."
                    ),
                    "base_return_label": "return_20d",
                    "time_horizon_trading_days": 20,
                    "top_quantile": 0.20,
                    "ranking_scope": "same_date_cross_section",
                    "primary_key": ["ticker", "date"],
                    "requires_price_history": True,
                },
                {
                    "target_id": "downside_20d",
                    "task_type": "classification",
                    "label_definition": (
                        "downside_20d = 1 if min(close[t..t+20])/close(t)-1 <= -0.10; otherwise 0."
                    ),
                    "time_horizon_trading_days": 20,
                    "threshold": -0.10,
                    "granularity": "daily",
                    "primary_key": ["ticker", "date"],
                    "requires_price_history": True,
                },
                {
                    "target_id": "personalized_ranking_score",
                    "task_type": "scoring",
                    "label_definition": (
                        "personalized_ranking_score = "
                        "w_user * RankScore(rank_20d or rank_percentile_20d) "
                        "+ alpha * P(top_quantile_20d) "
                        "- beta * P(downside_20d)"
                    ),
                    "depends_on_targets": [
                        "rank_20d",
                        "rank_percentile_20d",
                        "top_quantile_20d",
                        "downside_20d",
                    ],
                    "requires_user_snapshot": True,
                    "primary_key": ["user_id", "ticker", "as_of_date"],
                },
            ],
            "recommended_run_order": [
                "rank_20d",
                "rank_percentile_20d",
                "top_quantile_20d",
                "downside_20d",
                "personalized_ranking_score",
            ],
        }

    def _build_feature_contract(
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
            "close", "open", "high", "low", "volume",
            "daily_return", "log_return",
            "ma20_close", "ma60_close", "ma120_close", "ma240_close",
            "price_to_ma20", "price_to_ma60", "price_to_ma120", "price_to_ma240",
            "ma20_slope", "ma60_slope", "ma120_slope", "ma240_slope",
            "intraday_volatility", "gap_return", "true_range", "atr_14",
            "ret_vol_20d", "ret_vol_ann_20d",
            "ret_vol_60d", "ret_vol_ann_60d",
            "ret_vol_120d", "ret_vol_ann_120d",
            "ret_vol_252d", "ret_vol_ann_252d",
            "drawdown_60d", "drawdown_120d", "drawdown_252d",
            "volume_zscore_20d", "volume_zscore_60d", "volume_zscore_120d",
            "liquidity_percentile_20d", "liquidity_percentile_60d",
            "dist_to_52w_high", "dist_to_52w_low",
            "rsi_14", "rsi_28",
            "liquidity_state_20d_High", "liquidity_state_20d_Low", "liquidity_state_20d_Medium",
            "liquidity_state_60d_High", "liquidity_state_60d_Low", "liquidity_state_60d_Medium",
            "market_regime_20_Down", "market_regime_20_Side", "market_regime_20_Up",
            "market_regime_60_Down", "market_regime_60_Side", "market_regime_60_Up",
            "시장_KOSDAQ", "시장_KOSPI",
            "market_phase",
            "prev_close",
            "rolling_max_60d",
            "rolling_max_120d",
            "rolling_max_252d",
            "high_52w",
            "low_52w",
            "liquidity_state_20d",
            "liquidity_state_60d",
            "market_regime_20",
            "market_regime_60",
        ]

        news_feature_candidates = [
            "recent_news_ratio",
            "event_news_ratio",
            "negative_news_ratio",
            "time_concentration_score",
            "latest_negative_flag",
        ]

        foreign_feature_candidates = [
            "foreign_net_flow_ratio",
            "foreign_ownership_level",
            "frgn_ntby_qty",
            "volume",
        ]

        finance_feature_candidates = [
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
            "year",
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

        market_json_features = [
            "ticker_type",
            "phase",
            "tone",
            "rsi_state",
            "return_state",
            "compact_signals.return_20d",
            "compact_signals.return_60d",
            "compact_signals.return_252d",
            "compact_signals.dist_to_ma20",
            "compact_signals.dist_to_ma60",
            "compact_signals.rsi_14",
            "compact_signals.ann_volatility_252d",
            "compact_signals.max_drawdown_252d",
            "compact_signals.market_regime_20d",
            "compact_signals.market_regime_60d",
            "compact_signals.foreign_ownership_level",
            "compact_signals.revenue_yoy",
            "compact_signals.operating_income_yoy",
            "compact_signals.roe",
            "compact_signals.debt_ratio",
        ]

        news_json_features = [
            "news_signal_score",
            "confidence_level",
            "verdict",
            "news_summary.n_articles_7d_unique",
            "news_summary.n_articles_30d_unique",
            "news_summary.noise_ratio_0_1",
            "news_summary.high_impact_ratio_0_1",
            "news_summary.article_quality_score_0_1",
        ]

        risk_json_features = [
            "risk_scores.price_overheat_risk",
            "risk_scores.downside_risk",
            "risk_scores.financial_risk",
            "risk_scores.news_event_risk",
            "risk_scores.uncertainty_risk",
            "risk_scores.overall_risk_score",
            "risk_scores.risk_level",
        ]

        common_optional_groups = [
            "news_features",
            "foreign_features",
            "finance_features",
            "agent_json_features.market_analysis",
            "agent_json_features.news_invest_rag",
            "agent_json_features.risk_score",
        ]

        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "notes": [
                    "Feature contract for 20-day stock ranking prediction.",
                    "Main target is rank_20d, not exact return_20d regression.",
                    "return_20d is used as an auxiliary base label to construct ranking labels.",
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
                {"name": "news_invest_rag_result.json", "exists": news_json is not None},
                {"name": "risk_score_result.json", "exists": risk_json is not None},
            ],
            "feature_groups": {
                "price_features": price_features,
                "news_features": news_features,
                "foreign_features": foreign_features,
                "finance_features": finance_features,
                "user_features": user_features,
                "agent_json_features": {
                    "market_analysis": market_json_features,
                    "news_invest_rag": news_json_features,
                    "risk_score": risk_json_features,
                },
            },
            "targets_to_features": {
                "rank_20d": {
                    "required_groups": ["price_features"],
                    "optional_groups": common_optional_groups,
                    "missing_value_policy": {
                        "numeric": "median_impute",
                        "categorical": "unknown",
                    },
                },
                "rank_percentile_20d": {
                    "required_groups": ["price_features"],
                    "optional_groups": common_optional_groups,
                    "missing_value_policy": {
                        "numeric": "median_impute",
                        "categorical": "unknown",
                    },
                },
                "top_quantile_20d": {
                    "required_groups": ["price_features"],
                    "optional_groups": common_optional_groups,
                    "missing_value_policy": {
                        "numeric": "median_impute",
                        "categorical": "unknown",
                    },
                },
                "downside_20d": {
                    "required_groups": ["price_features"],
                    "optional_groups": common_optional_groups,
                    "missing_value_policy": {
                        "numeric": "median_impute",
                        "categorical": "unknown",
                    },
                },
                "personalized_ranking_score": {
                    "required_groups": ["user_features"],
                    "optional_groups": [
                        "agent_json_features.risk_score",
                        "agent_json_features.market_analysis",
                        "agent_json_features.news_invest_rag",
                    ],
                    "depends_on_model_outputs": [
                        "pred_rank_score_20d",
                        "pred_rank_percentile_20d",
                        "P_top_quantile_20d",
                        "P_downside_20d",
                    ],
                    "missing_value_policy": {
                        "numeric": "median_impute",
                        "categorical": "unknown",
                    },
                },
            },
        }

    def _build_model_candidates(self) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
            },
            "candidates": {
                "rank_20d": [
                    {
                        "model_id": "lgbm_ranker_v1",
                        "family": "LightGBM",
                        "task": "ranking",
                    },
                    {
                        "model_id": "lgbm_regressor_rank_proxy_v1",
                        "family": "LightGBM",
                        "task": "regression_to_rank",
                    },
                    {
                        "model_id": "ridge_rank_proxy_baseline",
                        "family": "LinearModel",
                        "task": "regression_to_rank",
                    },
                ],
                "rank_percentile_20d": [
                    {
                        "model_id": "lgbm_rank_percentile_regressor_v1",
                        "family": "LightGBM",
                        "task": "regression",
                    },
                    {
                        "model_id": "ridge_rank_percentile_baseline",
                        "family": "LinearModel",
                        "task": "regression",
                    },
                ],
                "top_quantile_20d": [
                    {
                        "model_id": "lgbm_top_quantile_classifier_v1",
                        "family": "LightGBM",
                        "task": "classification",
                    },
                    {
                        "model_id": "logistic_top_quantile_baseline",
                        "family": "LinearModel",
                        "task": "classification",
                    },
                ],
                "downside_20d": [
                    {
                        "model_id": "lgbm_downside_classifier_v1",
                        "family": "LightGBM",
                        "task": "classification",
                    },
                    {
                        "model_id": "logistic_downside_baseline",
                        "family": "LinearModel",
                        "task": "classification",
                    },
                ],
                "personalized_ranking_score": [
                    {
                        "model_id": "rule_based_personalized_rank_blend_v1",
                        "family": "ScoringRule",
                        "task": "scoring",
                        "definition": (
                            "score = w_user*pred_rank_score_20d "
                            "+ alpha*P(top_quantile_20d) "
                            "- beta*P(downside_20d)"
                        ),
                    }
                ],
            },
        }

    def _build_evaluation_plan(self) -> Dict[str, Any]:
        return {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
            },
            "data_split": {
                "method": "time_based_split",
                "suggested": {
                    "train_window_days": 252,
                    "validation_window_days": 60,
                    "test_window_days": 60,
                    "embargo_days": 5,
                },
            },
            "label_generation": {
                "return_20d": "close(t+20)/close(t)-1",
                "rank_20d": "cross-sectional rank of return_20d within each date",
                "rank_percentile_20d": "percentile rank of return_20d within each date, scaled 0 to 1",
                "top_quantile_20d": "1 if return_20d is in top 20% within each date",
                "downside_20d": "1 if future 20-day drawdown is less than or equal to -10%",
            },
            "metrics": {
                "rank_20d": {
                    "primary": ["NDCG@5", "NDCG@10", "SpearmanRankIC"],
                    "secondary": [
                        "TopK_HitRate",
                        "Precision@5",
                        "Precision@10",
                        "MeanRankOfActualTopK",
                    ],
                },
                "rank_percentile_20d": {
                    "primary": ["SpearmanRankIC", "NDCG@10"],
                    "secondary": ["MAE_on_rank_percentile", "TopK_HitRate"],
                },
                "top_quantile_20d": {
                    "primary": ["AUC", "PR-AUC", "Precision@Top20%"],
                    "secondary": ["Recall@Top20%", "F1"],
                },
                "downside_20d": {
                    "primary": ["AUC", "Recall@PositiveClass"],
                    "secondary": ["Precision", "F1", "PR-AUC"],
                },
                "personalized_ranking_score": {
                    "primary": ["NDCG@K", "TopK_HitRate"],
                    "secondary": [
                        "UserSegmentBreakdown(risk_score bins)",
                        "AverageDownsideRiskInTopK",
                    ],
                },
            },
        }