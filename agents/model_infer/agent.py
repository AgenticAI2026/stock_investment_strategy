from __future__ import annotations

import inspect
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.agent_base import BaseAgent
from core.artifacts import ArtifactPaths
from core.context import RunContext
from core.result import StageResult


@dataclass
class ModelInferenceArtifacts:
    run_dir: Path
    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    personalized_scores_path: Path
    manifest_path: Path
    model_dir: Path
    output_files: List[str]


class ModelInferenceAgent(BaseAgent):
    """
    Model Runner Agent compatible with ModelTargetMatcherAgent v1.1-ranking-20d.

    Expected upstream artifacts from model_match_v2:
      - target_spec.json
      - feature_contract_model_target.json
      - model_candidates.json
      - evaluation_plan.json

    Main targets:
      - rank_20d
      - rank_percentile_20d
      - top_quantile_20d
      - downside_20d
      - personalized_ranking_score
    """

    stage = "model_infer"
    version = "2.0-ranking-20d"

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = ModelInferenceArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            metrics_path=output_dir / "model_infer_metrics.json",
            predictions_path=output_dir / "model_infer_predictions_20d.csv",
            feature_importance_path=output_dir / "model_infer_feature_importance.json",
            personalized_scores_path=output_dir / "model_infer_personalized_scores.csv",
            manifest_path=output_dir / "manifest.json",
            model_dir=output_dir / "models",
            output_files=[],
        )
        artifacts.model_dir.mkdir(parents=True, exist_ok=True)

        ctx.logger.info("[ModelInferenceAgent] Starting model runner.")

        matcher_dir = run_dir / "model_match_v2"

        paths = {
            "target_spec": matcher_dir / "target_spec.json",
            "feature_contract": matcher_dir / "feature_contract_model_target.json",
            "model_candidates": matcher_dir / "model_candidates.json",
            "evaluation_plan": matcher_dir / "evaluation_plan.json",
            "ohlcv": self._find_single_file(run_dir, "preprocessed__*__price__ohlcv_last365.csv", ctx),
            "news_feat": self._find_single_file(run_dir, "preprocessed__*__news__news_features_by_stock.csv", ctx),
            "foreign": self._find_single_file(run_dir, "preprocessed__*__price__foreign_snapshot_today.csv", ctx),
            "finance": self._find_single_file(run_dir, "preprocessed__*__finance__financial_features.csv", ctx),
            "user": self._find_single_file(run_dir, "preprocessed__*__user__user_snapshot.csv", ctx),
            "market_json": self._find_single_file(run_dir, "market_analysis_result.json", ctx),
            "news_json": self._find_single_file(run_dir, "news_invest_result.json", ctx),
            "risk_json": self._find_single_file(run_dir, "risk_score_result.json", ctx),
        }

        self._require_files(
            [
                paths["target_spec"],
                paths["feature_contract"],
                paths["model_candidates"],
                paths["evaluation_plan"],
                paths["ohlcv"],
            ]
        )

        target_spec = self._read_json(paths["target_spec"])
        feature_contract = self._read_json(paths["feature_contract"])
        model_candidates = self._read_json(paths["model_candidates"])
        evaluation_plan = self._read_json(paths["evaluation_plan"])

        split_params = self._get_split_params(evaluation_plan)
        horizon = self._get_horizon(target_spec)
        downside_threshold = self._get_downside_threshold(target_spec)
        top_quantile = self._get_top_quantile(target_spec)

        ctx.logger.info(
            f"[ModelInferenceAgent] horizon={horizon}, "
            f"top_quantile={top_quantile}, downside_threshold={downside_threshold}"
        )

        dataset = self._build_dataset(
            ctx=ctx,
            paths=paths,
            feature_contract=feature_contract,
            horizon=horizon,
            downside_threshold=downside_threshold,
            top_quantile=top_quantile,
        )

        X = dataset["X"]
        feature_cols = dataset["feature_cols"]
        ticker_col = dataset["ticker_col"]
        date_col = dataset["date_col"]

        train_mask, val_mask, test_mask = self._make_time_split(
            X=X,
            date_col=date_col,
            split_params=split_params,
            ctx=ctx,
        )

        candidate_map = model_candidates.get("candidates", {})
        if not candidate_map:
            raise ValueError("model_candidates.json has no candidates.")

        results: Dict[str, Any] = {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "horizon_days": horizon,
                "top_quantile": top_quantile,
                "downside_threshold": downside_threshold,
                "split_params": split_params,
                "feature_count": len(feature_cols),
                "rows": {
                    "train": int(train_mask.sum()),
                    "val": int(val_mask.sum()),
                    "test": int(test_mask.sum()),
                },
            },
            "models_run": [],
            "rank_20d": {},
            "rank_percentile_20d": {},
            "top_quantile_20d": {},
            "downside_20d": {},
            "personalized_ranking_score": {},
            "selected_models": {},
        }

        predictions = X.loc[test_mask, [ticker_col, date_col]].copy()
        predictions = predictions.rename(columns={ticker_col: "ticker", date_col: "date"})
        predictions["y_true_return_20d"] = X.loc[test_mask, "return_20d"].values
        predictions["y_true_rank_20d"] = X.loc[test_mask, "rank_20d"].values
        predictions["y_true_rank_percentile_20d"] = X.loc[test_mask, "rank_percentile_20d"].values
        predictions["y_true_top_quantile_20d"] = X.loc[test_mask, "top_quantile_20d"].values
        predictions["y_true_downside_20d"] = X.loc[test_mask, "downside_20d"].values

        feature_importance_out: Dict[str, Any] = {}

        # 1) rank_20d models
        for model_info in candidate_map.get("rank_20d", []):
            model_id = model_info.get("model_id")
            if not model_id:
                continue

            ctx.logger.info(f"[ModelInferenceAgent] Running rank_20d model: {model_id}")
            y_pred, model, family = self._fit_predict_model(
                model_id=model_id,
                target="rank_20d",
                X=X,
                feature_cols=feature_cols,
                date_col=date_col,
                train_mask=train_mask,
                test_mask=test_mask,
                ctx=ctx,
            )

            pred_col = f"pred_rank_score_20d__{model_id}"
            predictions[pred_col] = y_pred

            metrics = self._ranking_metrics(
                df=predictions,
                score_col=pred_col,
                true_return_col="y_true_return_20d",
                date_col="date",
                ks=[5, 10],
            )
            results["rank_20d"][model_id] = metrics
            results["models_run"].append({"target": "rank_20d", "model_id": model_id, "family": family})

            self._save_model(model, artifacts.model_dir / f"{model_id}.pkl")
            feature_importance_out[model_id] = self._extract_feature_importance(model, feature_cols)

        # 2) rank_percentile_20d models
        for model_info in candidate_map.get("rank_percentile_20d", []):
            model_id = model_info.get("model_id")
            if not model_id:
                continue

            ctx.logger.info(f"[ModelInferenceAgent] Running rank_percentile_20d model: {model_id}")
            y_pred, model, family = self._fit_predict_model(
                model_id=model_id,
                target="rank_percentile_20d",
                X=X,
                feature_cols=feature_cols,
                date_col=date_col,
                train_mask=train_mask,
                test_mask=test_mask,
                ctx=ctx,
            )

            y_pred = np.clip(y_pred, 0.0, 1.0)
            pred_col = f"pred_rank_percentile_20d__{model_id}"
            predictions[pred_col] = y_pred

            metrics = self._ranking_metrics(
                df=predictions,
                score_col=pred_col,
                true_return_col="y_true_return_20d",
                date_col="date",
                ks=[5, 10],
            )
            metrics["MAE_rank_percentile"] = self._safe_mae(
                predictions["y_true_rank_percentile_20d"], predictions[pred_col]
            )

            results["rank_percentile_20d"][model_id] = metrics
            results["models_run"].append({"target": "rank_percentile_20d", "model_id": model_id, "family": family})

            self._save_model(model, artifacts.model_dir / f"{model_id}.pkl")
            feature_importance_out[model_id] = self._extract_feature_importance(model, feature_cols)

        # 3) top_quantile_20d models
        for model_info in candidate_map.get("top_quantile_20d", []):
            model_id = model_info.get("model_id")
            if not model_id:
                continue

            ctx.logger.info(f"[ModelInferenceAgent] Running top_quantile_20d model: {model_id}")
            proba, model, family = self._fit_predict_model(
                model_id=model_id,
                target="top_quantile_20d",
                X=X,
                feature_cols=feature_cols,
                date_col=date_col,
                train_mask=train_mask,
                test_mask=test_mask,
                ctx=ctx,
            )

            pred_col = f"proba_top_quantile_20d__{model_id}"
            predictions[pred_col] = proba

            metrics = self._classification_metrics(
                y_true=predictions["y_true_top_quantile_20d"],
                proba=predictions[pred_col],
            )
            metrics.update(
                self._ranking_metrics(
                    df=predictions,
                    score_col=pred_col,
                    true_return_col="y_true_return_20d",
                    date_col="date",
                    ks=[5, 10],
                )
            )

            results["top_quantile_20d"][model_id] = metrics
            results["models_run"].append({"target": "top_quantile_20d", "model_id": model_id, "family": family})

            self._save_model(model, artifacts.model_dir / f"{model_id}.pkl")
            feature_importance_out[model_id] = self._extract_feature_importance(model, feature_cols)

        # 4) downside_20d models
        for model_info in candidate_map.get("downside_20d", []):
            model_id = model_info.get("model_id")
            if not model_id:
                continue

            ctx.logger.info(f"[ModelInferenceAgent] Running downside_20d model: {model_id}")
            proba, model, family = self._fit_predict_model(
                model_id=model_id,
                target="downside_20d",
                X=X,
                feature_cols=feature_cols,
                date_col=date_col,
                train_mask=train_mask,
                test_mask=test_mask,
                ctx=ctx,
            )

            pred_col = f"proba_downside_20d__{model_id}"
            predictions[pred_col] = proba

            metrics = self._classification_metrics(
                y_true=predictions["y_true_downside_20d"],
                proba=predictions[pred_col],
            )
            results["downside_20d"][model_id] = metrics
            results["models_run"].append({"target": "downside_20d", "model_id": model_id, "family": family})

            self._save_model(model, artifacts.model_dir / f"{model_id}.pkl")
            feature_importance_out[model_id] = self._extract_feature_importance(model, feature_cols)

        selected = self._select_best_models(results)
        results["selected_models"] = selected
        ctx.logger.info(f"[ModelInferenceAgent] selected_models={selected}")

        # 5) personalized_ranking_score
        personalized_df = self._build_personalized_scores(
            ctx=ctx,
            X=X,
            predictions=predictions,
            paths=paths,
            candidate_map=candidate_map,
            selected=selected,
            artifacts=artifacts,
        )

        if personalized_df is not None:
            results["models_run"].append(
                {
                    "target": "personalized_ranking_score",
                    "model_id": "rule_based_personalized_rank_blend_v1",
                    "family": "ScoringRule",
                }
            )
            results["personalized_ranking_score"]["rule_based_personalized_rank_blend_v1"] = {
                "rows": int(len(personalized_df)),
                "output_path": str(artifacts.personalized_scores_path),
                "uses": selected,
            }

        predictions.to_csv(artifacts.predictions_path, index=False, encoding="utf-8-sig")
        self._save_json(results, artifacts.metrics_path)
        self._save_json(feature_importance_out, artifacts.feature_importance_path)

        artifacts.output_files = [
            str(artifacts.metrics_path),
            str(artifacts.predictions_path),
            str(artifacts.feature_importance_path),
            str(artifacts.manifest_path),
        ]
        if artifacts.personalized_scores_path.exists():
            artifacts.output_files.append(str(artifacts.personalized_scores_path))

        manifest = {
            "stage": self.stage,
            "version": self.version,
            "created_at_utc": self._now_utc(),
            "input_files": {k: str(v) if v is not None else None for k, v in paths.items()},
            "output_files": artifacts.output_files,
            "selected_models": selected,
        }
        self._save_json(manifest, artifacts.manifest_path)

        outputs = {
            "output_dir": str(output_dir),
            "metrics": str(artifacts.metrics_path),
            "predictions": str(artifacts.predictions_path),
            "feature_importance": str(artifacts.feature_importance_path),
            "manifest": str(artifacts.manifest_path),
            "model_dir": str(artifacts.model_dir),
        }
        if artifacts.personalized_scores_path.exists():
            outputs["personalized_scores"] = str(artifacts.personalized_scores_path)

        metrics_summary = {
            "feature_count": int(len(feature_cols)),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "models_run": int(len(results["models_run"])),
        }

        ctx.logger.info("[ModelInferenceAgent] Completed model runner.")

        return self._make_stage_result(
            status="success",
            message="Model running completed.",
            metrics=metrics_summary,
            outputs=outputs,
        )

    # =========================================================
    # Dataset build
    # =========================================================
    def _build_dataset(
        self,
        ctx: RunContext,
        paths: Dict[str, Optional[Path]],
        feature_contract: Dict[str, Any],
        horizon: int,
        downside_threshold: float,
        top_quantile: float,
    ) -> Dict[str, Any]:
        ohlcv = self._read_csv(paths["ohlcv"])

        ticker_col = self._infer_col(ohlcv, ["종목코드", "ticker", "code", "symbol"])
        date_col = self._infer_col(ohlcv, ["date", "datetime", "dt", "일자", "날짜"])
        close_col = self._infer_col(ohlcv, ["close", "Close", "종가"])

        if ticker_col is None or date_col is None or close_col is None:
            raise ValueError("OHLCV must include ticker/date/close columns.")

        ohlcv[ticker_col] = ohlcv[ticker_col].apply(self._zfill6)
        ohlcv[date_col] = pd.to_datetime(ohlcv[date_col], errors="coerce")
        ohlcv = ohlcv.dropna(subset=[date_col])
        ohlcv = ohlcv.sort_values([ticker_col, date_col]).reset_index(drop=True)

        fg = feature_contract.get("feature_groups", {})
        price_cols = fg.get("price_features", [])
        news_cols = fg.get("news_features", [])
        foreign_cols = fg.get("foreign_features", [])
        finance_cols = fg.get("finance_features", [])

        keep_cols = [ticker_col, date_col, close_col]
        keep_cols += [c for c in price_cols if c in ohlcv.columns and c not in keep_cols]
        X = ohlcv[keep_cols].copy()

        X["return_20d"] = (
            X.groupby(ticker_col)[close_col].shift(-horizon) / X[close_col] - 1.0
        )

        def add_downside(group: pd.DataFrame) -> pd.Series:
            s = group[close_col].astype(float)
            future_min = self._future_min_window(s, window=horizon + 1)
            downside = future_min / s - 1.0
            return (downside <= downside_threshold).astype(int)

        X["downside_20d"] = (
            X.groupby(ticker_col, group_keys=False)
            .apply(add_downside)
            .reset_index(level=0, drop=True)
        )

        X = X.dropna(subset=["return_20d"]).copy()

        X["rank_20d"] = X.groupby(date_col)["return_20d"].rank(
            method="first",
            ascending=False,
        )

        X["rank_percentile_20d"] = X.groupby(date_col)["return_20d"].rank(
            method="average",
            pct=True,
            ascending=True,
        )

        X["top_quantile_20d"] = X.groupby(date_col)["return_20d"].transform(
            lambda s: (s >= s.quantile(1.0 - top_quantile)).astype(int)
        )

        X = self._join_ticker_level(
            X=X,
            left_ticker_col=ticker_col,
            path=paths["news_feat"],
            cols=news_cols,
            name="news_features",
            latest_by_date=True,
            ctx=ctx,
        )
        X = self._join_ticker_level(
            X=X,
            left_ticker_col=ticker_col,
            path=paths["foreign"],
            cols=foreign_cols,
            name="foreign_snapshot",
            latest_by_date=True,
            ctx=ctx,
        )
        X = self._join_ticker_level(
            X=X,
            left_ticker_col=ticker_col,
            path=paths["finance"],
            cols=finance_cols,
            name="finance_features",
            latest_by_year=True,
            ctx=ctx,
        )

        agent_static = self._build_agent_static_features(feature_contract, paths)
        if agent_static is not None and len(agent_static) > 0:
            before = set(X.columns)
            agent_static = agent_static.rename(columns={"종목코드": "__ticker_key"})
            X = X.merge(agent_static, left_on=ticker_col, right_on="__ticker_key", how="left")
            X = X.drop(columns=["__ticker_key"], errors="ignore")
            after = set(X.columns)
            ctx.logger.info(f"[ModelInferenceAgent] JOIN agent_json_features: +{len(after - before)} cols")
        else:
            ctx.logger.info("[ModelInferenceAgent] SKIP agent_json_features: none")

        X = self._normalize_known_categorical_features(X)

        excluded = {
            ticker_col,
            date_col,
            close_col,
            "return_20d",
            "rank_20d",
            "rank_percentile_20d",
            "top_quantile_20d",
            "downside_20d",
        }
        feature_cols = [c for c in X.columns if c not in excluded]

        ctx.logger.info(f"[ModelInferenceAgent] Dataset rows={len(X)}, features={len(feature_cols)}")

        return {
            "X": X,
            "feature_cols": feature_cols,
            "ticker_col": ticker_col,
            "date_col": date_col,
            "close_col": close_col,
        }

    # =========================================================
    # Model fit/predict
    # =========================================================
    def _fit_predict_model(
        self,
        model_id: str,
        target: str,
        X: pd.DataFrame,
        feature_cols: List[str],
        date_col: str,
        train_mask: pd.Series,
        test_mask: pd.Series,
        ctx: RunContext,
    ) -> Tuple[np.ndarray, Any, str]:
        if target in ["top_quantile_20d", "downside_20d"]:
            y_col = target
        elif target == "rank_percentile_20d":
            y_col = "rank_percentile_20d"
        elif target == "rank_20d":
            # Regression-style rank proxy models can use continuous percentile labels.
            # LGBMRanker itself will receive discrete integer relevance labels below.
            y_col = "rank_percentile_20d"
        else:
            raise ValueError(f"Unsupported target: {target}")

        train_df = X.loc[train_mask].copy()
        test_df = X.loc[test_mask].copy()

        # Sort training data by date so LGBMRanker group sizes align with rows.
        train_df = train_df.sort_values(date_col).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        X_train = train_df[feature_cols]
        y_train = train_df[y_col]
        X_test = test_df[feature_cols]

        task_type, model, family = self._make_model(model_id, X_train)

        if task_type == "ranking":
            # LightGBM Ranker requires integer relevance labels.
            # Convert same-date future return into 5-level relevance labels:
            # 0 = weakest future return group, 4 = strongest future return group.
            y_train_rank = train_df.groupby(date_col)["return_20d"].rank(
                method="first",
                pct=True,
                ascending=True,
            )
            y_train_rank = np.floor(y_train_rank * 5).clip(0, 4).astype(int)

            group = train_df.groupby(date_col, sort=False).size().tolist()
            model.fit(X_train, y_train_rank, model__group=group)
            pred = model.predict(X_test)
            return np.asarray(pred), model, family

        if task_type == "regression":
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            return np.asarray(pred), model, family

        if task_type == "classification":
            model.fit(X_train, y_train.astype(int))
            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Model {model_id} does not support predict_proba.")
            proba = model.predict_proba(X_test)[:, 1]
            return np.asarray(proba), model, family

        raise ValueError(f"Unsupported task_type: {task_type}")

    def _make_model(self, model_id: str, sample_X: pd.DataFrame) -> Tuple[str, Pipeline, str]:
        preprocessor = self._make_preprocessor(sample_X)

        if model_id == "lgbm_ranker_v1":
            LGBMRanker = self._require_lgbm_ranker()
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            return "ranking", Pipeline([("prep", preprocessor), ("model", model)]), "LightGBM"

        if model_id == "lgbm_regressor_rank_proxy_v1":
            LGBMRegressor, _ = self._require_lgbm_models()
            model = LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            return "regression", Pipeline([("prep", preprocessor), ("model", model)]), "LightGBM"

        if model_id == "ridge_rank_proxy_baseline":
            model = Ridge(alpha=1.0)
            return "regression", Pipeline([("prep", preprocessor), ("model", model)]), "LinearModel"

        if model_id == "lgbm_rank_percentile_regressor_v1":
            LGBMRegressor, _ = self._require_lgbm_models()
            model = LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            return "regression", Pipeline([("prep", preprocessor), ("model", model)]), "LightGBM"

        if model_id == "ridge_rank_percentile_baseline":
            model = Ridge(alpha=1.0)
            return "regression", Pipeline([("prep", preprocessor), ("model", model)]), "LinearModel"

        if model_id == "lgbm_top_quantile_classifier_v1":
            _, LGBMClassifier = self._require_lgbm_models()
            model = LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
            )
            return "classification", Pipeline([("prep", preprocessor), ("model", model)]), "LightGBM"

        if model_id == "logistic_top_quantile_baseline":
            model = LogisticRegression(max_iter=3000, class_weight="balanced")
            return "classification", Pipeline([("prep", preprocessor), ("model", model)]), "LinearModel"

        if model_id == "lgbm_downside_classifier_v1":
            _, LGBMClassifier = self._require_lgbm_models()
            model = LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
            )
            return "classification", Pipeline([("prep", preprocessor), ("model", model)]), "LightGBM"

        if model_id == "logistic_downside_baseline":
            model = LogisticRegression(max_iter=3000, class_weight="balanced")
            return "classification", Pipeline([("prep", preprocessor), ("model", model)]), "LinearModel"

        raise ValueError(f"Unknown model_id from model_candidates.json: {model_id}")

    def _make_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )

        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ],
            remainder="drop",
        )

    # =========================================================
    # Personalized score
    # =========================================================
    def _build_personalized_scores(
        self,
        ctx: RunContext,
        X: pd.DataFrame,
        predictions: pd.DataFrame,
        paths: Dict[str, Optional[Path]],
        candidate_map: Dict[str, Any],
        selected: Dict[str, Optional[str]],
        artifacts: ModelInferenceArtifacts,
    ) -> Optional[pd.DataFrame]:
        score_models = [m.get("model_id") for m in candidate_map.get("personalized_ranking_score", [])]
        if "rule_based_personalized_rank_blend_v1" not in score_models:
            ctx.logger.info("[ModelInferenceAgent] SKIP personalized score: candidate not listed.")
            return None

        if paths.get("user") is None or not paths["user"].exists():
            ctx.logger.info("[ModelInferenceAgent] SKIP personalized score: user snapshot missing.")
            return None

        user = self._read_csv(paths["user"])
        uid = self._infer_col(user, ["user_id", "userid", "uid", "사용자ID", "사용자_id"])
        if uid is None:
            ctx.logger.warning("[ModelInferenceAgent] SKIP personalized score: no user_id column.")
            return None

        user[uid] = user[uid].astype(str)

        if "risk_score" in user.columns:
            risk = pd.to_numeric(user["risk_score"], errors="coerce")
            if risk.min(skipna=True) >= 0 and risk.max(skipna=True) <= 1:
                user["w_user"] = risk.clip(0, 1)
            else:
                user["w_user"] = risk.rank(pct=True).clip(0, 1)
        else:
            user["w_user"] = 0.55

        rank_model = selected.get("rank_20d") or selected.get("rank_percentile_20d")
        top_model = selected.get("top_quantile_20d")
        downside_model = selected.get("downside_20d")

        if rank_model is None or downside_model is None:
            ctx.logger.warning("[ModelInferenceAgent] SKIP personalized score: missing selected ranking/downside model.")
            return None

        rank_col = self._find_prediction_col(predictions, rank_model)
        top_col = self._find_prediction_col(predictions, top_model) if top_model else None
        downside_col = self._find_prediction_col(predictions, downside_model)

        if rank_col is None or downside_col is None:
            ctx.logger.warning("[ModelInferenceAgent] SKIP personalized score: required prediction columns missing.")
            return None

        last_date = pd.to_datetime(predictions["date"]).max()
        base = predictions[pd.to_datetime(predictions["date"]) == last_date].copy()

        out = base[["ticker", "date", rank_col, downside_col]].copy()
        out = out.rename(
            columns={
                rank_col: "pred_rank_score_20d",
                downside_col: "P_downside_20d",
            }
        )

        if top_col is not None:
            out["P_top_quantile_20d"] = base[top_col].values
        else:
            out["P_top_quantile_20d"] = 0.0

        # Normalize rank score inside latest cross-section.
        out["rank_score_norm"] = pd.to_numeric(out["pred_rank_score_20d"], errors="coerce").rank(pct=True)

        user2 = user[[uid, "w_user"]].copy()
        user2["key"] = 1
        out["key"] = 1

        pers = user2.merge(out, on="key", how="outer").drop(columns=["key"])
        pers = pers.rename(columns={uid: "user_id", "date": "as_of_date"})

        alpha = 0.25
        beta = 0.50
        pers["personalized_ranking_score"] = (
            pers["w_user"] * pers["rank_score_norm"]
            + alpha * pers["P_top_quantile_20d"]
            - beta * pers["P_downside_20d"]
        )

        pers = pers.sort_values(["user_id", "personalized_ranking_score"], ascending=[True, False])
        pers.to_csv(artifacts.personalized_scores_path, index=False, encoding="utf-8-sig")

        ctx.logger.info(f"[ModelInferenceAgent] Saved personalized scores: {artifacts.personalized_scores_path}")
        return pers

    # =========================================================
    # Metrics
    # =========================================================
    def _ranking_metrics(
        self,
        df: pd.DataFrame,
        score_col: str,
        true_return_col: str,
        date_col: str,
        ks: List[int],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        ic_values = []
        ndcg_values = {k: [] for k in ks}
        precision_values = {k: [] for k in ks}
        hit_values = {k: [] for k in ks}

        for _, g in df.groupby(date_col):
            g = g.dropna(subset=[score_col, true_return_col]).copy()
            if len(g) < 3 or g[score_col].nunique() < 2 or g[true_return_col].nunique() < 2:
                continue

            ic, _ = spearmanr(g[true_return_col], g[score_col])
            if not np.isnan(ic):
                ic_values.append(float(ic))

            y_true = g[true_return_col].rank(pct=True).values.reshape(1, -1)
            y_score = g[score_col].values.reshape(1, -1)

            for k in ks:
                kk = min(k, len(g))
                try:
                    ndcg_values[k].append(float(ndcg_score(y_true, y_score, k=kk)))
                except Exception:
                    pass

                actual_top = set(g.nlargest(kk, true_return_col).index)
                pred_top = set(g.nlargest(kk, score_col).index)
                overlap = len(actual_top & pred_top)
                precision_values[k].append(float(overlap / kk))
                hit_values[k].append(float(1.0 if overlap > 0 else 0.0))

        out["SpearmanRankIC"] = self._mean_or_none(ic_values)
        for k in ks:
            out[f"NDCG@{k}"] = self._mean_or_none(ndcg_values[k])
            out[f"Precision@{k}"] = self._mean_or_none(precision_values[k])
            out[f"TopK_HitRate@{k}"] = self._mean_or_none(hit_values[k])

        return out

    def _classification_metrics(self, y_true: pd.Series, proba: pd.Series) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true).astype(int)
        proba_arr = np.asarray(proba).astype(float)
        y_hat = (proba_arr >= 0.5).astype(int)

        return {
            "AUC": self._safe_auc(y_true_arr, proba_arr),
            "Recall": float(recall_score(y_true_arr, y_hat, zero_division=0)),
            "Precision": float(precision_score(y_true_arr, y_hat, zero_division=0)),
            "F1": float(f1_score(y_true_arr, y_hat, zero_division=0)),
            "PositiveRate": float(np.mean(y_true_arr)) if len(y_true_arr) else None,
        }

    def _select_best_models(self, results: Dict[str, Any]) -> Dict[str, Optional[str]]:
        selected: Dict[str, Optional[str]] = {
            "rank_20d": None,
            "rank_percentile_20d": None,
            "top_quantile_20d": None,
            "downside_20d": None,
        }

        for target in ["rank_20d", "rank_percentile_20d"]:
            best_model = None
            best_score = -1e18
            for model_id, metrics in results.get(target, {}).items():
                score = metrics.get("SpearmanRankIC")
                if score is not None and score > best_score:
                    best_model = model_id
                    best_score = score
            selected[target] = best_model

        for target in ["top_quantile_20d", "downside_20d"]:
            best_model = None
            best_score = -1e18
            for model_id, metrics in results.get(target, {}).items():
                score = metrics.get("AUC")
                if score is not None and score > best_score:
                    best_model = model_id
                    best_score = score
            selected[target] = best_model

        return selected

    # =========================================================
    # Join helpers
    # =========================================================
    def _join_ticker_level(
        self,
        X: pd.DataFrame,
        left_ticker_col: str,
        path: Optional[Path],
        cols: List[str],
        name: str,
        ctx: RunContext,
        latest_by_date: bool = False,
        latest_by_year: bool = False,
    ) -> pd.DataFrame:
        if path is None or not path.exists():
            ctx.logger.info(f"[ModelInferenceAgent] SKIP {name}: missing")
            return X

        src = self._read_csv(path)
        tc = self._infer_col(src, ["종목코드", "ticker", "code", "symbol"])
        if tc is None:
            ctx.logger.info(f"[ModelInferenceAgent] SKIP {name}: no ticker col")
            return X

        src[tc] = src[tc].apply(self._zfill6)

        if latest_by_date:
            dc = self._infer_col(src, ["date", "datetime", "dt", "일자", "날짜"])
            if dc:
                src[dc] = pd.to_datetime(src[dc], errors="coerce")
                src = src.dropna(subset=[dc]).sort_values([tc, dc]).groupby(tc, as_index=False).tail(1)

        if latest_by_year:
            yc = self._infer_col(src, ["year", "년도"])
            if yc:
                src = src.sort_values([tc, yc]).groupby(tc, as_index=False).tail(1)

        keep = [tc] + [c for c in cols if c in src.columns and c != tc]
        src = src[keep].copy()

        before = set(X.columns)
        X = X.merge(src, left_on=left_ticker_col, right_on=tc, how="left")
        if tc != left_ticker_col:
            X = X.drop(columns=[tc], errors="ignore")
        after = set(X.columns)

        ctx.logger.info(f"[ModelInferenceAgent] JOIN {name}: +{len(after - before)} cols")
        return X

    def _build_agent_static_features(
        self,
        feature_contract: Dict[str, Any],
        paths: Dict[str, Optional[Path]],
    ) -> Optional[pd.DataFrame]:
        aj = feature_contract.get("feature_groups", {}).get("agent_json_features", {})

        def load_ticker_rows(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
            if path is None or not path.exists():
                return {}
            j = self._read_json(path)
            rows = j.get("tickers", []) if isinstance(j, dict) else []
            out = {}
            for row in rows:
                ticker = row.get("ticker")
                if ticker:
                    out[self._zfill6(ticker)] = row
            return out

        market_map = load_ticker_rows(paths.get("market_json"))
        news_map = load_ticker_rows(paths.get("news_json"))
        risk_map = load_ticker_rows(paths.get("risk_json"))

        tickers = sorted(set(market_map) | set(news_map) | set(risk_map))
        if not tickers:
            return None

        cols = []
        cols += aj.get("market_analysis", [])
        cols += aj.get("news_invest", [])
        cols += aj.get("risk_score", [])
        cols = list(dict.fromkeys(cols))

        records = []
        for ticker in tickers:
            row = {"종목코드": ticker}
            sources = [market_map.get(ticker, {}), news_map.get(ticker, {}), risk_map.get(ticker, {})]
            for col in cols:
                value = None
                for source in sources:
                    value = self._flatten_path_get(source, col)
                    if value is not None:
                        break
                row[col] = value
            records.append(row)

        return pd.DataFrame(records)

    def _normalize_known_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if "confidence_level" in X.columns:
            X["confidence_level_num"] = (
                X["confidence_level"].astype(str).str.lower().map({"low": 0.0, "medium": 0.5, "high": 1.0})
            )
            X = X.drop(columns=["confidence_level"])

        if "risk_scores.risk_level" in X.columns:
            X["risk_level_num"] = (
                X["risk_scores.risk_level"]
                .astype(str)
                .str.lower()
                .map({"low": 0.0, "medium": 1.0, "high": 2.0, "critical": 3.0})
            )
            X = X.drop(columns=["risk_scores.risk_level"])

        return X

    # =========================================================
    # Split / target spec helpers
    # =========================================================
    def _make_time_split(
        self,
        X: pd.DataFrame,
        date_col: str,
        split_params: Dict[str, int],
        ctx: RunContext,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        val_days = split_params["validation_window_days"]
        test_days = split_params["test_window_days"]
        embargo = split_params["embargo_days"]

        all_dates = pd.Series(sorted(pd.to_datetime(X[date_col]).dropna().unique()))

        if len(all_dates) < val_days + test_days + embargo + 10:
            ctx.logger.warning("[ModelInferenceAgent] Not enough dates. Using last 20% as test.")
            cutoff = all_dates.iloc[int(len(all_dates) * 0.8)]
            train_mask = pd.to_datetime(X[date_col]) <= cutoff
            val_mask = pd.Series(False, index=X.index)
            test_mask = pd.to_datetime(X[date_col]) > cutoff
            return train_mask, val_mask, test_mask

        test_start = all_dates.iloc[-test_days]
        test_end = all_dates.iloc[-1]
        val_start = all_dates.iloc[-test_days - embargo - val_days]
        val_end = all_dates.iloc[-test_days - embargo - 1]

        train_end_idx = len(all_dates) - (test_days + embargo + val_days + embargo) - 1
        train_end = all_dates.iloc[max(train_end_idx, 0)]

        dates = pd.to_datetime(X[date_col])
        train_mask = dates <= train_end
        val_mask = (dates >= val_start) & (dates <= val_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        ctx.logger.info(
            f"[ModelInferenceAgent] Split rows train/val/test: "
            f"{int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}"
        )

        return train_mask, val_mask, test_mask

    def _get_split_params(self, evaluation_plan: Dict[str, Any]) -> Dict[str, int]:
        params = {
            "train_window_days": 252,
            "validation_window_days": 60,
            "test_window_days": 60,
            "embargo_days": 5,
        }
        suggested = evaluation_plan.get("data_split", {}).get("suggested", {})
        for key in params:
            if key in suggested and isinstance(suggested[key], (int, float)):
                params[key] = int(suggested[key])
        return params

    def _get_horizon(self, target_spec: Dict[str, Any]) -> int:
        for item in target_spec.get("targets", []):
            if item.get("target_id") == "rank_20d":
                return int(item.get("time_horizon_trading_days", 20))
        return 20

    def _get_downside_threshold(self, target_spec: Dict[str, Any]) -> float:
        for item in target_spec.get("targets", []):
            if item.get("target_id") == "downside_20d":
                return float(item.get("threshold", -0.10))
        return -0.10

    def _get_top_quantile(self, target_spec: Dict[str, Any]) -> float:
        for item in target_spec.get("targets", []):
            if item.get("target_id") == "top_quantile_20d":
                return float(item.get("top_quantile", 0.20))
        return 0.20

    # =========================================================
    # File / json / csv helpers
    # =========================================================
    def _find_single_file(self, run_dir: Path, pattern: str, ctx: RunContext) -> Optional[Path]:
        matches = sorted(run_dir.rglob(pattern))
        if not matches:
            ctx.logger.info(f"[ModelInferenceAgent] No file matched: {pattern}")
            return None
        if len(matches) > 1:
            ctx.logger.warning(f"[ModelInferenceAgent] Multiple files matched: {pattern}. Using first: {matches[0]}")
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

    def _read_csv(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_model(model: Any, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(model, f)

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

    @staticmethod
    def _future_min_window(series: pd.Series, window: int) -> pd.Series:
        s = series.astype(float)
        s_rev = s.iloc[::-1]
        min_rev = s_rev.rolling(window=window, min_periods=window).min()
        return min_rev.iloc[::-1]

    @staticmethod
    def _flatten_path_get(d: Dict[str, Any], path: str) -> Any:
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    @staticmethod
    def _mean_or_none(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.mean(values))

    @staticmethod
    def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
        try:
            if len(np.unique(y_true)) < 2:
                return None
            return float(roc_auc_score(y_true, proba))
        except Exception:
            return None

    @staticmethod
    def _safe_mae(y_true: pd.Series, y_pred: pd.Series) -> Optional[float]:
        try:
            return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
        except Exception:
            return None

    @staticmethod
    def _find_prediction_col(predictions: pd.DataFrame, model_id: Optional[str]) -> Optional[str]:
        if model_id is None:
            return None
        matches = [c for c in predictions.columns if c.endswith(f"__{model_id}")]
        return matches[0] if matches else None

    def _extract_feature_importance(self, model: Any, feature_cols: List[str]) -> List[Dict[str, Any]]:
        try:
            estimator = model.named_steps.get("model")
        except Exception:
            estimator = None

        if estimator is None:
            return []

        if hasattr(estimator, "feature_importances_"):
            # After one-hot encoding, transformed feature count may differ from original feature count.
            importance = list(np.ravel(estimator.feature_importances_))
            return [
                {"feature": f"transformed_feature_{i}", "importance": float(v)}
                for i, v in sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:300]
            ]

        if hasattr(estimator, "coef_"):
            coef = list(np.ravel(estimator.coef_))
            return [
                {"feature": f"transformed_feature_{i}", "importance": float(v)}
                for i, v in sorted(enumerate(coef), key=lambda x: abs(x[1]), reverse=True)[:300]
            ]

        return []

    def _require_lgbm_models(self):
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            return LGBMRegressor, LGBMClassifier
        except Exception as e:
            raise ImportError(
                "LightGBM is required for the selected model candidates. "
                "Install it with: pip install lightgbm. "
                f"Original error: {e}"
            )

    def _require_lgbm_ranker(self):
        try:
            from lightgbm import LGBMRanker

            return LGBMRanker
        except Exception as e:
            raise ImportError(
                "LightGBM is required for lgbm_ranker_v1. "
                "Install it with: pip install lightgbm. "
                f"Original error: {e}"
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

        kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
        return StageResult(**kwargs)
