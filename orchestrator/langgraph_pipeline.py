from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from core.context import RunContext
from core.artifacts import ArtifactPaths

from agents.ingest.agent import DataIngestionAgent
from agents.feature_table.agent import FeatureExtractionAgent
from agents.prep_reco.agent import PreprocessingRecommenderAgent
from agents.model_match_v1.agent import ModelDataMatcherAgent
from agents.prep_apply.agent import PreprocessingImplementorAgent


class PipelineState(TypedDict, total=False):
    ctx: RunContext
    ap: ArtifactPaths
    status: str
    stage_outputs: dict[str, Any]
    error_stage: str
    error_message: str


def _normalize_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return {
            "status": result.get("status"),
            "error": result.get("error"),
            "outputs": result.get("outputs"),
            "details": result.get("details"),
        }

    return {
        "status": getattr(result, "status", None),
        "error": getattr(result, "error", None),
        "outputs": getattr(result, "outputs", None),
        "details": getattr(result, "details", None),
    }


def _run_stage(state: PipelineState, stage_name: str, agent: Any) -> PipelineState:
    ctx = state["ctx"]
    ap = state["ap"]

    ctx.logger.info(f"🚀 {stage_name} start")

    result = agent.execute(ctx, ap)
    normalized = _normalize_result(result)

    if normalized["status"] != "success":
        error_message = normalized["error"] or f"{stage_name} failed"
        ctx.logger.error(f"❌ {stage_name} failed: {error_message}")
        raise RuntimeError(f"{stage_name} failed: {error_message}")

    ctx.logger.info(f"✅ {stage_name} done")

    return {
        **state,
        "status": "success",
        "stage_outputs": {
            **state.get("stage_outputs", {}),
            stage_name: {
                "outputs": normalized["outputs"],
                "details": normalized["details"],
            },
        },
    }


def run_ingest(state: PipelineState) -> PipelineState:
    agent = DataIngestionAgent(
        user_csv_map={
            "event": "user_data/user_event_log.csv",
            "profile": "user_data/user_profile.csv",
        }
    )
    return _run_stage(state, "ingest", agent)

def run_feature_table(state: PipelineState) -> PipelineState:
    agent = FeatureExtractionAgent()
    return _run_stage(state, "feature_table", agent)


def run_prep_reco(state: PipelineState) -> PipelineState:
    agent = PreprocessingRecommenderAgent()
    return _run_stage(state, "prep_reco", agent)

def run_model_match_v1(state: PipelineState) -> PipelineState:
    agent = ModelDataMatcherAgent()
    return _run_stage(state, "model_match_v1", agent)

def run_prep_apply(state: PipelineState) -> PipelineState:
    agent = PreprocessingImplementorAgent()
    return _run_stage(state, "prep_apply", agent)


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", run_ingest)
    graph.add_node("feature_table", run_feature_table)
    graph.add_node("prep_reco", run_prep_reco)
    graph.add_node("model_match_v1", run_model_match_v1)
    graph.add_node("prep_apply", run_prep_apply)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "feature_table")
    graph.add_edge("feature_table", "prep_reco")
    graph.add_edge("prep_reco", "model_match_v1")
    graph.add_edge("model_match_v1", "prep_apply")
    graph.add_edge("prep_apply", END)

    return graph.compile()