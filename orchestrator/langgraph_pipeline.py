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
from agents.market_analysis.agent import MarketAnalysisAgent
from agents.news_invest.agent import NewsInvestigationAgent
from agents.risk_score.agent import RiskScoreAgent
from agents.market_flow.agent import MarketFlowAgent
from agents.report_target_planner.agent import ReportTargetPlannerAgent


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

def run_market_analysis(state: PipelineState) -> PipelineState:
    agent = MarketAnalysisAgent()
    return _run_stage(state, "market_analysis", agent)

def run_news_invest(state: PipelineState) -> PipelineState:
    agent = NewsInvestigationAgent()
    return _run_stage(state, "news_invest", agent)

def run_risk_score(state: PipelineState) -> PipelineState:
    agent = RiskScoreAgent()
    return _run_stage(state, "risk_score", agent)

def run_report_target_planner(state: PipelineState) -> PipelineState:
    agent = ReportTargetPlannerAgent()
    return _run_stage(state, "report_target_planner", agent)

def run_market_flow(state: PipelineState) -> PipelineState:
    agent = MarketFlowAgent()
    return _run_stage(state, "market_flow", agent)

# def run_model_match_v2(state: PipelineState) -> PipelineState:
#     agent = ModelTargetMatcherAgent()
#     return _run_stage(state, "model_match_v2", agent)

# def run_model_infer(state: PipelineState) -> PipelineState:
#     agent = ModelInferenceAgent()
#     return _run_stage(state, "model_infer", agent)

# def run_report_gen(state: PipelineState) -> PipelineState:
#     agent = ReportGenerativeAgent()
#     return _run_stage(state, "report_gen", agent)

def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", run_ingest)
    graph.add_node("feature_table", run_feature_table)
    graph.add_node("prep_reco", run_prep_reco)
    graph.add_node("model_match_v1", run_model_match_v1)
    graph.add_node("prep_apply", run_prep_apply)
    graph.add_node("market_analysis", run_market_analysis)
    graph.add_node("news_invest", run_news_invest)
    graph.add_node("risk_score", run_risk_score)
    graph.add_node("report_target_planner", run_report_target_planner)
    graph.add_node("market_flow", run_market_flow)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "feature_table")
    graph.add_edge("feature_table", "prep_reco")
    graph.add_edge("prep_reco", "model_match_v1")
    graph.add_edge("model_match_v1", "prep_apply")
    graph.add_edge("prep_apply", "market_analysis")
    graph.add_edge("market_analysis", "news_invest")
    graph.add_edge("news_invest", "risk_score")
    graph.add_edge("risk_score", "report_target_planner")
    graph.add_edge("report_target_planner", "market_flow")
    graph.add_edge("market_flow", END)

    return graph.compile()