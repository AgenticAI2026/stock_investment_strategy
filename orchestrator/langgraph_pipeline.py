from __future__ import annotations

from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import TypedDict

from core.context import RunContext
from core.artifacts import ArtifactPaths

# agents
from agents.ingest.agent import DataIngestionAgent
from agents.feature_table.agent import FeatureExtractionAgent
# from agents.prep_reco.agent import PreprocessingRecommenderAgent
# from agents.model_match_v1.agent import ModelMatchV1Agent
# from agents.prep_apply_v1.agent import PreprocessingImplementorV1Agent
# from agents.market_rag.agent import MarketRagAgent
# from agents.news_verify_rag.agent import NewsVerifyRagAgent
# from agents.risk_score.agent import RiskScoreAgent
# from agents.model_match_v2.agent import ModelMatchV2Agent
# from agents.infer.agent import InferenceAgent
# from agents.report_gen.agent import ReportGenAgent
# from agents.report_validate.agent import ReportValidateAgent
# from agents.shortform_rag.agent import ShortformRagAgent

class PipelineState(TypedDict, total=False):
    ctx: RunContext
    ap: ArtifactPaths
    status: str
    stage_outputs: dict


def run_ingest(state: PipelineState) -> PipelineState:
    ctx = state["ctx"]
    ap = state["ap"]

    agent = DataIngestionAgent(
                user_csv_map={
                    "event": "user_data/user_event_log.csv",
                    "profile": "user_data/user_profile.csv",
                }
            )
    result = agent.execute(ctx, ap)

    if result.status != "success":
        raise RuntimeError(f"Ingest failed: {result.metrics or result}")

    ctx.logger.info("⭐ Ingest DONE")

    return {
        **state,
        "status": result.status,
    }


def run_feature_table(state: PipelineState) -> PipelineState:
    ctx = state["ctx"]
    ap = state["ap"]

    agent = FeatureExtractionAgent()
    result = agent.execute(ctx, ap)
    
    if result.status != "success":
        raise RuntimeError(f"Feature Extract failed: {getattr(result, 'error', None)}")

    ctx.logger.info("⭐ Feature Extract DONE")

    return {
        **state,
        "status": result.status,
        "stage_outputs": {
            **state.get("stage_outputs", {}),
            "feature": {
                "summary": getattr(result, "summary", None),
                "artifacts": getattr(result, "artifacts", None),
            }
        }
    }

# -----------------------
# Build Graph
# -----------------------
def build_pipeline():

    graph = StateGraph(PipelineState)

    # nodes
    graph.add_node("ingest",run_ingest)
    graph.add_node("feature_table", run_feature_table)
    # graph.add_node("prep_reco", run_agent(PreprocessingRecommenderAgent()))
    # graph.add_node("model_match_v1", run_agent(ModelMatchV1Agent()))
    # graph.add_node("prep_apply_v1", run_agent(PreprocessingImplementorV1Agent()))
    # graph.add_node("market_rag", run_agent(MarketRagAgent()))
    # graph.add_node("news_verify_rag", run_agent(NewsVerifyRagAgent()))
    # graph.add_node("risk_score", run_agent(RiskScoreAgent()))
    # graph.add_node("model_match_v2", run_agent(ModelMatchV2Agent()))
    # graph.add_node("infer", run_agent(InferenceAgent()))
    # graph.add_node("report_gen", run_agent(ReportGenAgent()))
    # graph.add_node("report_validate", run_agent(ReportValidateAgent()))
    # graph.add_node("shortform_rag", run_agent(ShortformRagAgent()))

    # edges
    graph.set_entry_point("ingest")

    graph.add_edge("ingest", "feature_table")
    # graph.add_edge("feature_table", "prep_reco")
    # graph.add_edge("prep_reco", "model_match_v1")
    # graph.add_edge("model_match_v1", "prep_apply_v1")
    # graph.add_edge("prep_apply_v1", "market_rag")
    # graph.add_edge("market_rag", "news_verify_rag")
    # graph.add_edge("news_verify_rag", "risk_score")
    # graph.add_edge("risk_score", "model_match_v2")
    # graph.add_edge("model_match_v2", "infer")
    # graph.add_edge("infer", "report_gen")
    # graph.add_edge("report_gen", "report_validate")
    # graph.add_edge("report_validate", "shortform_rag")

    # graph.add_edge("shortform_rag", END)
    graph.add_edge("feature_table", END)

    return graph.compile()

def make_initial_state(
    project_root: str = ".",
    asof_date: str = "2026-03-18",
    universe: str = "KR_TOP100_LIQUIDITY",
) -> PipelineState:
    ctx = RunContext.create(
        project_root=Path(project_root),
        asof_date=asof_date,
        universe=universe,
    )
    ap = ArtifactPaths(ctx.artifact_root)

    return {
        "ctx": ctx,
        "ap": ap,
        "status": "initialized",
    }


if __name__ == "__main__":
    app = build_pipeline()
    state = make_initial_state()

    final_state = app.invoke(state)

    print("\n=== PIPELINE DONE ===")
    print(f"run_id       : {final_state['ctx'].run_id}")
    print(f"artifact_root: {final_state['ctx'].artifact_root}")
    print(f"status       : {final_state['status']}")