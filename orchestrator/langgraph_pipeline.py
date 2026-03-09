from langgraph.graph import StateGraph, END
from typing import TypedDict

from core.context import RunContext
from core.artifacts import ArtifactPaths

# agents
from agents.ingest.agent import IngestAgent
# from agents.feature_table.agent import FeatureTableAgent
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


# -----------------------
# Graph State
# -----------------------

class PipelineState(TypedDict):
    ctx: RunContext
    ap: ArtifactPaths
    status: str


# -----------------------
# Agent Wrapper
# -----------------------

def run_agent(agent):

    def node(state: PipelineState):

        ctx = state["ctx"]
        ap = state["ap"]

        result = agent.execute(ctx, ap)

        if result.status == "failed":
            return {"status": "failed"}

        return {"status": "success"}

    return node


# -----------------------
# Build Graph
# -----------------------
def build_pipeline():

    graph = StateGraph(PipelineState)

    # nodes
    graph.add_node("ingest", run_agent(IngestAgent()))
    # graph.add_node("feature_table", run_agent(FeatureTableAgent()))
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

    # graph.add_edge("ingest", "feature_table")
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
    graph.add_edge("ingest", END)

    return graph.compile()