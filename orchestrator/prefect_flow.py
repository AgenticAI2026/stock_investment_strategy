from pathlib import Path
from prefect import flow

from core.context import RunContext
from core.artifacts import ArtifactPaths

from orchestrator.langgraph_pipeline import build_pipeline


@flow
def daily_run(asof_date: str, universe: str = "KOSPI100"):

    project_root = Path(__file__).resolve().parents[1]

    # create run context
    ctx = RunContext.create(
        project_root=project_root,
        asof_date=asof_date,
        universe=universe
    )

    ap = ArtifactPaths(ctx.artifact_root)

    # build graph
    pipeline = build_pipeline()

    state = {
        "ctx": ctx,
        "ap": ap,
        "status": "start"
    }

    result = pipeline.invoke(state)

    print("PIPELINE FINISHED:", result)

    return result