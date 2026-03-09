# 모든 agent 실행

from orchestrator.prefect_flow import daily_run

if __name__ == "__main__":

    daily_run(
        asof_date="2026-03-10",
        universe="KOSPI100"
    )