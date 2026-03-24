# 모든 agent 실행
import argparse
from orchestrator.prefect_flow import daily_run


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-03-23")
    parser.add_argument("--universe", default="KOSPI100")

    args = parser.parse_args()

    daily_run(
        asof_date=args.date,
        universe=args.universe
    )


if __name__ == "__main__":
    main()