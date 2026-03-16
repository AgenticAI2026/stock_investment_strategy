import pandas as pd
import numpy as np
from pathlib import Path
import re


class FeatureExtractionAgent:

    # ============================================================
    # 1. FINANCE FEATURES
    # ============================================================

    def run_finance_features(self, ingest_dir, feature_dir):

        input_dir = ingest_dir / "finance"
        output_dir = feature_dir / "finance"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.csv"))

        print(f"📂 Finance files: {len(files)}")

        for f in files:

            try:
                df = pd.read_csv(f)

                if "revenue" in df.columns and "net_income" in df.columns:
                    df["profit_margin"] = df["net_income"] / df["revenue"]

                if "assets" in df.columns and "liabilities" in df.columns:
                    df["debt_ratio"] = df["liabilities"] / df["assets"]

                out_name = f.name.replace("_financials_", "_financial_features_")
                out_path = output_dir / out_name

                df.to_csv(out_path, index=False)

                print("☑️", out_name)

            except Exception as e:
                print("❌", f.name, e)

        feature_dict = pd.DataFrame([
            {
                "feature_name": "year",
                "description_ko": "연도 (시간축 기준 컬럼)",
                "formula": "",
                "window": "time_col",
                "notes": "role=time_col, semantic_type=year/datetime, model_use=false"
            },
            {
                "feature_name": "종목코드",
                "description_ko": "종목코드 (조인 키, 6자리 문자열)",
                "formula": "",
                "window": "",
                "notes": "role=join_key, semantic_type=ticker, model_use=false"
            },
            {
                "feature_name": "_source_file",
                "description_ko": "원본 파일명/출처 (concat provenance)",
                "formula": "",
                "window": "",
                "notes": "role=provenance, model_use=false (기업별 파일 concat 시 생길 수 있음)"
            },
            {
                "feature_name": "revenue_yoy",
                "description_ko": "매출액 전년 대비 성장률 (YoY)",
                "formula": "(금년 매출 − 전년 매출) / 전년 매출",
                "window": "연간 (t vs t-1)",
                "notes": "전년 매출이 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "operating_income",
                "description_ko": "영업이익률",
                "formula": "영업이익 / 매출액",
                "window": "연간",
                "notes": "매출액이 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "net_income",
                "description_ko": "순이익률",
                "formula": "당기순이익 / 매출액",
                "window": "연간",
                "notes": "매출액이 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "operating_income_yoy",
                "description_ko": "영업이익 전년 대비 성장률 (YoY)",
                "formula": "(금년 영업이익 − 전년 영업이익) / 전년 영업이익",
                "window": "연간 (t vs t-1)",
                "notes": "전년 영업이익이 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "debt_ratio",
                "description_ko": "부채비율",
                "formula": "부채총계 / 자본총계",
                "window": "연간 (BS)",
                "notes": "자본총계가 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "current_ratio",
                "description_ko": "유동비율",
                "formula": "유동자산 / 유동부채",
                "window": "연간 (BS)",
                "notes": "유동부채가 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "roe",
                "description_ko": "자기자본이익률 (ROE)",
                "formula": "당기순이익 / 자본총계",
                "window": "연간",
                "notes": "자본총계가 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "roa",
                "description_ko": "총자산이익률 (ROA)",
                "formula": "당기순이익 / 자산총계",
                "window": "연간",
                "notes": "자산총계가 0 또는 결측이면 NaN"
            },
            {
                "feature_name": "asset_turnover",
                "description_ko": "총자산회전율",
                "formula": "매출액 / 자산총계",
                "window": "연간",
                "notes": "자산 활용 효율성 지표"
            },
            {
                "feature_name": "revenue_cagr_3y",
                "description_ko": "최근 3년 매출 연평균 성장률 (CAGR)",
                "formula": "(매출_t / 매출_t-3)^(1/3) − 1",
                "window": "3년",
                "notes": "3년치 매출이 모두 존재해야 계산 가능"
            },
            {
                "feature_name": "profit_volatility_3y",
                "description_ko": "최근 3년 영업이익 변동성 (표준편차)",
                "formula": "std(영업이익_amount, window=3)",
                "window": "3년",
                "notes": "이익 안정성 판단용 지표"
            },
            {
                "feature_name": "loss_year_count_5y",
                "description_ko": "최근 5년 중 순손실 발생 연도 수",
                "formula": "count(net_income_amount < 0, window=5)",
                "window": "5년",
                "notes": "지속적 적자 기업 리스크 측정"
            },
        ])
        dict_path = output_dir / "finance_feature_dictionary.csv"
        feature_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        print("☑️ finance_feature_dictionary.csv")

    # ============================================================
    # 2. PRICE FEATURES
    # ============================================================

    def run_price_features(self, ingest_dir, feature_dir):

        price_dir = ingest_dir / "price"
        output_dir = feature_dir / "price"
        output_dir.mkdir(parents=True, exist_ok=True)

        ohlcv_path = price_dir / "top100_ohlcv_last365.csv"

        df = pd.read_csv(ohlcv_path)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["종목코드", "date"])

        g = df.groupby("종목코드")

        df["prev_close"] = g["close"].shift(1)
        df["daily_return"] = (df["close"] - df["prev_close"]) / df["prev_close"]
        df["log_return"] = np.log(df["close"] / df["prev_close"])

        df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
        df["ma60"] = g["close"].transform(lambda x: x.rolling(60).mean())

        df["volatility_20"] = g["daily_return"].transform(
            lambda x: x.rolling(20).std()
        )

        out_path = output_dir / "price_features.csv"
        df.to_csv(out_path, index=False)

        print("☑️ price_features.csv")

        feature_dict = pd.DataFrame([
            {
                "feature_name": "date",
                "description_ko": "거래일자",
                "formula": "",
                "window": "1d",
                "notes": "time column"
            },
            {
                "feature_name": "종목코드",
                "description_ko": "종목코드",
                "formula": "",
                "window": "",
                "notes": "join key"
            },
            {
                "feature_name": "prev_close",
                "description_ko": "전일 종가",
                "formula": "close.shift(1) by 종목코드",
                "window": "1d lag",
                "notes": ""
            },
            {
                "feature_name": "daily_return",
                "description_ko": "일간 수익률",
                "formula": "(close - prev_close) / prev_close",
                "window": "1d",
                "notes": ""
            },
            {
                "feature_name": "log_return",
                "description_ko": "로그 수익률",
                "formula": "ln(close / prev_close)",
                "window": "1d",
                "notes": ""
            },
            {
                "feature_name": "ma20",
                "description_ko": "20일 이동평균",
                "formula": "rolling_mean(close, 20)",
                "window": "20d",
                "notes": ""
            },
            {
                "feature_name": "ma60",
                "description_ko": "60일 이동평균",
                "formula": "rolling_mean(close, 60)",
                "window": "60d",
                "notes": ""
            },
            {
                "feature_name": "volatility_20",
                "description_ko": "20일 수익률 변동성",
                "formula": "rolling_std(daily_return, 20)",
                "window": "20d",
                "notes": ""
            },
        ])

        dict_path = output_dir / "price_feature_dictionary.csv"
        feature_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        print("☑️ price_feature_dictionary.csv")

    # ============================================================
    # 3. NEWS FEATURES
    # ============================================================

    def run_news_features(self, ingest_dir, feature_dir):

        news_dir = ingest_dir / "news"
        output_dir = feature_dir / "news"
        output_dir.mkdir(parents=True, exist_ok=True)

        news_path = news_dir / "top100_naver_news_merged.csv"

        df = pd.read_csv(news_path)

        df["pubDate"] = pd.to_datetime(df["pubDate"], utc=True)

        now = df["pubDate"].max()
        cutoff = now - pd.Timedelta(hours=24)

        EVENT_KEYWORDS = ["실적", "인수", "합병", "규제", "소송", "수주"]
        NEG_KEYWORDS = ["적자", "하락", "급락", "소송", "리콜"]

        event_pat = re.compile("|".join(EVENT_KEYWORDS))
        neg_pat = re.compile("|".join(NEG_KEYWORDS))

        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

        def compute(g):

            recent_ratio = (g["pubDate"] >= cutoff).mean()
            event_ratio = g["text"].str.contains(event_pat).mean()
            neg_ratio = g["text"].str.contains(neg_pat).mean()

            return pd.Series(
                {
                    "recent_news_ratio": recent_ratio,
                    "event_news_ratio": event_ratio,
                    "negative_news_ratio": neg_ratio,
                }
            )

        features = (
            df.groupby("종목코드")
            .apply(compute, include_groups=False)
            .reset_index()
        )

        out_path = output_dir / "news_features.csv"
        features.to_csv(out_path, index=False)

        print("☑️ news_features.csv")

        feature_dict = pd.DataFrame([
            {
                "feature_name": "종목코드",
                "description_ko": "종목코드",
                "formula": "",
                "window": "",
                "notes": "join key"
            },
            {
                "feature_name": "recent_news_ratio",
                "description_ko": "최근 24시간 내 기사 비율",
                "formula": "count(pubDate >= now-24h) / N",
                "window": "24h",
                "notes": ""
            },
            {
                "feature_name": "event_news_ratio",
                "description_ko": "이벤트 키워드 포함 기사 비율",
                "formula": "count(text contains EVENT_KEYWORDS) / N",
                "window": "all news",
                "notes": ""
            },
            {
                "feature_name": "negative_news_ratio",
                "description_ko": "부정 키워드 포함 기사 비율",
                "formula": "count(text contains NEG_KEYWORDS) / N",
                "window": "all news",
                "notes": ""
            },
        ])

        dict_path = output_dir / "news_feature_dictionary.csv"
        feature_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        print("☑️ news_feature_dictionary.csv")

    # ============================================================
    # 4. USER FEATURES
    # ============================================================

    def run_user_features(self, ingest_dir, feature_dir):

        user_dir = ingest_dir / "user"
        output_dir = feature_dir / "user"
        output_dir.mkdir(parents=True, exist_ok=True)

        user_path = user_dir / "user_profile.csv"
        event_path = user_dir / "user_event_log.csv"

        user_df = pd.read_csv(user_path)
        event_df = pd.read_csv(event_path)

        event_df["created_at"] = pd.to_datetime(event_df["created_at"])

        now = event_df["created_at"].max()
        window = now - pd.Timedelta(days=7)

        event_7d = event_df[event_df["created_at"] >= window]

        avg_dwell = event_7d.groupby("user_id")["dwell_time"].mean()
        event_freq = event_7d.groupby("user_id").size()
        high_risk_ratio = event_7d.groupby("user_id")["is_high_risk"].mean()

        feature_df = user_df[["user_id"]].copy()

        feature_df["avg_dwell_time"] = feature_df["user_id"].map(avg_dwell)
        feature_df["event_frequency_7d"] = feature_df["user_id"].map(event_freq)
        feature_df["high_risk_action_ratio"] = feature_df["user_id"].map(
            high_risk_ratio
        )

        feature_df = feature_df.fillna(0)

        out_path = output_dir / "user_features.csv"
        feature_df.to_csv(out_path, index=False)

        print("☑️ user_features.csv")

        feature_dict = pd.DataFrame([
            {
                "feature_name": "user_id",
                "description_ko": "유저 ID",
                "formula": "",
                "window": "",
                "notes": "join key"
            },
            {
                "feature_name": "avg_dwell_time",
                "description_ko": "평균 체류 시간",
                "formula": "mean(dwell_time) over last 7d",
                "window": "7d",
                "notes": ""
            },
            {
                "feature_name": "event_frequency_7d",
                "description_ko": "최근 7일 이벤트 수",
                "formula": "count(*) over last 7d",
                "window": "7d",
                "notes": ""
            },
            {
                "feature_name": "high_risk_action_ratio",
                "description_ko": "고위험 행동 비율",
                "formula": "mean(is_high_risk) over last 7d",
                "window": "7d",
                "notes": ""
            },
        ])

        dict_path = output_dir / "user_feature_dictionary.csv"
        feature_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        print("☑️ user_feature_dictionary.csv")

    # ============================================================
    # 5. 데이터 정의서
    # ============================================================
    

    # ============================================================
    # LangGraph 실행 인터페이스
    # ============================================================

    def execute(self, ctx, ap):

        print("🚀 Feature Extraction Start")
        ingest_dir = ap.ingest_dir()
        feature_dir = ap.feature_table_dir()

        self.run_finance_features(ingest_dir, feature_dir)
        self.run_price_features(ingest_dir, feature_dir)
        self.run_news_features(ingest_dir, feature_dir)
        self.run_user_features(ingest_dir, feature_dir)

        print("☑️ Feature Extraction Complete")

        return {"status": "success"}


# if __name__ == "__main__":

#     ingest_dir = Path("artifacts/test_run/ingest")
#     feature_dir = Path("artifacts/test_run/features")

#     agent = FeatureExtractionAgent()

#     agent.run_finance_features(ingest_dir, feature_dir)
#     agent.run_price_features(ingest_dir, feature_dir)
#     agent.run_news_features(ingest_dir, feature_dir)
#     agent.run_user_features(ingest_dir, feature_dir)