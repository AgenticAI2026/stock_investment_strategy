from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import re

from core.result import StageResult


class FeatureExtractionAgent:

    # ============================================================
    # 1. FINANCE FEATURES
    # ============================================================
    def run_finance_features(self, ap, feature_dir, ctx):
        input_dir = ap.stage_dir("ingest/finance")
        output_dir = feature_dir / "finance"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in input_dir.glob("*.csv") if "_financials_" in f.name])

        if not files:
            raise FileNotFoundError(f"No financial csv files in {input_dir}")

        ctx.logger.info(f"📂 Found {len(files)} financial files")

        # ===============================
        # helpers
        # ===============================
        def _to_num(x):
            if pd.isna(x):
                return np.nan
            try:
                return float(str(x).replace(",", "").strip())
            except:
                return np.nan

        def pick_amount(df_year, sj_div, account_ids=(), account_nms=()):
            if isinstance(sj_div, (list, tuple)):
                d = df_year[df_year["sj_div"].isin(sj_div)]
            else:
                d = df_year[df_year["sj_div"] == sj_div]

            if account_ids:
                d = d[d["account_id"].astype(str).isin(account_ids)]

            if account_nms:
                d = d[d["account_nm"].astype(str).isin(account_nms)]

            if d.empty:
                return np.nan

            d["val"] = d["thstrm_amount"].map(_to_num)
            d = d.dropna(subset=["val"])

            if d.empty:
                return np.nan

            return float(d.loc[d["val"].abs().idxmax(), "val"])

        # ===============================
        # main loop
        # ===============================
        outputs = []

        for f in files:
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
                df["thstrm_amount"] = df["thstrm_amount"].map(_to_num)

                ticker = f.name.split("_")[0]
                years = sorted(df["year"].dropna().astype(int).unique())

                rows = []

                for y in years:
                    df_y = df[df["year"].astype(int) == y]

                    revenue = pick_amount(df_y, ("IS","CIS"),
                        account_ids=("ifrs_Revenue","ifrs-full_Revenue"))

                    op_income = pick_amount(df_y, ("IS","CIS"),
                        account_ids=("dart_OperatingIncomeLoss","ifrs_OperatingIncomeLoss"))

                    net_income_amt = pick_amount(df_y, ("IS","CIS"),
                        account_ids=("ifrs_ProfitLoss",))

                    total_liab = pick_amount(df_y, "BS", account_nms=("부채총계",))
                    total_equity = pick_amount(df_y, "BS", account_nms=("자본총계",))
                    total_assets = pick_amount(df_y, "BS", account_nms=("자산총계",))
                    current_assets = pick_amount(df_y, "BS", account_nms=("유동자산",))
                    current_liab = pick_amount(df_y, "BS", account_nms=("유동부채",))

                    rows.append({
                        "종목코드": ticker,
                        "year": y,
                        "revenue": revenue,
                        "operating_income_amount": op_income,
                        "net_income_amount": net_income_amt,
                        "total_liabilities": total_liab,
                        "total_equity": total_equity,
                        "current_assets": current_assets,
                        "current_liabilities": current_liab,
                        "total_assets": total_assets,
                    })

                base = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

                # ===============================
                # feature 계산
                # ===============================
                base["revenue_yoy"] = (base["revenue"] - base["revenue"].shift(1)) / base["revenue"].shift(1)

                base["operating_margin"] = base["operating_income_amount"] / base["revenue"]
                base["net_margin"] = base["net_income_amount"] / base["revenue"]

                base["operating_income_yoy"] = (
                    base["operating_income_amount"] - base["operating_income_amount"].shift(1)
                ) / base["operating_income_amount"].shift(1)

                base["debt_ratio"] = base["total_liabilities"] / base["total_equity"]
                base["current_ratio"] = base["current_assets"] / base["current_liabilities"]
                base["roe"] = base["net_income_amount"] / base["total_equity"]
                base["roa"] = base["net_income_amount"] / base["total_assets"]
                base["asset_turnover"] = base["revenue"] / base["total_assets"]

                base["revenue_cagr_3y"] = (base["revenue"] / base["revenue"].shift(3)) ** (1/3) - 1

                base["profit_volatility_3y"] = base["operating_income_amount"].rolling(3).std()
                base["loss_year_count_5y"] = (base["net_income_amount"] < 0).rolling(5).sum()

                out = base[[
                    "종목코드","year",
                    "revenue_yoy","operating_margin","net_margin",
                    "operating_income_yoy","debt_ratio","current_ratio",
                    "roe","roa","asset_turnover",
                    "revenue_cagr_3y","profit_volatility_3y","loss_year_count_5y"
                ]]

                # ===============================
                # 저장
                # ===============================
                out_path = output_dir / f"{ticker}_finance_features.csv"
                out.to_csv(out_path, index=False)
                outputs.append(str(out_path))

                ctx.logger.info(f"☑️ {out_path.name}")

            except Exception as e:
                ctx.logger.error(f"❌ Failed: {f.name} | {e}")

        # ============================================================
        # dictionary
        # ============================================================

        dict_path = output_dir / "finance_feature_dictionary.csv"

        rows = [
            {"feature_name": "종목코드", "description_ko": "조인 키", "formula": "", "window": "", "notes": "join_key"},
            {"feature_name": "year", "description_ko": "연도", "formula": "", "window": "time_col", "notes": ""},

            {"feature_name": "revenue_yoy", "description_ko": "매출 성장률",
            "formula": "(revenue_t - revenue_t-1) / revenue_t-1", "window": "1y"},

            {"feature_name": "operating_margin", "description_ko": "영업이익률",
            "formula": "operating_income_amount / revenue", "window": "1y"},

            {"feature_name": "net_margin", "description_ko": "순이익률",
            "formula": "net_income_amount / revenue", "window": "1y"},

            {"feature_name": "operating_income_yoy", "description_ko": "영업이익 성장률",
            "formula": "(op_income_t - op_income_t-1) / op_income_t-1", "window": "1y"},

            {"feature_name": "debt_ratio", "description_ko": "부채비율",
            "formula": "total_liabilities / total_equity", "window": "snapshot"},

            {"feature_name": "current_ratio", "description_ko": "유동비율",
            "formula": "current_assets / current_liabilities", "window": "snapshot"},

            {"feature_name": "roe", "description_ko": "ROE",
            "formula": "net_income / equity", "window": "1y"},

            {"feature_name": "roa", "description_ko": "ROA",
            "formula": "net_income / assets", "window": "1y"},

            {"feature_name": "asset_turnover", "description_ko": "자산회전율",
            "formula": "revenue / assets", "window": "1y"},

            {"feature_name": "revenue_cagr_3y", "description_ko": "3년 CAGR",
            "formula": "(revenue_t / revenue_t-3)^(1/3) - 1", "window": "3y"},

            {"feature_name": "profit_volatility_3y", "description_ko": "이익 변동성",
            "formula": "rolling_std(operating_income_amount, 3)", "window": "3y"},

            {"feature_name": "loss_year_count_5y", "description_ko": "적자 횟수",
            "formula": "count(net_income < 0, 5y)", "window": "5y"},
        ]

        pd.DataFrame(rows).to_csv(dict_path, index=False, encoding="utf-8-sig")

        ctx.logger.info("📊 Finance Feature Extraction SAVED")

        return outputs

    # ============================================================
    # 2. PRICE FEATURES
    # ============================================================
    def run_price_features(self, feature_dir, ctx, ap):
        output_dir = feature_dir / "price"
        output_dir.mkdir(parents=True, exist_ok=True)

        ohlcv = pd.read_csv(ap.price_ohlcv())
        foreign = pd.read_csv(ap.price_foreign())

        MIN_PERIODS = 20
        MA_WINDOWS = [20, 60, 120, 240]
        Z_WINDOWS = [20, 60, 120]
        LIQ_WINDOWS = [20, 60]
        VOL_WINDOWS = [20, 60, 120, 252]
        DD_WINDOWS = [60, 120, 252]
        RSI_WINDOWS = [14, 28]
        ATR_WINDOW = 14

        # ===============================
        # Utils
        # ===============================
        def safe_div(a, b):
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            return np.where((b.isna()) | (b == 0), np.nan, a / b)
        
        def rolling_pct_rank(series: pd.Series, window=20, minp=20):
            s = pd.to_numeric(series, errors="coerce").to_numpy()
            out = np.full(len(s), np.nan, dtype=float)
            for i in range(len(s)):
                start = max(0, i - window + 1)
                w = s[start:i + 1]
                w = w[~np.isnan(w)]
                if len(w) < minp or np.isnan(s[i]):
                    continue
                out[i] = (w <= s[i]).mean()
            return pd.Series(out, index=series.index)

        def rsi(series, window=14, minp=14):
            s = pd.to_numeric(series, errors="coerce")
            delta = s.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(window, min_periods=minp).mean()
            avg_loss = loss.rolling(window, min_periods=minp).mean()
            rs = safe_div(avg_gain, avg_loss)
            return 100 - (100 / (1 + rs))

        # ===============================
        # OHLCV 처리
        # ===============================
        need_ohlcv = ["date", "종목코드", "open", "high", "low", "close", "volume", "row_type"]
        missing = [c for c in need_ohlcv if c not in ohlcv.columns]
        if missing:
            raise ValueError(f"[OHLCV] 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {list(ohlcv.columns)}")

        ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
        ohlcv["종목코드"] = ohlcv["종목코드"].astype(str).str.zfill(6)
        ohlcv = ohlcv.dropna(subset=["date"]).sort_values(["종목코드", "date"]).reset_index(drop=True)

        row_priority = {"ERROR": 0, "hist": 1, "today": 2}
        ohlcv["_rp"] = ohlcv["row_type"].map(row_priority).fillna(1).astype(int)
        ohlcv = (
            ohlcv.sort_values(["종목코드", "date", "_rp"])
            .drop_duplicates(["종목코드", "date"], keep="last")
            .drop(columns=["_rp"])
            .sort_values(["종목코드", "date"])
            .reset_index(drop=True)
        )

        for c in ["open", "high", "low", "close", "volume"]:
            ohlcv[c] = pd.to_numeric(ohlcv[c], errors="coerce")

        err_mask = ohlcv["row_type"].astype(str).str.upper() == "ERROR"
        ohlcv.loc[err_mask, ["open", "high", "low", "close", "volume"]] = np.nan

        g = ohlcv.groupby("종목코드", group_keys=False)

        ohlcv["prev_close"] = g["close"].shift(1)
        ohlcv["daily_return"] = safe_div(ohlcv["close"] - ohlcv["prev_close"], ohlcv["prev_close"])
        ohlcv["log_return"] = np.log(safe_div(ohlcv["close"], ohlcv["prev_close"]))
        ohlcv["intraday_volatility"] = safe_div(ohlcv["high"] - ohlcv["low"], ohlcv["open"])
        ohlcv["gap_return"] = safe_div(ohlcv["open"] - ohlcv["prev_close"], ohlcv["prev_close"])

        tr1 = (ohlcv["high"] - ohlcv["low"]).abs()
        tr2 = (ohlcv["high"] - ohlcv["prev_close"]).abs()
        tr3 = (ohlcv["low"] - ohlcv["prev_close"]).abs()
        ohlcv["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)

        ohlcv[f"atr_{ATR_WINDOW}"] = g["true_range"].transform(
            lambda s: s.rolling(ATR_WINDOW, min_periods=min(ATR_WINDOW, MIN_PERIODS)).mean()
        )

        for w in MA_WINDOWS:
            ma_col = f"ma{w}_close"
            ohlcv[ma_col] = g["close"].transform(
                lambda s: s.rolling(w, min_periods=min(w, MIN_PERIODS)).mean()
            )
            prev_ma = ohlcv.groupby("종목코드")[ma_col].shift(1)
            ohlcv[f"ma{w}_slope"] = safe_div(ohlcv[ma_col], prev_ma) - 1
            ohlcv[f"price_to_ma{w}"] = safe_div(ohlcv["close"], ohlcv[ma_col]) - 1

        for w in Z_WINDOWS:
            meanw = g["volume"].transform(
                lambda s: s.rolling(w, min_periods=min(w, MIN_PERIODS)).mean()
            )
            stdw = g["volume"].transform(
                lambda s: s.rolling(w, min_periods=min(w, MIN_PERIODS)).std(ddof=0)
            )
            ohlcv[f"volume_zscore_{w}d"] = np.where(
                (pd.isna(stdw)) | (stdw == 0),
                np.nan,
                (ohlcv["volume"] - meanw) / stdw,
            )

        for w in LIQ_WINDOWS:
            pcol = f"liquidity_percentile_{w}d"
            scol = f"liquidity_state_{w}d"

            ohlcv[pcol] = ohlcv.groupby("종목코드")["volume"].transform(
                lambda s: rolling_pct_rank(s, window=w, minp=min(w, MIN_PERIODS))
            )

            lp = ohlcv[pcol]
            ohlcv[scol] = np.select(
                [lp < 0.33, (lp >= 0.33) & (lp < 0.66), lp >= 0.66],
                ["Low", "Medium", "High"],
                default="Unknown",
            )
            ohlcv.loc[ohlcv[scol] == "Unknown", scol] = np.nan

        for w in VOL_WINDOWS:
            stdw = g["daily_return"].transform(
                lambda s: s.rolling(w, min_periods=min(w, MIN_PERIODS)).std(ddof=0)
            )
            ohlcv[f"ret_vol_{w}d"] = stdw
            ohlcv[f"ret_vol_ann_{w}d"] = stdw * np.sqrt(252)

        for w in DD_WINDOWS:
            maxc = g["close"].transform(
                lambda s: s.rolling(w, min_periods=min(w, MIN_PERIODS)).max()
            )
            ohlcv[f"rolling_max_{w}d"] = maxc
            ohlcv[f"drawdown_{w}d"] = safe_div(ohlcv["close"], maxc) - 1

        high_52w = g["close"].transform(
            lambda s: s.rolling(252, min_periods=min(252, MIN_PERIODS)).max()
        )
        low_52w = g["close"].transform(
            lambda s: s.rolling(252, min_periods=min(252, MIN_PERIODS)).min()
        )
        ohlcv["high_52w"] = high_52w
        ohlcv["low_52w"] = low_52w
        ohlcv["dist_to_52w_high"] = safe_div(ohlcv["close"], high_52w) - 1
        ohlcv["dist_to_52w_low"] = safe_div(ohlcv["close"], low_52w) - 1

        for w in RSI_WINDOWS:
            ohlcv[f"rsi_{w}"] = g["close"].transform(lambda s: rsi(s, window=w, minp=w))

        cond_up20 = (ohlcv["close"] > ohlcv["ma20_close"]) & (ohlcv["daily_return"] > 0)
        cond_dn20 = (ohlcv["close"] < ohlcv["ma20_close"]) & (ohlcv["daily_return"] < 0)
        ohlcv["market_regime_20"] = np.select(
            [cond_up20, cond_dn20], ["Up", "Down"], default="Side"
        )

        if "ma60_close" in ohlcv.columns:
            cond_up60 = (ohlcv["close"] > ohlcv["ma60_close"]) & (ohlcv["daily_return"] > 0)
            cond_dn60 = (ohlcv["close"] < ohlcv["ma60_close"]) & (ohlcv["daily_return"] < 0)
            ohlcv["market_regime_60"] = np.select(
                [cond_up60, cond_dn60], ["Up", "Down"], default="Side"
            )

        # ===============================
        # FOREIGN 처리
        # ===============================
        need_f = ["date", "종목코드", "volume", "frgn_ntby_qty", "hts_frgn_ehrt"]
        missing_f = [c for c in need_f if c not in foreign.columns]
        if missing_f:
            raise ValueError(f"[FOREIGN] 필요한 컬럼이 없습니다: {missing_f}\n현재 컬럼: {list(foreign.columns)}")

        foreign["date"] = pd.to_datetime(foreign["date"], errors="coerce")
        foreign["종목코드"] = foreign["종목코드"].astype(str).str.zfill(6)
        foreign = foreign.dropna(subset=["date"]).sort_values(["종목코드", "date"]).reset_index(drop=True)

        for c in ["volume", "frgn_ntby_qty", "hts_frgn_ehrt"]:
            foreign[c] = pd.to_numeric(foreign[c], errors="coerce")

        foreign["foreign_net_flow_ratio"] = safe_div(foreign["frgn_ntby_qty"], foreign["volume"])
        foreign["foreign_ownership_level"] = pd.to_numeric(foreign["hts_frgn_ehrt"], errors="coerce")

        # ===============================
        # 저장
        # ===============================
        ohlcv_path = output_dir / "price_ohlcv_features.csv"
        foreign_path = output_dir / "price_foreign_features.csv"

        ohlcv.to_csv(ohlcv_path, index=False)
        foreign.to_csv(foreign_path, index=False)
        
        # ============================================================
        # 11. 데이터 정의서
        # ============================================================

        rows = []

        # --------------------------------------------------
        # 기본 컬럼
        # --------------------------------------------------
        rows += [
            {"feature_name": "date", "description_ko": "거래일자", "formula": "", "window": "1d", "notes": "time column"},
            {"feature_name": "종목코드", "description_ko": "종목코드", "formula": "str.zfill(6)", "window": "", "notes": "join key"},
            {"feature_name": "open", "description_ko": "시가", "formula": "", "window": "1d", "notes": "numeric"},
            {"feature_name": "high", "description_ko": "고가", "formula": "", "window": "1d", "notes": "numeric"},
            {"feature_name": "low", "description_ko": "저가", "formula": "", "window": "1d", "notes": "numeric"},
            {"feature_name": "close", "description_ko": "종가", "formula": "", "window": "1d", "notes": "numeric"},
            {"feature_name": "volume", "description_ko": "거래량", "formula": "", "window": "1d", "notes": "integer/numeric"},
        ]

        # --------------------------------------------------
        # 기본 파생
        # --------------------------------------------------
        rows += [
            {"feature_name": "prev_close", "description_ko": "전일 종가", "formula": "close.shift(1)", "window": "1d lag", "notes": ""},
            {"feature_name": "daily_return", "description_ko": "일간 수익률", "formula": "(close - prev_close) / prev_close", "window": "1d", "notes": ""},
            {"feature_name": "log_return", "description_ko": "로그 수익률", "formula": "ln(close / prev_close)", "window": "1d", "notes": ""},
            {"feature_name": "intraday_volatility", "description_ko": "장중 변동성", "formula": "(high - low) / open", "window": "1d", "notes": ""},
            {"feature_name": "gap_return", "description_ko": "갭 수익률", "formula": "(open - prev_close) / prev_close", "window": "1d", "notes": ""},
            {
                "feature_name": "true_range",
                "description_ko": "True Range",
                "formula": "max(abs(high-low), abs(high-prev_close), abs(low-prev_close))",
                "window": "1d",
                "notes": "ATR 계산용 중간 피처",
            },
            {
                "feature_name": f"atr_{ATR_WINDOW}",
                "description_ko": f"ATR({ATR_WINDOW})",
                "formula": f"rolling_mean(true_range,{ATR_WINDOW})",
                "window": f"{ATR_WINDOW}d rolling",
                "notes": "평균 진폭",
            },
        ]

        # --------------------------------------------------
        # 거래량
        # --------------------------------------------------
        for w in Z_WINDOWS:
            rows.append({
                "feature_name": f"volume_zscore_{w}d",
                "description_ko": f"{w}일 거래량 z-score",
                "formula": f"(volume - rolling_mean(volume,{w})) / rolling_std(volume,{w})",
                "window": f"{w}d rolling",
                "notes": "비정상 거래량 탐지",
            })

        for w in LIQ_WINDOWS:
            rows.append({
                "feature_name": f"liquidity_percentile_{w}d",
                "description_ko": f"{w}일 거래량 분위수",
                "formula": f"pct_rank(volume within trailing {w}d window)",
                "window": f"{w}d rolling",
                "notes": "거래량의 상대적 수준",
            })
            rows.append({
                "feature_name": f"liquidity_state_{w}d",
                "description_ko": f"{w}일 거래량 상태",
                "formula": "Low/Medium/High by percentile thresholds (0.33, 0.66)",
                "window": f"{w}d rolling",
                "notes": "범주형 유동성 상태",
            })

        # --------------------------------------------------
        # 이동평균
        # --------------------------------------------------
        for w in [20, 60, 120, 240]:
            rows.append({
                "feature_name": f"ma{w}_close",
                "description_ko": f"종가 {w}일 이동평균",
                "formula": f"rolling_mean(close,{w})",
                "window": f"{w}d"
            })
            rows.append({
                "feature_name": f"ma{w}_slope",
                "description_ko": f"MA{w} 기울기",
                "formula": f"(ma{w}_close / ma{w}_close.shift(1)) - 1",
                "window": f"{w}d"
            })
            rows.append({
                "feature_name": f"price_to_ma{w}",
                "description_ko": f"MA{w} 대비 괴리율",
                "formula": f"(close / ma{w}_close) - 1",
                "window": f"{w}d"
            })

        # --------------------------------------------------
        # 변동성
        # --------------------------------------------------
        for w in [20, 60, 120, 252]:
            rows.append({
                "feature_name": f"ret_vol_{w}d",
                "description_ko": f"{w}일 변동성",
                "formula": f"rolling_std(daily_return,{w})",
                "window": f"{w}d"
            })
            rows.append({
                "feature_name": f"ret_vol_ann_{w}d",
                "description_ko": f"{w}일 연율화 변동성",
                "formula": "vol * sqrt(252)",
                "window": f"{w}d"
            })

        # --------------------------------------------------
        # 드로우다운
        # --------------------------------------------------
        for w in DD_WINDOWS:
            rows.append({
                "feature_name": f"rolling_max_{w}d",
                "description_ko": f"{w}일 롤링 최고 종가",
                "formula": f"rolling_max(close,{w})",
                "window": f"{w}d rolling",
                "notes": "드로우다운 계산용 중간 피처",
            })
            rows.append({
                "feature_name": f"drawdown_{w}d",
                "description_ko": f"{w}일 드로우다운",
                "formula": f"(close / rolling_max_{w}d) - 1",
                "window": f"{w}d rolling",
                "notes": "최고점 대비 하락률",
            })

        # --------------------------------------------------
        # 52주
        # --------------------------------------------------
        rows += [
            {"feature_name": "high_52w", "description_ko": "52주 최고", "formula": "rolling_max", "window": "252d"},
            {"feature_name": "low_52w", "description_ko": "52주 최저", "formula": "rolling_min", "window": "252d"},
            {"feature_name": "dist_to_52w_high", "description_ko": "고점 대비", "formula": "close/high - 1", "window": "252d"},
            {"feature_name": "dist_to_52w_low", "description_ko": "저점 대비", "formula": "close/low - 1", "window": "252d"},
        ]

        # --------------------------------------------------
        # RSI
        # --------------------------------------------------
        for w in [14, 28]:
            rows.append({
                "feature_name": f"rsi_{w}",
                "description_ko": f"RSI {w}",
                "formula": "RSI",
                "window": f"{w}d"
            })

        # --------------------------------------------------
        # 외국인
        # --------------------------------------------------
        rows += [
            {
                "feature_name": "market_regime_20",
                "description_ko": "20일 기준 시장 국면",
                "formula": "Up if close > ma20_close and daily_return > 0, Down if close < ma20_close and daily_return < 0, else Side",
                "window": "20d",
                "notes": "추세 상태",
            },
            {
                "feature_name": "market_regime_60",
                "description_ko": "60일 기준 시장 국면",
                "formula": "Up if close > ma60_close and daily_return > 0, Down if close < ma60_close and daily_return < 0, else Side",
                "window": "60d",
                "notes": "중기 추세 상태",
            },
            {"feature_name": "frgn_ntby_qty",
            "description_ko": "외국인 순매수 수량(또는 순매수량)",
            "data_source": "FOREIGN snapshot",
            "formula": "",
            "window": "today snapshot",
            "notes": "numeric; 부호(±) 의미 확인 필요"},

            {"feature_name": "hts_frgn_ehrt",
            "description_ko": "외국인 보유비중(%) 원천 값",
            "data_source": "FOREIGN snapshot",
            "formula": "",
            "window": "today snapshot",
            "notes": "numeric; 퍼센트 단위(%)"},

            {"feature_name":"foreign_net_flow_ratio",
            "description_ko":"외국인 순매수 비율",
            "formula":"frgn_ntby_qty / volume",
            "window":"1d",
            "notes":"0~±값; 거래량 대비 순매수 강도"},

            {"feature_name":"foreign_ownership_level",
            "description_ko":"외국인 보유 비중(%)",
            "formula":"to_numeric(hts_frgn_ehrt)",
            "window":"snapshot",
            "notes":"보유 구조 분석"}
        ]

        # --------------------------------------------------
        # 저장
        # --------------------------------------------------
        dict_path = output_dir / "price_feature_dictionary.csv"
        pd.DataFrame(rows).to_csv(dict_path, index=False, encoding="utf-8-sig")

        ctx.logger.info("📊 Price Feature Extraction SAVED")

        return [str(ohlcv_path), str(foreign_path), str(dict_path)]

    def run_news_features(self, ap, feature_dir, ctx):
        output_dir = feature_dir / "news"
        output_dir.mkdir(parents=True, exist_ok=True)

        news_path = ap.file("ingest/news", "top100_naver_news_merged.csv")

        # ============================================================
        # 1. Load + 컬럼 자동 매핑
        # ============================================================

        df = pd.read_csv(news_path)

        def pick_col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        col_code = pick_col(["stock_code", "종목코드", "code", "ticker"])
        col_name = pick_col(["stock_name", "종목명", "name"])
        col_title = pick_col(["title", "news_title"])
        col_desc = pick_col(["description", "desc", "summary", "content"])
        col_link = pick_col(["link", "url"])
        col_originallink = pick_col(["originallink", "originalLink", "original_link"])
        col_pub = pick_col(["pubDate", "pub_date", "published_at", "date"])

        if col_code is None or col_title is None or col_pub is None:
            raise ValueError("필수 컬럼 없음 (종목코드, title, pubDate)")

        # ============================================================
        # 2. 기준 시각
        # ============================================================
        now = pd.Timestamp(ctx.asof_date, tz="UTC")
        recent_cutoff = now - pd.Timedelta(hours=24)

        # ============================================================
        # 3. 전처리
        # ============================================================

        df[col_code] = df[col_code].astype(str).str.strip()
        mask = df[col_code].str.fullmatch(r"\d+")
        df.loc[mask, col_code] = df.loc[mask, col_code].str.zfill(6)

        df[col_pub] = pd.to_datetime(df[col_pub], errors="coerce", utc=True)
        df = df.dropna(subset=[col_pub])

        df = df[df[col_pub] <= now]

        # ============================================================
        # 정렬
        # ============================================================
        df = df.sort_values([col_code, col_pub]).reset_index(drop=True)

        # ============================================================
        # 4. RAW 저장 + RAW dictionary
        # ============================================================

        raw_out_path = output_dir / "news_raw.csv"
        raw_dict_path = output_dir / "news_raw_dictionary.csv"

        rows = []

        rows.append({
            "feature_name": col_code,
            "description_ko": "종목코드(조인 키, 6자리 문자열)",
            "formula": "astype(str) → strip → zfill(6)",
            "window": "",
            "notes": "role=join_key, semantic_type=ticker, model_use=false"
        })

        rows.append({
            "feature_name": col_pub,
            "description_ko": "기사 발행 시각(시간축 기준 컬럼)",
            "formula": "to_datetime(..., utc=True)",
            "window": "event time",
            "notes": "role=time_col, semantic_type=datetime, split 기준"
        })

        rows.append({
            "feature_name": col_title,
            "description_ko": "기사 제목",
            "formula": "",
            "window": "",
            "notes": "role=raw_text, semantic_type=text, dedup key 후보"
        })

        if col_desc is not None:
            rows.append({
                "feature_name": col_desc,
                "description_ko": "기사 요약/본문",
                "formula": "",
                "window": "",
                "notes": "role=raw_text, semantic_type=text"
            })

        if col_link is not None:
            rows.append({
                "feature_name": col_link,
                "description_ko": "기사 링크(URL)",
                "formula": "",
                "window": "",
                "notes": "role=dedup_key, semantic_type=url"
            })

        if col_originallink is not None:
            rows.append({
                "feature_name": col_originallink,
                "description_ko": "원문 링크(URL)",
                "formula": "",
                "window": "",
                "notes": "role=dedup_key_primary, semantic_type=url"
            })

        raw_dict = pd.DataFrame(rows)
        raw_dict.to_csv(raw_dict_path, index=False, encoding="utf-8-sig")

        df_raw = df.copy()

        if "fetched_at" in df_raw.columns:
            df_raw = df_raw.drop(columns=["fetched_at"])

        if col_name in df_raw.columns:
            df_raw = df_raw.drop(columns=[col_name])

        df_raw.to_csv(raw_out_path, index=False, encoding="utf-8-sig")

        # ============================================================
        # 5. 텍스트 생성
        # ============================================================

        df["__text"] = df[col_title].fillna("").astype(str)
        if col_desc:
            df["__text"] += " " + df[col_desc].fillna("").astype(str)

        EVENT_KEYWORDS = [
            "실적", "적자", "흑자", "전망",
            "인수", "합병", "M&A",
            "규제", "제재", "조사",
            "소송", "리콜", "사고",
            "계약", "수주", "신제품"
        ]

        NEG_KEYWORDS = [
            "적자", "하락", "급락", "부진",
            "제재", "규제", "소송", "조사",
            "위기", "논란", "사고", "리콜"
        ]

        event_pat = re.compile("|".join(EVENT_KEYWORDS))
        neg_pat = re.compile("|".join(NEG_KEYWORDS))

        # ============================================================
        # 6. Feature 계산
        # ============================================================

        def compute(g):
            N = len(g)
            if N == 0:
                return pd.Series({
                    "recent_news_ratio": np.nan,
                    "event_news_ratio": np.nan,
                    "negative_news_ratio": np.nan,
                    "time_concentration_score": np.nan,
                    "latest_negative_flag": np.nan,
                })

            recent_ratio = (g[col_pub] >= recent_cutoff).mean()

            text = g["__text"]
            event_ratio = text.str.contains(event_pat).mean()
            neg_ratio = text.str.contains(neg_pat).mean()

            bins = g[col_pub].dt.floor("6h")
            max_bin = bins.value_counts().max()
            time_conc = max_bin / N

            latest = g.sort_values(col_pub, ascending=False).head(5)["__text"]
            latest_flag = int(latest.str.contains(neg_pat).any())

            return pd.Series({
                "recent_news_ratio": recent_ratio,
                "event_news_ratio": event_ratio,
                "negative_news_ratio": neg_ratio,
                "time_concentration_score": time_conc,
                "latest_negative_flag": latest_flag,
            })

        features = (
            df.groupby([col_code])
            .apply(compute, include_groups=False)
            .reset_index()
        )

        # ============================================================
        # 7. 저장
        # ============================================================

        out_path = output_dir / "news_features.csv"
        features.to_csv(out_path, index=False)

        # ============================================================
        # 8. Feature Dictionary
        # ============================================================
        LATEST_N = 5          # latest_negative_flag 계산용
        RECENT_HOURS = 24     # recent_news_ratio 계산용
        BIN_HOURS = 6         # time_concentration_score 계산용

        feature_dict = pd.DataFrame([
            {
                "feature_name": col_code,
                "description_ko": "종목코드 (조인 키)",
                "formula": "",
                "window": "",
                "notes": "role=join_key, semantic_type=ticker, model_use=false"
            },
            {
                "feature_name": "recent_news_ratio",
                "description_ko": f"기준시각(now={now}) 기준 최근 {RECENT_HOURS}시간 내 기사 비율",
                "formula": f"(# pubDate >= now-{RECENT_HOURS}h) / N",
                "window": f"{RECENT_HOURS}h",
                "notes": "N은 해당 종목 기사 수(보통 50). 파일 내부 최신 기사 시각을 now로 사용."
            },
            {
                "feature_name": "event_news_ratio",
                "description_ko": "이벤트 키워드 포함 기사 비율",
                "formula": "(# text contains EVENT_KEYWORDS) / N",
                "window": "전체 기사(보통 50개)",
                "notes": f"EVENT_KEYWORDS={EVENT_KEYWORDS}"
            },
            {
                "feature_name": "negative_news_ratio",
                "description_ko": "부정 키워드(급락/부진/규제/소송/사고/리콜 등) 포함 기사 비율",
                "formula": "(# text contains NEGATIVE_KEYWORDS) / N",
                "window": "전체 기사(보통 50개)",
                "notes": f"NEGATIVE_KEYWORDS={NEG_KEYWORDS}"
            },
            {
                "feature_name": "time_concentration_score",
                "description_ko": f"{BIN_HOURS}시간 단위로 기사를 묶었을 때, 가장 많이 몰린 구간의 비중(집중도)",
                "formula": f"max(count_in_{BIN_HOURS}h_bins) / N",
                "window": "전체 기사(보통 50개)",
                "notes": "속보/공시/사건처럼 특정 시간대에 뉴스가 몰리는지 탐지"
            },
            {
                "feature_name": "latest_negative_flag",
                "description_ko": f"가장 최신 기사 {LATEST_N}개 중 부정 키워드 포함 기사가 하나라도 있으면 1, 아니면 0",
                "formula": f"any(NEGATIVE in latest_{LATEST_N}) -> 1 else 0",
                "window": f"latest {LATEST_N}",
                "notes": "단기 알림/리스크 플래그로 유용"
            },
        ])

        dict_path = output_dir / "news_feature_dictionary.csv"
        feature_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        ctx.logger.info("📰 News Feature Extraction Saved")

        return str(out_path)

    # ============================================================
    # 4. USER FEATURES
    # ============================================================
    def run_user_features(self, ap, feature_dir, ctx):
        output_dir = feature_dir / "user"
        output_dir.mkdir(parents=True, exist_ok=True)

        user_df = pd.read_csv(ap.file("ingest/user", "user_profile.csv"))
        event_df = pd.read_csv(ap.file("ingest/user", "user_event_log.csv"))

        event_df["created_at"] = pd.to_datetime(event_df["created_at"], errors="coerce")

        # ============================================================
        # 기준 시각 (leakage 방지)
        # ============================================================
        PIPELINE_TS = pd.Timestamp(ctx.asof_date)

        if event_df["created_at"].notna().any():
            NOW_TS = min(event_df["created_at"].max(), PIPELINE_TS)
        else:
            NOW_TS = PIPELINE_TS

        NOW_YEAR = NOW_TS.year
        WINDOW_7D_START = NOW_TS - pd.Timedelta(days=7)

        ctx.logger.info(f"🕒 NOW_TS: {NOW_TS}")

        # 정렬
        event_df = event_df.sort_values(["user_id", "created_at"]).reset_index(drop=True)

        # ============================================================
        # STATIC FEATURES
        # ============================================================

        risk_map = {"low": 0.0, "mid": 0.5, "high": 1.0}
        horizon_map = {"SHORT": 0.0, "MID": 0.5, "LONG": 1.0}

        static_feat = user_df[["user_id"]].copy()

        static_feat["user_age"] = NOW_YEAR - pd.to_numeric(user_df["birth_year"], errors="coerce")
        static_feat["risk_score"] = user_df["risk_tolerance"].map(risk_map).astype(float)
        static_feat["investment_horizon_score"] = user_df["investment_horizon"].map(horizon_map).astype(float)
        static_feat["leverage_preference"] = pd.to_numeric(user_df["leverage_allowed"], errors="coerce").fillna(0).astype(int)

        # ============================================================
        # DYNAMIC FEATURES (7D WINDOW)
        # ============================================================

        event_7d = event_df[
            (event_df["created_at"] >= WINDOW_7D_START) &
            (event_df["created_at"] <= NOW_TS)
        ].copy()

        # avg_dwell_time
        avg_dwell = (
            event_7d[event_7d["event_type"].isin(["view", "click"])]
            .groupby("user_id")["dwell_time"]
            .mean()
            .rename("avg_dwell_time")
        )

        # event_frequency
        event_freq = (
            event_7d.groupby("user_id")
            .size()
            .rename("event_frequency_7d")
        )

        # high_risk_action_ratio
        high_risk_ratio = (
            event_7d.groupby("user_id")["is_high_risk"]
            .mean()
            .rename("high_risk_action_ratio")
        )

        # ============================================================
        # exploration_ratio
        # ============================================================

        event_df["first_seen_at"] = (
            event_df.groupby(["user_id", "item_key"])["created_at"]
            .transform("min")
        )

        expl_events = event_df[
            (event_df["created_at"] >= WINDOW_7D_START) &
            (event_df["created_at"] <= NOW_TS) &
            (event_df["event_type"].isin(["view", "click"])) &
            (event_df["item_key"].notna())
        ].copy()

        expl_events["is_new_item"] = (
            expl_events["created_at"] == expl_events["first_seen_at"]
        )

        exploration_ratio = (
            expl_events.groupby("user_id")["is_new_item"]
            .mean()
            .rename("exploration_ratio")
        )

        # ============================================================
        # session_completion_rate
        # ============================================================

        session_end = event_df[
            (event_df["event_type"] == "session_end") &
            (event_df["created_at"] <= NOW_TS)
        ].copy()

        session_by_user = (
            session_end.groupby(["user_id", "session_id"])["is_session_complete"]
            .max()
            .reset_index()
        )

        session_completion_rate = (
            session_by_user.groupby("user_id")["is_session_complete"]
            .mean()
            .rename("session_completion_rate")
        )

        # ============================================================
        # MERGE
        # ============================================================

        feature_df = (
            static_feat
            .merge(avg_dwell, on="user_id", how="left")
            .merge(event_freq, on="user_id", how="left")
            .merge(high_risk_ratio, on="user_id", how="left")
            .merge(exploration_ratio, on="user_id", how="left")
            .merge(session_completion_rate, on="user_id", how="left")
        )

        # 결측 처리
        feature_df = feature_df.fillna({
            "avg_dwell_time": 0.0,
            "event_frequency_7d": 0,
            "high_risk_action_ratio": 0.0,
            "exploration_ratio": 0.0,
            "session_completion_rate": 0.0
        })

        # ============================================================
        # SAVE
        # ============================================================

        out_path = output_dir / "user_features.csv"
        feature_df.to_csv(out_path, index=False)

        # ============================================================
        # DATA DICTIONARY
        # ============================================================
        dict_rows = [
            {
                "feature_name": "user_id",
                "description_ko": "유저 ID (조인 키)",
                "formula": "",
                "window": "",
                "notes": "role=join_key, semantic_type=id, model_use=false"
            },
            {
                "feature_name": "user_age",
                "description_ko": "유저 나이",
                "formula": "NOW_YEAR - birth_year",
                "window": "static",
                "notes": "NOW_YEAR는 파이프라인 기준 시점의 연도"
            },
            {
                "feature_name": "risk_score",
                "description_ko": "리스크 성향 점수",
                "formula": "map(risk_tolerance): low=0, mid=0.5, high=1",
                "window": "static",
                "notes": "범주형 → 수치형 매핑"
            },
            {
                "feature_name": "investment_horizon_score",
                "description_ko": "투자 기간 점수",
                "formula": "map(investment_horizon): SHORT=0, MID=0.5, LONG=1",
                "window": "static",
                "notes": "투자 기간을 수치화한 feature"
            },
            {
                "feature_name": "leverage_preference",
                "description_ko": "레버리지 허용 여부",
                "formula": "int(leverage_allowed)",
                "window": "static",
                "notes": "0/1 binary"
            },
            {
                "feature_name": "avg_dwell_time",
                "description_ko": "평균 체류 시간(초)",
                "formula": "mean(dwell_time where event_type in {view, click})",
                "window": "7d",
                "notes": "최근 7일 기준, 사용자 관심도 proxy"
            },
            {
                "feature_name": "event_frequency_7d",
                "description_ko": "최근 7일 이벤트 수",
                "formula": "count(*)",
                "window": "7d",
                "notes": "전체 이벤트 기준 활동성 지표"
            },
            {
                "feature_name": "high_risk_action_ratio",
                "description_ko": "고위험 행동 비율",
                "formula": "mean(is_high_risk)",
                "window": "7d",
                "notes": "event_log에 is_high_risk(0/1) 필요"
            },
            {
                "feature_name": "exploration_ratio",
                "description_ko": "신규 탐색 비율",
                "formula": "mean(created_at == first_seen_at(user_id,item_key)) for view/click",
                "window": "7d",
                "notes": "탐색 성향 (explore vs exploit)"
            },
            {
                "feature_name": "session_completion_rate",
                "description_ko": "세션 완료율",
                "formula": "mean(max(is_session_complete) per session_id)",
                "window": "all_sessions",
                "notes": "session_end 이벤트 기반"
            },
        ]

        dict_path = output_dir / "user_feature_dictionary.csv"
        pd.DataFrame(dict_rows).to_csv(dict_path, index=False, encoding="utf-8-sig")

        ctx.logger.info("👤 User Feature Extraction SAVED")

        return str(out_path)

    def execute(self, ctx, ap):
        ctx.logger.info("🚀 Feature Extraction Start")

        feature_dir = ap.feature_table_dir()

        artifacts = {}

        try:
            # Finance
            finance_path = self.run_finance_features(ap, feature_dir, ctx)
            artifacts["finance"] = finance_path

            # Price
            price_paths = self.run_price_features(feature_dir, ctx, ap)
            artifacts["price_ohlcv"] = price_paths[0]
            artifacts["price_foreign"] = price_paths[1]
            artifacts["price_dict"] = price_paths[2]

            # News
            news_path = self.run_news_features(ap, feature_dir, ctx)
            artifacts["news"] = news_path

            # User
            user_path = self.run_user_features(ap, feature_dir, ctx)
            artifacts["user"] = user_path

            return StageResult.success(
                stage="feature_extraction",
                outputs=list(artifacts.values()),
                metrics={
                    "modules": list(artifacts.keys()),
                    "feature_root": str(feature_dir)
                }
            )

        except Exception as e:
            ctx.logger.error(f"❌ Feature Extraction Failed: {e}")

            return StageResult.failed(
                stage="feature_extraction",
                error=str(e)
            )