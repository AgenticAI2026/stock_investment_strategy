# 뉴스·시세·재무 수집 총괄 agent
from __future__ import annotations

import os
import re
import json
import time
import shutil
import datetime as dt
from html import unescape
from pathlib import Path
from typing import Optional, Dict, Any, List
import traceback

import requests
import pandas as pd
import dart_fss as dart
from tqdm import tqdm
from zoneinfo import ZoneInfo

from utils.env_loader import load_env

from core.agent_base import BaseAgent
from core.context import RunContext
from core.artifacts import ArtifactPaths
from core.result import StageResult


class DataIngestionAgent(BaseAgent):
    stage = "ingest"

    KIS_VOLUME_RANK_ENDPOINT = "/uapi/domestic-stock/v1/quotations/volume-rank"
    KIS_VOLUME_RANK_TR_ID = "FHPST01710000"

    KIS_SEARCH_STOCK_INFO_ENDPOINT = "/uapi/domestic-stock/v1/quotations/search-stock-info"
    KIS_SEARCH_STOCK_INFO_TR_ID = "CTPF1604R"

    def __init__(
        self,
        dart_api_key: Optional[str] = None,
        user_csv_map: Optional[Dict[str, str]] = None,
        data_dir: str = "data/raw",
    ):
        load_env()

        self.data_dir = data_dir
        self.user_csv_map = user_csv_map or {}

        self.dart_api_key = dart_api_key or os.getenv("DART_API_KEY")
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

        self.kis_app_key = os.getenv("KIS_APP_KEY")
        self.kis_app_secret = os.getenv("KIS_APP_SECRET")
        self.kis_base_url = os.getenv(
            "KIS_BASE_URL",
            "https://openapi.koreainvestment.com:9443",
        )

        self._kis_access_token: Optional[str] = None
        self._kis_token_expire_at: Optional[dt.datetime] = None

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

        self.universe_top100: Optional[pd.DataFrame] = None

        if self.naver_client_id is None or self.naver_client_secret is None:
            print("[WARN] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 설정되지 않았습니다.")
        if self.dart_api_key is None:
            print("[WARN] DART_API_KEY 가 설정되지 않았습니다.")
        if self.kis_app_key is None or self.kis_app_secret is None:
            print("[WARN] KIS_APP_KEY / KIS_APP_SECRET 이 설정되지 않았습니다.")

    # =========================================================
    # util
    # =========================================================
    def _safe_name(self, s: str, max_len: int = 80) -> str:
        s = str(s) if s is not None else ""
        s = re.sub(r"[^가-힣A-Za-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:max_len] if len(s) > max_len else s

    def _strip_html(self, s: str) -> str:
        if s is None:
            return ""
        txt = re.sub(r"<[^>]+>", " ", str(s))
        txt = unescape(txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    def _build_news_query(self, stock_name: str) -> str:
        name = str(stock_name).strip()
        if not name:
            return ""
        q = f'{name} (주가 OR 실적 OR 공시 OR 전망 OR 매출 OR 영업이익)'
        if len(q) > 60:
            q = f"{name} (주가 OR 실적 OR 공시)"
        return q

    def _to_float(self, v, default=0.0) -> float:
        try:
            if v is None or v == "":
                return default
            return float(str(v).replace(",", ""))
        except Exception:
            return default

    def _to_int(self, v, default=0) -> int:
        try:
            if v is None or v == "":
                return default
            return int(float(str(v).replace(",", "")))
        except Exception:
            return default

    def _fmt_date(self, yyyymmdd: Optional[str]) -> str:
        if yyyymmdd and len(yyyymmdd) == 8:
            return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
        return dt.date.today().isoformat()

    # =========================================================
    # user ingest
    # =========================================================
    def ingest_user_data(self, user_csv_map, ap):
        saved = {}

        if "profile" not in user_csv_map:
            raise ValueError("user_csv_map missing 'profile'")

        profile_src = user_csv_map["profile"]
        profile_dst = ap.file("ingest/user", "user_profile.csv")

        shutil.copy2(profile_src, profile_dst)
        saved["profile"] = str(profile_dst)

        if "event" not in user_csv_map:
            raise ValueError("user_csv_map missing 'event'")

        event_src = user_csv_map["event"]
        event_dst = ap.file("ingest/user", "user_event_log.csv")

        shutil.copy2(event_src, event_dst)
        saved["event"] = str(event_dst)

        return saved
    
    # =========================================================
    # KIS helpers
    # =========================================================
    def _kis_get_access_token(self) -> str:
        if not self.kis_app_key or not self.kis_app_secret:
            raise RuntimeError("환경변수 KIS_APP_KEY / KIS_APP_SECRET 가 필요합니다.")

        if self._kis_access_token and self._kis_token_expire_at:
            if dt.datetime.now() < self._kis_token_expire_at:
                return self._kis_access_token

        url = f"{self.kis_base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.kis_app_key,
            "appsecret": self.kis_app_secret,
        }

        resp = self.session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"KIS token 발급 실패: {data}")

        expires_in = int(data.get("expires_in", 60 * 60 * 23))
        self._kis_access_token = token
        self._kis_token_expire_at = dt.datetime.now() + dt.timedelta(
            seconds=max(expires_in - 60, 60)
        )
        return token

    def _kis_headers(self, tr_id: str) -> dict:
        token = self._kis_get_access_token()
        return {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.kis_app_key,
            "appsecret": self.kis_app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _request_with_retry(
        self,
        method,
        url,
        *,
        headers=None,
        params=None,
        json=None,
        timeout=15,
        max_retries=5,
        base_sleep=0.3,
    ):
        last_err = None
        for i in range(max_retries):
            try:
                resp = self.session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=timeout,
                )

                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp

                snippet = (resp.text or "")[:300].replace("\n", " ")
                last_err = RuntimeError(
                    f"HTTP {resp.status_code} from {url} | body={snippet}"
                )
            except Exception as e:
                last_err = e

            time.sleep(base_sleep * (2 ** i))

        raise RuntimeError(f"request failed after retries: {last_err}")

    def _pick_list(self, data: dict) -> list:
        for k in ("output", "output1", "output2"):
            if k in data and data[k]:
                return data[k]
        return []

    # =========================================================
    # 1) Top100
    # =========================================================
    def _kis_fetch_volume_rank(
        self,
        *,
        metric: str = "trade_value",
        fid_input_iscd: str = "0000",
        price1: str = "",
        price2: str = "",
    ) -> pd.DataFrame:
        blng = "3" if metric == "trade_value" else "0"

        url = f"{self.kis_base_url}{self.KIS_VOLUME_RANK_ENDPOINT}"
        headers = self._kis_headers(tr_id=self.KIS_VOLUME_RANK_TR_ID)

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": fid_input_iscd,
            "FID_DIV_CLS_CODE": "0",
            "FID_BLNG_CLS_CODE": blng,
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000",
            "FID_INPUT_PRICE_1": price1,
            "FID_INPUT_PRICE_2": price2,
            "FID_VOL_CNT": "",
            "FID_INPUT_DATE_1": "",
        }

        resp = self._request_with_retry("GET", url, headers=headers, params=params, timeout=15)
        data = resp.json()

        if str(data.get("rt_cd")) != "0":
            raise RuntimeError(
                f"volume-rank 실패: msg_cd={data.get('msg_cd')} msg1={data.get('msg1')}"
            )

        rows = data.get("output", [])
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame(columns=["종목명", "종목코드", "trade_volume", "trade_value"])

        df = pd.DataFrame(rows)

        def _pick_col(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in df.columns:
                    return c
            return None

        col_name = _pick_col(["hts_kor_isnm", "kor_isnm", "stck_issu_name"])
        col_code = _pick_col(["mksc_shrn_iscd", "stck_shrn_iscd", "shrn_iscd"])
        col_vol = _pick_col(["acml_vol", "acc_trdvol", "trdvol"])
        col_val = _pick_col(["acml_tr_pbmn", "acc_trdval", "tr_pbmn"])

        if not col_code or not col_name:
            return pd.DataFrame(columns=["종목명", "종목코드", "trade_volume", "trade_value"])

        out = pd.DataFrame({
            "종목명": df[col_name].astype(str).str.strip(),
            "종목코드": df[col_code].astype(str).str.zfill(6),
            "trade_volume": pd.to_numeric(df[col_vol], errors="coerce").fillna(0) if col_vol else 0,
            "trade_value": pd.to_numeric(df[col_val], errors="coerce").fillna(0) if col_val else 0,
        })
        return out.reset_index(drop=True)

    def fetch_top100_by_marcap(
        self,
        ap: ArtifactPaths,
        top_n: int = 100,
        save_snapshot: bool = True,
        metric: str = "trade_value",
    ) -> pd.DataFrame:
        print(f"🔍 KIS 유동성 Top{top_n} 산출, metric={metric}")

        sort_col = "trade_value" if metric == "trade_value" else "trade_volume"

        price_bins = [
            ("", ""),
            ("0", "5000"),
            ("5000", "20000"),
            ("20000", "100000"),
            ("100000", "500000"),
            ("500000", ""),
        ]
        markets = [("KOSPI", "0001"), ("KOSDAQ", "1001")]

        best: Dict[str, dict] = {}

        for mkt_name, fid_iscd in markets:
            for p1, p2 in price_bins:
                df = self._kis_fetch_volume_rank(
                    metric=metric,
                    fid_input_iscd=fid_iscd,
                    price1=p1,
                    price2=p2,
                )
                if df.empty:
                    continue

                for _, r in df.iterrows():
                    code = str(r["종목코드"]).zfill(6)
                    row = {
                        "종목명": r["종목명"],
                        "종목코드": code,
                        "시장": mkt_name,
                        "trade_volume": float(r.get("trade_volume", 0) or 0),
                        "trade_value": float(r.get("trade_value", 0) or 0),
                    }
                    if (code not in best) or (row[sort_col] > best[code].get(sort_col, 0)):
                        best[code] = row
                time.sleep(0.08)

        pool = pd.DataFrame(list(best.values()))
        if pool.empty:
            raise RuntimeError("volume-rank 다회 호출 후에도 pool이 비었습니다.")

        top = (
            pool.sort_values(sort_col, ascending=False)
            .head(top_n)
            .loc[:, ["종목명", "종목코드", "시장"]]
            .reset_index(drop=True)
        )

        top["종목코드"] = top["종목코드"].astype(str).str.zfill(6)
        self.universe_top100 = top.copy()

        latest_path = ap.file("ingest/meta", f"top{top_n}_liquidity_latest.csv")
        top.to_csv(latest_path, index=False, encoding="utf-8-sig")

        if save_snapshot:
            stamp = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
            snap_path = ap.file("ingest/meta", f"top{top_n}_liquidity_{stamp}.csv")
            top.to_csv(snap_path, index=False, encoding="utf-8-sig")

            print(f"✅ Top{top_n} 저장 완료 → {latest_path} / {snap_path}")
        else:
            print(f"✅ Top{top_n} 저장 완료 → {latest_path}")

        return top

    # =========================================================
    # 2) Price / Foreign
    # =========================================================
    def _ensure_market_col_from_latest_universe(self, universe_csv: str | None = None):
        if self.universe_top100 is None or self.universe_top100.empty:
            return
        if "시장" in self.universe_top100.columns:
            return

        if universe_csv is None:
            universe_csv = self._latest_universe_csv_path()

        if not os.path.exists(universe_csv):
            print(f"[WARN] 시장 컬럼 보강 실패: universe_csv 없음 → {universe_csv}")
            return

        u = pd.read_csv(universe_csv)

        if "종목코드" not in u.columns or "시장" not in u.columns:
            print(f"[WARN] 시장 컬럼 보강 실패: universe_csv에 '종목코드'/'시장' 없음 → cols={list(u.columns)}")
            return

        u["종목코드"] = u["종목코드"].astype(str).str.zfill(6)
        self.universe_top100["종목코드"] = self.universe_top100["종목코드"].astype(str).str.zfill(6)

        self.universe_top100 = self.universe_top100.merge(
            u[["종목코드", "시장"]], on="종목코드", how="left"
        )

    def _market_phase_kst(self, now: dt.datetime | None = None) -> str:
        kst = ZoneInfo("Asia/Seoul")
        now = now or dt.datetime.now(kst)

        if now.weekday() >= 5:
            return "pre"

        t = now.time()
        if t < dt.time(9, 0):
            return "pre"
        elif dt.time(9, 0) <= t <= dt.time(15, 30):
            return "open"
        else:
            return "post"

    def _kis_inquire_daily_price_list(self, stock_code: str) -> List[Dict[str, Any]]:
        url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = self._kis_headers(tr_id="FHKST01010400")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(stock_code).zfill(6),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }

        resp = self._request_with_retry("GET", url, headers=headers, params=params, timeout=15)
        data = resp.json()
        out_list = self._pick_list(data)
        if not out_list:
            raise RuntimeError(f"inquire-daily-price 응답 이상: {data}")
        return out_list

    def _kis_inquire_daily_itemchartprice(self, stock_code: str, start: str, end: str) -> list:
        url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        headers = self._kis_headers(tr_id="FHKST03010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(stock_code).zfill(6),
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": end,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }

        resp = self._request_with_retry("GET", url, headers=headers, params=params, timeout=15)
        data = resp.json()
        out_list = data.get("output2", [])

        if isinstance(out_list, list) and len(out_list) == 0:
            return []
        if not isinstance(out_list, list):
            raise RuntimeError(f"itemchartprice output2 형식이상: {str(data)[:200]}")
        return out_list

    def _kis_inquire_price_domestic(self, stock_code: str) -> Dict[str, Any]:
        url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._kis_headers(tr_id="FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(stock_code).zfill(6),
        }
        resp = self._request_with_retry("GET", url, headers=headers, params=params, timeout=15)
        data = resp.json()
        if "output" not in data:
            raise RuntimeError(f"inquire-price 응답 이상: {data}")
        return data["output"]

    def _kis_extract_ohlcv_from_daily_row(self, it: dict, source: str) -> dict:
        d = it.get("stck_bsop_date") or it.get("bsop_date") or it.get("date") or ""
        return {
            "date": self._fmt_date(d),
            "open": self._to_float(it.get("stck_oprc") or it.get("open")),
            "high": self._to_float(it.get("stck_hgpr") or it.get("high")),
            "low": self._to_float(it.get("stck_lwpr") or it.get("low")),
            "close": self._to_float(
                it.get("stck_clpr") or it.get("close") or it.get("stck_prpr")
            ),
            "volume": self._to_int(it.get("acml_vol") or it.get("volume")),
            "source": source,
        }

    def _kis_extract_realtime_from_inquire_price(self, out: dict) -> dict:
        return {
            "date": self._fmt_date(out.get("stck_bsop_date")),
            "open": self._to_float(out.get("stck_oprc")),
            "high": self._to_float(out.get("stck_hgpr")),
            "low": self._to_float(out.get("stck_lwpr")),
            "close": self._to_float(out.get("stck_prpr")),
            "volume": self._to_int(out.get("acml_vol")),
            "source": "inquire-price",
        }

    def fetch_ohlcv_last_n_days_for_top100(
        self,
        ap: ArtifactPaths,
        n_days: int = 365,
        sleep_sec: float = 0.12,
        out_filename: str = "price_ohlcv.csv",
        min_required_hist: int = 365,
    ) -> pd.DataFrame:
        if self.universe_top100 is None or self.universe_top100.empty:
            raise RuntimeError("Top100 universe is empty. Run fetch_top100_by_marcap() first.")
        self._ensure_market_col_from_latest_universe()

        buffer_calendar_days = max(int(n_days * 2.2), n_days + 60)
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=buffer_calendar_days)

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        phase = self._market_phase_kst()
        print(f"🕒 시장 상태(KST): {phase} (pre/open/post)")
        print(f"📆 itemchartprice 조회범위(초기): {start_str} ~ {end_str} (buffer_calendar_days={buffer_calendar_days})")

        rows = []

        for _, r in tqdm(
            self.universe_top100.iterrows(),
            total=len(self.universe_top100),
            ncols=90,
            desc=f"OHLCV(hist {n_days}+today)"
        ):
            name = r["종목명"]
            code = str(r["종목코드"]).zfill(6)
            market = r.get("시장", "")

            try:
                need = n_days + 5
                collected_rows = []
                seen_dates = set()

                end_cursor = end_str
                last_oldest = None
                max_loops = 30

                def _raw_date(it: dict) -> str:
                    return it.get("stck_bsop_date") or it.get("bsop_date") or it.get("date") or ""

                for _loop in range(max_loops):
                    chart_data = self._kis_inquire_daily_itemchartprice(code, start_str, end_cursor)
                    if not chart_data:
                        break

                    batch_dates = [d for d in (_raw_date(it) for it in chart_data) if d]
                    if not batch_dates:
                        break

                    oldest_raw = min(batch_dates)
                    if last_oldest == oldest_raw:
                        break
                    last_oldest = oldest_raw

                    added = 0
                    for it in chart_data:
                        d = _raw_date(it)
                        if not d or d in seen_dates:
                            continue
                        seen_dates.add(d)
                        collected_rows.append(it)
                        added += 1

                    if len(collected_rows) >= need:
                        break
                    if added == 0:
                        break

                    try:
                        oldest_dt = dt.datetime.strptime(oldest_raw, "%Y%m%d").date()
                        end_cursor = (oldest_dt - dt.timedelta(days=1)).strftime("%Y%m%d")
                    except Exception:
                        break

                    time.sleep(0.03)

                if len(collected_rows) == 0:
                    fallback = self._kis_inquire_daily_price_latest_n(code, n=min(30, n_days))
                    daily_ohlcvs = [
                        self._kis_extract_ohlcv_from_daily_row(it, source="inquire-daily-price")
                        for it in fallback
                    ]
                else:
                    daily_ohlcvs = [
                        self._kis_extract_ohlcv_from_daily_row(it, source="inquire-daily-itemchartprice")
                        for it in collected_rows
                    ]

                daily_ohlcvs = sorted(daily_ohlcvs, key=lambda x: x["date"], reverse=True)

                if not daily_ohlcvs:
                    raise RuntimeError("가격 데이터가 비어있음 (itemchartprice + daily-price 모두 실패)")

                latest_daily = daily_ohlcvs[0]

                if phase == "pre":
                    today = latest_daily.copy()
                    today_source = "pre_use_prev_close(latest_daily)"
                elif phase == "open":
                    out = self._kis_inquire_price_domestic(code)
                    today = self._kis_extract_realtime_from_inquire_price(out)
                    today_source = "open_realtime(inquire-price)"
                else:
                    today = latest_daily.copy()
                    today_source = "post_use_today_close(latest_daily)"

                hist_candidates = [x for x in daily_ohlcvs if x["date"] != today["date"]]
                hist = hist_candidates[:n_days]

                if len(hist) < int(min_required_hist):
                    print(f"[INFO] {code}({name}) 데이터 부족({len(hist)}일) → 종목 제외")
                    continue

                for ohlcv in reversed(hist):
                    rows.append({
                        "date": ohlcv["date"],
                        "종목명": name,
                        "종목코드": code,
                        "시장": market,
                        "open": ohlcv["open"],
                        "high": ohlcv["high"],
                        "low": ohlcv["low"],
                        "close": ohlcv["close"],
                        "volume": ohlcv["volume"],
                        "row_type": "hist",
                        "market_phase": phase,
                        "source": ohlcv["source"],
                    })

                rows.append({
                    "date": today["date"],
                    "종목명": name,
                    "종목코드": code,
                    "시장": market,
                    "open": today["open"],
                    "high": today["high"],
                    "low": today["low"],
                    "close": today["close"],
                    "volume": today["volume"],
                    "row_type": "today",
                    "market_phase": phase,
                    "today_source": today_source,
                    "source": today["source"],
                })

            except Exception as e:
                rows.append({
                    "date": dt.date.today().isoformat(),
                    "종목명": name,
                    "종목코드": code,
                    "시장": market,
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "close": 0.0,
                    "volume": 0,
                    "row_type": "ERROR",
                    "market_phase": phase,
                    "source": "ERROR",
                    "error": str(e)[:200],
                })

            time.sleep(sleep_sec)

        df = pd.DataFrame(rows)
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)

        row_order = {"hist": 0, "today": 1, "ERROR": 2}
        df["row_order"] = df["row_type"].map(row_order).fillna(9).astype(int)
        df = df.sort_values(["종목코드", "row_order", "date"], ascending=[True, True, True]).reset_index(drop=True)
        df = df.drop(columns=["row_order"])
        
        path = ap.price_ohlcv()
        df.to_csv(path, index=False, encoding="utf-8-sig")

        return df

    def fetch_foreign_snapshot_today_for_top100(
        self,
        ap: ArtifactPaths,
        sleep_sec: float = 0.05,
        out_filename: str = "price_foreign.csv",
    ) -> pd.DataFrame:
        if self.universe_top100 is None or self.universe_top100.empty:
            raise RuntimeError("Top100 universe is empty. Run fetch_top100_by_marcap() first.")

        rows = []

        for _, r in tqdm(
            self.universe_top100.iterrows(),
            total=len(self.universe_top100),
            ncols=90,
            desc="외국인스냅샷(오늘)"
        ):
            name = r["종목명"]
            code = str(r["종목코드"]).zfill(6)
            market = r.get("시장", "")

            try:
                out = self._kis_inquire_price_domestic(code)
                date_fmt = self._fmt_date(out.get("stck_bsop_date"))
                vol = self._to_int(out.get("acml_vol"))
                frgn_net = self._to_int(out.get("frgn_ntby_qty"))
                frgn_own = self._to_float(out.get("hts_frgn_ehrt"))

                rows.append({
                    "date": date_fmt,
                    "종목명": name,
                    "종목코드": code,
                    "시장": market,
                    "volume": vol,
                    "frgn_ntby_qty": frgn_net,
                    "hts_frgn_ehrt": frgn_own,
                    "foreign_net_flow_ratio": (frgn_net / vol) if vol > 0 else 0.0,
                    "source": "inquire-price",
                })
            except Exception as e:
                rows.append({
                    "date": dt.date.today().isoformat(),
                    "종목명": name,
                    "종목코드": code,
                    "시장": market,
                    "volume": 0,
                    "frgn_ntby_qty": 0,
                    "hts_frgn_ehrt": 0.0,
                    "foreign_net_flow_ratio": 0.0,
                    "source": "ERROR",
                    "error": str(e)[:200],
                })
            time.sleep(sleep_sec)

        df = pd.DataFrame(rows)
        out_path = ap.file("ingest/price", out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 외국인 스냅샷 저장 완료 → {out_path}")
        return df

    # =========================================================
    # 3) Financial
    # =========================================================
    def fetch_financials_topN_from_csv(
        self,
        ap: ArtifactPaths,
        csv_path=None,
        start_year=2015,
        end_year=None,
        fs_div_priority=("CFS", "OFS"),
        reprt_codes=("11011",),
        sleep_sec=0.3,
        save_per_stock=True,
        return_concat=False,
    ):
        if not self.dart_api_key:
            raise RuntimeError("DART API 키가 없습니다.")

        if csv_path is None:
            csv_path = ap.file("ingest/meta", "top100_liquidity_latest.csv")

        if end_year is None:
            end_year = dt.datetime.today().year

        dart.set_api_key(self.dart_api_key)
        corp_list = dart.get_corp_list()

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        df = pd.read_csv(csv_path)
        all_concat = []

        for _, row in tqdm(df.iterrows(), total=len(df), ncols=90, desc="재무제표"):
            name = row["종목명"]
            code = str(row["종목코드"]).zfill(6)

            corp = corp_list.find_by_stock_code(code)
            if not corp:
                continue

            corp_code = corp.corp_code
            parts = []

            for year in range(start_year, end_year + 1):
                for rc in reprt_codes:
                    for fs_div in fs_div_priority:
                        url = (
                            "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?"
                            f"crtfc_key={self.dart_api_key}&corp_code={corp_code}"
                            f"&bsns_year={year}&reprt_code={rc}&fs_div={fs_div}"
                        )

                        try:
                            res = session.get(url, timeout=20).json()
                        except Exception:
                            continue

                        if (not isinstance(res, dict)) or res.get("status") != "000" or "list" not in res:
                            continue

                        part = pd.DataFrame(res["list"])
                        part["종목코드"] = code
                        part["종목명"] = name
                        part["year"] = year
                        part["reprt_code"] = rc
                        part["fs_div"] = fs_div
                        parts.append(part)

                        time.sleep(sleep_sec)
                        break

                    time.sleep(sleep_sec)

            if parts:
                out = pd.concat(parts, ignore_index=True)

                if save_per_stock:
                    safe = self._safe_name(name)
                    out_path = ap.file(
                        "ingest/finance",
                        f"{code}_{safe}_financials_{start_year}_{end_year}.csv",
                    )

                    out.to_csv(out_path, index=False, encoding="utf-8-sig")

                if return_concat:
                    all_concat.append(out)

        if return_concat and all_concat:
            return pd.concat(all_concat, ignore_index=True)

        return pd.DataFrame()

    # =========================================================
    # 4) News
    # =========================================================
    def fetch_news_for_top100_via_naver_api(
        self,
        ap: ArtifactPaths,
        display=100,
        max_pages=10,
        sort="date",
        sleep_sec=0.2,
        save_per_stock=True,
        save_merged=True,
        dedupe_global=False,
        keep_per_stock=50,
        target_fetch=200,
        fill_to_keep=True,
    ) -> pd.DataFrame:
        if self.universe_top100 is None or self.universe_top100.empty:
            raise RuntimeError("Top100 universe is empty. Run fetch_top100_by_marcap() first.")

        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 필요합니다.")

        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

        merged = []
        seen_global = set()

        def _fetch_one_query(query: str, code: str, name: str, seen_stock: set) -> list:
            items_all = []
            start = 1

            for _ in range(max_pages):
                params = {"query": query, "display": display, "start": start, "sort": sort}

                try:
                    resp = self.session.get(url, headers=headers, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    break

                items = data.get("items", [])
                if not items:
                    break

                for it in items:
                    link = it.get("link", "") or ""
                    originallink = it.get("originallink", "") or ""
                    key = link or originallink

                    if key and key in seen_stock:
                        continue
                    if key:
                        seen_stock.add(key)

                    if dedupe_global and key:
                        if key in seen_global:
                            continue
                        seen_global.add(key)

                    items_all.append({
                        "종목코드": code,
                        "종목명": name,
                        "title": self._strip_html(it.get("title", "")),
                        "description": self._strip_html(it.get("description", "")),
                        "link": link,
                        "originallink": originallink,
                        "pubDate": it.get("pubDate", ""),
                        "fetched_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                    })

                start += display
                time.sleep(sleep_sec)

                if target_fetch is not None and len(items_all) >= int(target_fetch):
                    break

            return items_all

        for _, r in tqdm(
            self.universe_top100.iterrows(),
            total=len(self.universe_top100),
            ncols=90,
            desc="뉴스(Top100)"
        ):
            name = r["종목명"]
            code = str(r["종목코드"]).zfill(6)

            seen_stock = set()
            query1 = self._build_news_query(name)
            items_all = _fetch_one_query(query1, code, name, seen_stock)

            if fill_to_keep and keep_per_stock is not None and len(items_all) < int(keep_per_stock):
                items_all += _fetch_one_query(name, code, name, seen_stock)

            df_stock = pd.DataFrame(items_all)
            if not df_stock.empty:
                df_stock["pub_dt"] = pd.to_datetime(df_stock["pubDate"], errors="coerce", utc=True)
                df_stock = (
                    df_stock.sort_values("pub_dt", ascending=False, na_position="last")
                    .head(keep_per_stock if keep_per_stock is not None else len(df_stock))
                    .drop(columns=["pub_dt"])
                    .reset_index(drop=True)
                )

            if save_per_stock:
                safe = self._safe_name(name)
                out_path = ap.file("ingest/news", f"{code}_{safe}_naver_news.csv")
                df_stock.to_csv(out_path, index=False, encoding="utf-8-sig")

            merged.append(df_stock)
            time.sleep(sleep_sec)

        merged_df = (
            pd.concat([d for d in merged if d is not None and not d.empty], ignore_index=True)
            if merged else pd.DataFrame()
        )

        if save_merged:
            out_path = ap.file("ingest/news", "top100_naver_news_merged.csv")
            merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"🧾 Top100 뉴스 합본 저장 → {out_path} (rows={len(merged_df)})")

        return merged_df

    # =========================================================
    # core-compatible run
    # =========================================================
    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        ctx.logger.info(f"run_id={ctx.run_id}")
        ctx.logger.info(f"artifact_root={ctx.artifact_root}")

        summary: Dict[str, Any] = {
            "run_id": ctx.run_id,
            "artifact_root": str(ctx.artifact_root),
            "stage_name": self.stage,
            "asof_date": ctx.asof_date,
            "universe": ctx.universe,
        }

        outputs = []

        try:
            # -------------------------
            # user
            # -------------------------
            if self.user_csv_map:
                saved_user_paths = self.ingest_user_data(self.user_csv_map, ap)
                summary["user_files_saved"] = saved_user_paths
                outputs.extend(saved_user_paths.values())

            # -------------------------
            # Top100
            # -------------------------
            top100 = self.fetch_top100_by_marcap(
                ap=ap,
                top_n=100,
                save_snapshot=True,
                metric="trade_value"
            )

            summary["top100_rows"] = len(top100)

            top_path = ap.file("ingest/meta", "top100_liquidity_latest.csv")
            summary["top100_path"] = str(top_path)
            outputs.append(str(top_path))

            # -------------------------
            # OHLCV
            # -------------------------
            ohlcv_df = self.fetch_ohlcv_last_n_days_for_top100(
                ap=ap,
                n_days=365,
                sleep_sec=0.12,
                out_filename="price_ohlcv.csv",
                min_required_hist=365,
            )

            summary["ohlcv_rows"] = len(ohlcv_df)

            ohlcv_path = ap.file("ingest/price", "price_ohlcv.csv")
            summary["ohlcv_path"] = str(ohlcv_path)
            outputs.append(str(ohlcv_path))

            # -------------------------
            # Foreign
            # -------------------------
            foreign_df = self.fetch_foreign_snapshot_today_for_top100(
                ap=ap
            )

            summary["foreign_rows"] = len(foreign_df)

            foreign_path = ap.file("ingest/price", "price_foreign.csv")
            summary["foreign_path"] = str(foreign_path)
            outputs.append(str(foreign_path))

            # -------------------------
            # Financial
            # -------------------------
            fin = self.fetch_financials_topN_from_csv(
                ap=ap,
                start_year=2015,
                reprt_codes=("11011",),
                save_per_stock=True,
                return_concat=False,
            )

            summary["financials_rows"] = len(fin)

            finance_dir = ap.stage_dir("ingest/finance")
            summary["finance_dir"] = str(finance_dir)
            outputs.append(str(finance_dir))

            # -------------------------
            # News
            # -------------------------
            news = self.fetch_news_for_top100_via_naver_api(
                ap=ap,
                display=50,
                max_pages=2,
                keep_per_stock=50,
                dedupe_global=False,
            )

            summary["news_rows"] = len(news)

            news_path = ap.file("ingest/news", "top100_naver_news_merged.csv")
            summary["news_merged_path"] = str(news_path)
            outputs.append(str(news_path))

            return StageResult.success(
                stage=self.stage,
                outputs=outputs,
                metrics=summary,
            )

        except Exception as e:
            summary["fatal_error"] = str(e)
            traceback.print_exc()
            raise