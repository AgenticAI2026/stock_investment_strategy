# 뉴스·시세·재무 수집 총괄 agent
from __future__ import annotations

import os
import re
import json
import time
import shutil
import datetime as dt
from html import unescape
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import dart_fss as dart
from tqdm import tqdm
from zoneinfo import ZoneInfo

from utils.env_loader import load_env


class DataIngestionAgent:
    # =========================================================
    # ✅ KIS endpoints / TR IDs (환경에 따라 조정 가능)
    # =========================================================
    KIS_VOLUME_RANK_ENDPOINT = "/uapi/domestic-stock/v1/quotations/volume-rank"
    KIS_VOLUME_RANK_TR_ID = "FHPST01710000"

    KIS_SEARCH_STOCK_INFO_ENDPOINT = "/uapi/domestic-stock/v1/quotations/search-stock-info"
    KIS_SEARCH_STOCK_INFO_TR_ID = "CTPF1604R"

    def __init__(
        self,
        dart_api_key: Optional[str] = None,
        user_csv_map: Optional[Dict[str, str]] = None,
        data_dir: str = "data/raw",              # 원천 데이터 읽기 기준
        artifacts_root: str = "artifacts",       # 실행 결과 저장 루트
        run_id: Optional[str] = None,            # 전체 파이프라인에서 주입 권장
        stage_name: str = "ingest",              # 현재 agent stage
    ):
        # ✅ .env 로드 (프로젝트 루트)
        load_env()

        self.data_dir = data_dir
        self.user_csv_map = user_csv_map or {}

        # Keys (가능하면 .env에서 읽고, 필요 시 인자로 override)
        self.dart_api_key = dart_api_key or os.getenv("DART_API_KEY")
        self.naver_client_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")

        # -------------------------
        # run_id / artifacts 경로
        # -------------------------
        self.artifacts_root = artifacts_root
        self.stage_name = stage_name
        self.run_id = run_id or dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%dT%H%M%S")

        self.run_root = os.path.join(self.artifacts_root, f"run_id={self.run_id}")
        self.stage_root = os.path.join(self.run_root, self.stage_name)

        self.user_dir = os.path.join(self.stage_root, "user")
        self.meta_dir = os.path.join(self.stage_root, "meta")
        self.price_dir = os.path.join(self.stage_root, "price")
        self.finance_dir = os.path.join(self.stage_root, "finance")
        self.news_dir = os.path.join(self.stage_root, "news")

        os.makedirs(self.user_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.price_dir, exist_ok=True)
        os.makedirs(self.finance_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)

        # 세션
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

        # Universe cache
        self.universe_top100: Optional[pd.DataFrame] = None

        # KIS (한국투자증권)
        self.kis_app_key = os.getenv("KIS_APP_KEY")
        self.kis_app_secret = os.getenv("KIS_APP_SECRET")
        self.kis_base_url = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")

        self._kis_access_token: Optional[str] = None
        self._kis_token_expire_at: Optional[dt.datetime] = None

        # Warnings
        if self.naver_client_id is None or self.naver_client_secret is None:
            print("[WARN] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 설정되지 않았습니다. 뉴스 수집이 실패할 수 있어요.")
        if self.dart_api_key is None:
            print("[WARN] DART_API_KEY 가 설정되지 않았습니다. 재무 수집이 실패할 수 있어요.")
        if self.kis_app_key is None or self.kis_app_secret is None:
            print("[WARN] KIS_APP_KEY / KIS_APP_SECRET 이 설정되지 않았습니다. 시세 수집이 실패할 수 있어요.")

    # --------------------------
    # util (뉴스/파일명/경로)
    # --------------------------
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

    def _artifact_path(self, category: str, filename: str) -> str:
        mapping = {
            "user": self.user_dir,
            "meta": self.meta_dir,
            "price": self.price_dir,
            "finance": self.finance_dir,
            "news": self.news_dir,
        }
        if category not in mapping:
            raise ValueError(f"Unknown category: {category}")
        return os.path.join(mapping[category], filename)

    def _latest_universe_csv_path(self) -> str:
        return self._artifact_path("meta", "top100_liquidity_latest.csv")

    def _save_run_summary(self, summary: Dict[str, Any]) -> None:
        out_path = os.path.join(self.stage_root, "ingest_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"🧾 ingest summary 저장 → {out_path}")

    # --------------------------
    # 0) 사용자 데이터 로드/저장
    # --------------------------
    def ingest_user_data(self, user_csv_map: dict) -> dict:
        saved_paths = {}

        for role, csv_path in user_csv_map.items():
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"{role} CSV not found: {csv_path}")

            dst_path = self._artifact_path("user", f"user_{role}.csv")
            shutil.copy2(csv_path, dst_path)

            print(f"✅ 사용자 데이터 저장 완료 → {dst_path}")
            saved_paths[role] = dst_path

        return saved_paths

    # =========================================================
    # ✅ KIS 토큰/헤더 + 공통 request
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
        self._kis_token_expire_at = dt.datetime.now() + dt.timedelta(seconds=max(expires_in - 60, 60))
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
                    method, url, headers=headers, params=params, json=json, timeout=timeout
                )

                # ✅ 2xx/4xx는 바로 raise_for_status()로 원인 잡기
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp

                # ✅ 5xx는 last_err에 상태/본문을 넣고 재시도
                snippet = (resp.text or "")[:300].replace("\n", " ")
                last_err = RuntimeError(f"HTTP {resp.status_code} from {url} | body={snippet}")

            except Exception as e:
                last_err = e

            time.sleep(base_sleep * (2 ** i))

        raise RuntimeError(f"request failed after retries: {last_err}")

    def _pick_list(self, data: dict) -> list:
        for k in ("output", "output1", "output2"):
            if k in data and data[k]:
                return data[k]
        return []

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

    # =========================================================
    # ✅ 1) Top100 유니버스: KIS 유동성(거래대금/거래량) 기반
    # =========================================================
    def _kis_fetch_volume_rank(
        self,
        *,
        metric: str = "trade_value",
        fid_input_iscd: str = "0000",
        price1: str = "",
        price2: str = "",
    ) -> pd.DataFrame:
        """
        metric:
        - trade_value  : 거래대금 기준
        - trade_volume : 거래량 기준
        """
        blng = "3" if metric == "trade_value" else "0"  # 3=거래대금 / 0=거래량

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
            raise RuntimeError(f"volume-rank 실패: msg_cd={data.get('msg_cd')} msg1={data.get('msg1')}")

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

    def _kis_get_stock_market_cached(self, stock_code: str, cache: dict) -> str:
        """
        search-stock-info로 KOSPI/KOSDAQ 판별 (캐시)
        """
        stock_code = str(stock_code).zfill(6)
        if stock_code in cache:
            return cache[stock_code]

        url = f"{self.kis_base_url}{self.KIS_SEARCH_STOCK_INFO_ENDPOINT}"
        headers = self._kis_headers(tr_id=self.KIS_SEARCH_STOCK_INFO_TR_ID)
        params = {"PRDT_TYPE_CD": "300", "PDNO": stock_code}

        resp = self._request_with_retry("GET", url, headers=headers, params=params, timeout=15)
        data = resp.json()

        if str(data.get("rt_cd")) != "0":
            cache[stock_code] = ""
            return ""

        out = data.get("output", {}) or {}
        kosdaq_dt = str(out.get("kosdaq_mket_lstg_dt", "") or "").strip()
        kospi_dt = str(out.get("scts_mket_lstg_dt", "") or "").strip()

        if kosdaq_dt:
            cache[stock_code] = "KOSDAQ"
        elif kospi_dt:
            cache[stock_code] = "KOSPI"
        else:
            cache[stock_code] = ""

        return cache[stock_code]

    def fetch_top100_by_marcap(
        self,
        top_n: int = 100,
        save_snapshot: bool = True,
        metric: str = "trade_value",
    ) -> pd.DataFrame:
        print(f"🔍 KIS 유동성 Top{top_n} 산출 (volume-rank 다회 호출로 후보풀 확장), metric={metric}")

        sort_col = "trade_value" if metric == "trade_value" else "trade_volume"

        price_bins = [
            ("", ""),
            ("0", "5000"),
            ("5000", "20000"),
            ("20000", "100000"),
            ("100000", "500000"),
            ("500000", ""),
        ]

        markets = [
            ("KOSPI", "0001"),
            ("KOSDAQ", "1001"),
        ]

        best: Dict[str, dict] = {}

        for mkt_name, fid_iscd in markets:
            for p1, p2 in price_bins:
                df = self._kis_fetch_volume_rank(metric=metric, fid_input_iscd=fid_iscd, price1=p1, price2=p2)
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
            raise RuntimeError("volume-rank 다회 호출 후에도 pool이 비었습니다. (파라미터/권한/장상태 확인 필요)")

        top = (
            pool.sort_values(sort_col, ascending=False)
            .head(top_n)
            .loc[:, ["종목명", "종목코드", "시장"]]
            .reset_index(drop=True)
        )
        top["종목코드"] = top["종목코드"].astype(str).str.zfill(6)

        self.universe_top100 = top.copy()

        latest_path = self._artifact_path("meta", f"top{top_n}_liquidity_latest.csv")
        top.to_csv(latest_path, index=False, encoding="utf-8-sig")

        if save_snapshot:
            stamp = dt.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
            snap_path = self._artifact_path("meta", f"top{top_n}_liquidity_{stamp}.csv")
            top.to_csv(snap_path, index=False, encoding="utf-8-sig")
            print(f"✅ Top{top_n} 저장 완료 → {latest_path} / {snap_path}")
        else:
            print(f"✅ Top{top_n} 저장 완료 → {latest_path}")

        print(f"pool unique codes = {len(pool)} / top rows = {len(top)}")
        return top

    # --------------------------
    # 유니버스 시장 컬럼 보강
    # --------------------------
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

    # --------------------------
    # 시장 상태
    # --------------------------
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

    # --------------------------
    # (A) 최신 30개(제한) 일봉: inquire-daily-price
    # --------------------------
    def _kis_inquire_daily_price_list(self, stock_code: str) -> List[Dict[str, Any]]:
        url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = self._kis_headers(tr_id="FHKST01010400")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(stock_code).zfill(6),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }

        resp = self._request_with_retry(
            method="GET",
            url=url,
            headers=headers,
            params=params,
            timeout=15,
        )
        data = resp.json()
        out_list = self._pick_list(data)
        if not out_list:
            raise RuntimeError(f"inquire-daily-price 응답 이상: {data}")
        return out_list

    def _kis_inquire_daily_price_latest_n(self, stock_code: str, n: int = 20) -> List[Dict[str, Any]]:
        out_list = self._kis_inquire_daily_price_list(stock_code)

        def _get_date(x):
            return x.get("stck_bsop_date") or x.get("bsop_date") or x.get("date") or ""

        out_list_sorted = sorted(out_list, key=_get_date, reverse=True)
        return out_list_sorted[:n]

    # --------------------------
    # (B) 장기 일봉: inquire-daily-itemchartprice
    # --------------------------
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
            msg_cd = data.get("msg_cd")
            msg1 = data.get("msg1")
            rt_cd = data.get("rt_cd")
            print(f"[WARN] itemchartprice output2 empty: code={stock_code}, rt_cd={rt_cd}, msg_cd={msg_cd}, msg1={msg1}")
            return []

        if not isinstance(out_list, list):
            keys = list(data.keys())
            raise RuntimeError(f"itemchartprice output2 형식이상: keys={keys}, sample={str(data)[:200]}")

        return out_list

    # --------------------------
    # 파서
    # --------------------------
    def _fmt_date(self, yyyymmdd: Optional[str]) -> str:
        if yyyymmdd and len(yyyymmdd) == 8:
            return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
        return dt.date.today().isoformat()

    def _kis_extract_ohlcv_from_daily_row(self, it: dict, source: str) -> dict:
        def _get_date(x):
            return x.get("stck_bsop_date") or x.get("bsop_date") or x.get("date") or ""

        d = self._fmt_date(_get_date(it))

        o = it.get("stck_oprc") or it.get("open")
        h = it.get("stck_hgpr") or it.get("high")
        l = it.get("stck_lwpr") or it.get("low")
        c = it.get("stck_clpr") or it.get("close") or it.get("stck_prpr")
        v = it.get("acml_vol") or it.get("volume")

        return {
            "date": d,
            "open": self._to_float(o),
            "high": self._to_float(h),
            "low": self._to_float(l),
            "close": self._to_float(c),
            "volume": self._to_int(v),
            "source": source,
        }

    def _kis_extract_realtime_from_inquire_price(self, out: dict) -> dict:
        date_fmt = self._fmt_date(out.get("stck_bsop_date"))
        return {
            "date": date_fmt,
            "open": self._to_float(out.get("stck_oprc")),
            "high": self._to_float(out.get("stck_hgpr")),
            "low": self._to_float(out.get("stck_lwpr")),
            "close": self._to_float(out.get("stck_prpr")),
            "volume": self._to_int(out.get("acml_vol")),
            "source": "inquire-price",
        }

    # --------------------------
    # 오늘 외국인 스냅샷
    # --------------------------
    def _kis_inquire_price_domestic(self) -> Dict[str, Any]:
        raise NotImplementedError("Use _kis_inquire_price_domestic(stock_code)")

    def _kis_inquire_price_domestic(self, stock_code: str) -> Dict[str, Any]:
        url = f"{self.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self._kis_headers(tr_id="FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": str(stock_code).zfill(6),
        }
        resp = self._request_with_retry(
            method="GET",
            url=url,
            headers=headers,
            params=params,
            timeout=15,
        )

        data = resp.json()
        if "output" not in data:
            raise RuntimeError(f"inquire-price 응답 이상: {data}")
        return data["output"]

    # --------------------------
    # main: OHLCV hist(n_days) + today(as-of)
    # --------------------------
    def fetch_ohlcv_last_n_days_for_top100(
        self,
        n_days: int = 365,
        sleep_sec: float = 0.12,
        out_filename: str = "top100_ohlcv_last365.csv",
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

        out_path = self._artifact_path("price", out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ OHLCV(hist {n_days}+today) 저장 완료 → {out_path} (rows={len(df)})")
        return df

    def fetch_foreign_snapshot_today_for_top100(
        self,
        sleep_sec: float = 0.05,
        out_filename: str = "top100_foreign_snapshot_today.csv",
    ) -> pd.DataFrame:
        if self.universe_top100 is None or self.universe_top100.empty:
            raise RuntimeError("Top100 universe is empty. Run fetch_top100_by_marcap() first.")
        self._ensure_market_col_from_latest_universe()

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

        out_path = self._artifact_path("price", out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 외국인 스냅샷(오늘) 저장 완료 → {out_path} (rows={len(df)})")
        return df

    def fetch_prices_minimal_bundle_top100(self, n_days: int = 365) -> dict:
        ohlcv_df = self.fetch_ohlcv_last_n_days_for_top100(n_days=n_days)
        foreign_df = self.fetch_foreign_snapshot_today_for_top100()
        return {"ohlcv_last_n_days": ohlcv_df, "foreign_snapshot_today": foreign_df}

    # --------------------------
    # 3) 재무제표 수집
    # --------------------------
    def fetch_financials_topN_from_csv(
        self,
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
            raise RuntimeError("❌ DART API 키가 없습니다.")

        if csv_path is None:
            csv_path = self._latest_universe_csv_path()

        if end_year is None:
            end_year = dt.datetime.today().year

        dart.set_api_key(self.dart_api_key)
        corp_list = dart.get_corp_list()

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        df = pd.read_csv(csv_path)
        os.makedirs(self.finance_dir, exist_ok=True)

        report_labels = {
            "11011": "사업보고서(연간)",
            "11012": "반기보고서",
            "11013": "1분기보고서",
            "11014": "3분기보고서",
        }

        all_concat = []

        for _, row in tqdm(df.iterrows(), total=len(df), ncols=90, desc="재무제표"):
            name = row["종목명"]
            code = str(row["종목코드"]).zfill(6)

            corp = corp_list.find_by_stock_code(code)
            if not corp:
                print(f"⚠️ [{name}] corp_code 매핑 실패 → skip")
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
                        except Exception as e:
                            print(f"❌ [{name}] {year} {report_labels.get(rc, rc)} {fs_div} 요청 오류: {e}")
                            continue

                        if (not isinstance(res, dict)) or res.get("status") != "000" or "list" not in res:
                            continue

                        part = pd.DataFrame(res["list"])
                        part["종목코드"] = code
                        part["종목명"] = name
                        part["year"] = year
                        part["reprt_code"] = rc
                        part["reprt_label"] = report_labels.get(rc, rc)
                        part["fs_div"] = fs_div
                        parts.append(part)

                        time.sleep(sleep_sec)
                        break  # CFS 성공 시 OFS 생략

                    time.sleep(sleep_sec)

            if parts:
                out = pd.concat(parts, ignore_index=True)

                if save_per_stock:
                    safe = self._safe_name(name)
                    out_path = self._artifact_path("finance", f"{code}_{safe}_financials_{start_year}_{end_year}.csv")
                    out.to_csv(out_path, index=False, encoding="utf-8-sig")
                    print(f"✅ [{name}] 저장 → {out_path} (rows={len(out)})")

                if return_concat:
                    all_concat.append(out)
            else:
                print(f"⚠️ [{name}] {start_year}~{end_year} 재무제표 데이터 없음")

        if return_concat and all_concat:
            return pd.concat(all_concat, ignore_index=True)

        return pd.DataFrame()

    # --------------------------
    # 4) 네이버 뉴스 검색 API 수집 (Top100 기준)
    # --------------------------
    def fetch_news_for_top100_via_naver_api(
        self,
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
            raise RuntimeError("환경변수 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 필요합니다.")

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
                except Exception as e:
                    print(f"⚠️ 뉴스 호출 실패: {code}({name}) / {query} - {e}")
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
                query2 = name
                items_all += _fetch_one_query(query2, code, name, seen_stock)

            df_stock = pd.DataFrame(items_all)

            base_cols = ["종목코드", "종목명", "title", "description", "link", "originallink", "pubDate", "fetched_at"]
            if df_stock.empty:
                df_stock = pd.DataFrame(columns=base_cols)
            else:
                df_stock["pub_dt"] = pd.to_datetime(df_stock["pubDate"], errors="coerce", utc=True)
                df_stock = (
                    df_stock.sort_values("pub_dt", ascending=False, na_position="last")
                    .head(keep_per_stock if keep_per_stock is not None else len(df_stock))
                    .drop(columns=["pub_dt"])
                    .reset_index(drop=True)
                )

            if save_per_stock:
                safe = self._safe_name(name)
                out_path = self._artifact_path("news", f"{code}_{safe}_naver_news.csv")
                df_stock.to_csv(out_path, index=False, encoding="utf-8-sig")

            merged.append(df_stock)
            time.sleep(sleep_sec)

        merged_df = (
            pd.concat([d for d in merged if d is not None and not d.empty], ignore_index=True)
            if merged else pd.DataFrame()
        )

        if save_merged:
            out_path = self._artifact_path("news", "top100_naver_news_merged.csv")
            merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"🧾 Top100 뉴스 합본 저장 → {out_path} (rows={len(merged_df)})")

        return merged_df

    # --------------------------
    # run() - 실행 엔트리
    # --------------------------
    def run(self) -> dict:
        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_root": self.run_root,
            "stage_root": self.stage_root,
            "stage_name": self.stage_name,
        }

        # 0) user
        try:
            if self.user_csv_map:
                saved_user_paths = self.ingest_user_data(self.user_csv_map)
                summary["user_files_saved"] = saved_user_paths
        except Exception as e:
            summary["user_error"] = str(e)

        # 1) top100
        try:
            top100 = self.fetch_top100_by_marcap(top_n=100, save_snapshot=True, metric="trade_value")
            summary["top100_rows"] = len(top100)
            summary["top100_path"] = self._latest_universe_csv_path()
        except Exception as e:
            summary["top100_error"] = str(e)
            self._save_run_summary(summary)
            raise

        # 2) prices
        try:
            ohlcv_df = self.fetch_ohlcv_last_n_days_for_top100(
                n_days=365,
                sleep_sec=0.12,
                out_filename="top100_ohlcv_last365.csv",
                min_required_hist=365,
            )
            summary["ohlcv_hist_today_rows"] = len(ohlcv_df)
            summary["ohlcv_path"] = self._artifact_path("price", "top100_ohlcv_last365.csv")
        except Exception as e:
            summary["ohlcv_error"] = str(e)

        try:
            foreign_df = self.fetch_foreign_snapshot_today_for_top100()
            summary["foreign_snapshot_today_rows"] = len(foreign_df)
            summary["foreign_path"] = self._artifact_path("price", "top100_foreign_snapshot_today.csv")
        except Exception as e:
            summary["foreign_error"] = str(e)

        # 3) financials
        try:
            fin = self.fetch_financials_topN_from_csv(
                csv_path=self._latest_universe_csv_path(),
                start_year=2015,
                reprt_codes=("11011",),
                save_per_stock=True,
                return_concat=False,
            )
            summary["financials_concat_rows"] = len(fin)
            summary["finance_dir"] = self.finance_dir
        except Exception as e:
            summary["financials_error"] = str(e)

        # 4) news
        try:
            news = self.fetch_news_for_top100_via_naver_api(
                display=50,
                max_pages=2,
                keep_per_stock=50,
                dedupe_global=False,
            )
            summary["news_rows"] = len(news)
            summary["news_merged_path"] = self._artifact_path("news", "top100_naver_news_merged.csv")
        except Exception as e:
            summary["news_error"] = str(e)

        self._save_run_summary(summary)
        print("✅ DataIngestionAgent run() 완료:", summary)
        return summary