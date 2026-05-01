from __future__ import annotations

import os
import re
import json
import glob
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


class NewsInvestigationAgent(BaseAgent):
    stage = "news_invest_rag"

    def __init__(
        self,
        base_dir: str = "",
        output_dir: str = "",
        news_raw_path: Optional[str] = None,
        news_features_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
    ):
        self.base_dir = base_dir or ""
        self.output_dir = output_dir or ""

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.news_raw_path = self._abspath(news_raw_path) if news_raw_path else None
        self.news_features_path = self._abspath(news_features_path) if news_features_path else None
        self.output_path = self._abspath(output_json_path) if output_json_path else None

        self.thresh_high = 65.0
        self.thresh_med = 45.0

    # =========================================================
    # Pipeline entry
    # =========================================================

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        try:
            runtime_agent = self if self._is_configured() else self._build_runtime_agent(ctx, ap)
            outputs = runtime_agent.run()

            return StageResult.success(
                stage=self.stage,
                outputs=[outputs["news_invest_rag_result"]],
            )
        except Exception as e:
            ctx.logger.exception("❌ News Investigation Agent Failed")
            return StageResult.failed(stage=self.stage, error=str(e))

    def _is_configured(self) -> bool:
        return all([
            self.news_raw_path,
            self.news_features_path,
            self.output_path,
        ])

    def _build_runtime_agent(self, ctx: RunContext, ap: ArtifactPaths) -> "NewsInvestigationAgent":
        prep_apply_dir = Path(ap.stage_dir("prep_apply"))
        stage_dir = Path(ap.stage_dir(self.stage))
        stage_dir.mkdir(parents=True, exist_ok=True)

        if not prep_apply_dir.exists():
            raise FileNotFoundError(f"prep_apply 디렉토리를 찾지 못했습니다: {prep_apply_dir}")

        news_raw_path = self._pick_latest_matching_file(
            str(prep_apply_dir / "preprocessed__*__news__news_raw_merged.csv")
        )

        news_features_path = self._pick_latest_matching_file(
            str(prep_apply_dir / "preprocessed__*__news__news_features_by_stock.csv")
        )

        if not news_raw_path:
            raise FileNotFoundError(f"news_raw_merged 파일을 찾지 못했습니다: {prep_apply_dir}")

        if not news_features_path:
            raise FileNotFoundError(f"news_features_by_stock 파일을 찾지 못했습니다: {prep_apply_dir}")

        output_json_path = stage_dir / "news_invest_rag_result.json"

        ctx.logger.info("news_raw_path=%s", news_raw_path)
        ctx.logger.info("news_features_path=%s", news_features_path)
        ctx.logger.info("output_json_path=%s", output_json_path)

        return NewsInvestigationAgent(
            base_dir=self.base_dir,
            output_dir=str(stage_dir),
            news_raw_path=str(news_raw_path),
            news_features_path=str(news_features_path),
            output_json_path=str(output_json_path),
        )

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _pick_latest_matching_file(pattern: str) -> Optional[str]:
        matches = glob.glob(pattern)
        if not matches:
            return None
        return sorted(matches, key=os.path.getmtime, reverse=True)[0]

    def _abspath(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        if self.base_dir:
            return os.path.abspath(os.path.join(self.base_dir, path))
        return os.path.abspath(path)

    @staticmethod
    def make_unique_columns(cols):
        seen = {}
        out = []
        for c in cols:
            c = str(c)
            if c not in seen:
                seen[c] = 0
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}__dup{seen[c]}")
        return out

    @staticmethod
    def drop_duplicated_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, ~df.columns.duplicated()].copy()

    @staticmethod
    def zfill6(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s.zfill(6)

    @staticmethod
    def normalize_text(s):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        s = str(s).strip()
        return re.sub(r"\s+", " ", s)

    @staticmethod
    def clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def safe_float(x):
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    @staticmethod
    def safe_mean(arr):
        s = pd.Series(arr).dropna()
        return float(s.mean()) if len(s) else None

    def pick_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = list(df.columns)
        colset = set(cols)

        for c in candidates:
            if c in colset:
                return c

        low_map = {c.lower(): c for c in cols}
        for c in candidates:
            if c.lower() in low_map:
                return low_map[c.lower()]

        for cand in candidates:
            cl = cand.lower()
            for c in cols:
                if cl in c.lower():
                    return c

        return None

    @staticmethod
    def parse_datetime_col(df: pd.DataFrame, colname: Optional[str]) -> pd.Series:
        if colname is None:
            return pd.Series([pd.NaT] * len(df), index=df.index)
        return pd.to_datetime(df[colname], errors="coerce")

    def normalize_title_for_dedupe(self, title):
        t = self.normalize_text(title).lower()
        t = re.sub(r"[\[\]\(\)\{\}]", " ", t)
        t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def press_from_url(url):
        if url is None or (isinstance(url, float) and np.isnan(url)):
            return None

        u = str(url).strip()
        if not u:
            return None

        try:
            netloc = urlparse(u).netloc.lower().replace("www.", "")
            if not netloc:
                return None

            mapping = {
                "news.naver.com": "Naver News",
                "n.news.naver.com": "Naver News",
                "m.news.naver.com": "Naver News",
                "finance.naver.com": "Naver Finance",
                "edaily.co.kr": "이데일리",
                "hankyung.com": "한국경제",
                "mk.co.kr": "매일경제",
                "yna.co.kr": "연합뉴스",
                "sedaily.com": "서울경제",
                "biz.chosun.com": "조선비즈",
                "fnnews.com": "파이낸셜뉴스",
                "news.mt.co.kr": "머니투데이",
                "etnews.com": "전자신문",
            }

            if netloc in mapping:
                return mapping[netloc]

            for k, v in mapping.items():
                if netloc.endswith(k):
                    return v

            return netloc
        except Exception:
            return None

    # =========================================================
    # Scoring
    # =========================================================

    def rule_score_article(self, title, desc):
        noise_patterns = [
            r"주가", r"급등", r"급락", r"상한가", r"하한가", r"왜", r"이유", r"전망",
            r"관련주", r"테마", r"수혜주", r"대장주", r"오늘", r"장중", r"마감",
            r"개미", r"차트", r"기술적", r"PER", r"PBR",
        ]

        high_impact_patterns = [
            r"실적", r"잠정", r"어닝", r"흑자전환", r"적자", r"매출", r"영업이익", r"순이익",
            r"공시", r"증자", r"유상증자", r"무상증자", r"감자",
            r"수주", r"계약", r"공급계약", r"MOU", r"투자", r"합작", r"JV",
            r"인수", r"매각", r"M&A", r"스핀오프",
            r"소송", r"피소", r"규제", r"제재", r"리콜", r"사고", r"횡령", r"배임",
            r"배당", r"자사주", r"소각", r"분할", r"상장", r"상폐", r"거래정지",
        ]

        text = f"{self.normalize_text(title)} {self.normalize_text(desc)}"

        noise = any(re.search(p, text) for p in noise_patterns)
        high = any(re.search(p, text) for p in high_impact_patterns)

        score = 0.15
        if high:
            score += 0.70
        if noise:
            score -= 0.25

        return self.clip01(score), noise, high

    # =========================================================
    # Main
    # =========================================================

    def run(self) -> Dict[str, Any]:
        if not self.news_raw_path:
            raise ValueError("news_raw_path가 설정되지 않았습니다.")
        if not self.news_features_path:
            raise ValueError("news_features_path가 설정되지 않았습니다.")
        if not self.output_path:
            raise ValueError("output_json_path가 설정되지 않았습니다.")

        news_raw = pd.read_csv(self.news_raw_path)
        news_feat = pd.read_csv(self.news_features_path)

        news_raw.columns = self.make_unique_columns(news_raw.columns)
        news_feat.columns = self.make_unique_columns(news_feat.columns)

        news_raw = self.drop_duplicated_columns(news_raw)
        news_feat = self.drop_duplicated_columns(news_feat)

        ticker_col = self.pick_col(news_raw, ["종목코드", "ticker", "code", "stock_code", "symbol"])
        date_col = self.pick_col(news_raw, ["date", "pubDate", "published_at", "publishedAt", "datetime", "created_at"])
        title_col = self.pick_col(news_raw, ["title", "제목", "news_title"])
        desc_col = self.pick_col(news_raw, ["description", "desc", "summary", "요약", "content", "본문요약"])
        url_col = self.pick_col(news_raw, ["link", "url", "originallink", "original_link"])
        press_col = self.pick_col(news_raw, ["press", "media", "source", "언론사"])

        if ticker_col is None:
            raise ValueError(f"[news_raw] ticker column not found. cols={list(news_raw.columns)[:40]}")

        news_raw[ticker_col] = news_raw[ticker_col].apply(self.zfill6)
        news_raw["__ni_dt"] = self.parse_datetime_col(news_raw, date_col)

        if news_raw["__ni_dt"].notna().any():
            as_of_date = str(news_raw["__ni_dt"].max().date())
        else:
            as_of_date = datetime.now().date().isoformat()

        as_of_dt = pd.to_datetime(as_of_date)

        feat_ticker_col = self.pick_col(news_feat, ["종목코드", "ticker", "code", "stock_code", "symbol"])
        feat_map = {}

        if feat_ticker_col:
            news_feat[feat_ticker_col] = news_feat[feat_ticker_col].apply(self.zfill6)
            feat_map = news_feat.set_index(feat_ticker_col).to_dict(orient="index")

        results = []
        tickers = sorted(news_raw[ticker_col].dropna().unique().tolist())

        for tk in tickers:
            df = news_raw[news_raw[ticker_col] == tk].copy()
            if len(df) == 0:
                continue

            df["__ni_days_ago"] = (as_of_dt - df["__ni_dt"]).dt.days

            df30 = df[df["__ni_days_ago"].between(0, 30, inclusive="both")].copy()
            df7 = df[df["__ni_days_ago"].between(0, 7, inclusive="both")].copy()

            calc_df = self.drop_duplicated_columns(df30.copy())

            calc_df["__ni_title"] = calc_df[title_col].apply(self.normalize_text) if title_col else ""
            calc_df["__ni_desc"] = calc_df[desc_col].apply(self.normalize_text) if desc_col else ""
            calc_df["__ni_norm_title"] = calc_df["__ni_title"].apply(self.normalize_title_for_dedupe)

            scores = calc_df.apply(
                lambda r: self.rule_score_article(r["__ni_title"], r["__ni_desc"]),
                axis=1,
            )

            calc_df["__ni_rule"] = [s[0] for s in scores]
            calc_df["__ni_noise"] = [s[1] for s in scores]
            calc_df["__ni_high"] = [s[2] for s in scores]

            calc_df = self.drop_duplicated_columns(calc_df)

            if len(calc_df):
                calc_df_sorted = calc_df.sort_values(
                    ["__ni_norm_title", "__ni_rule", "__ni_dt"],
                    ascending=[True, False, False],
                )
                df30_u = calc_df_sorted.drop_duplicates(subset=["__ni_norm_title"], keep="first").copy()
            else:
                df30_u = calc_df.copy()

            n30_raw = int(len(df30))
            n30_unique = int(len(df30_u))

            if len(df7) and title_col:
                df7["__ni_title"] = df7[title_col].apply(self.normalize_text)
                df7["__ni_norm_title"] = df7["__ni_title"].apply(self.normalize_title_for_dedupe)
                n7_unique = int(df7["__ni_norm_title"].nunique())
            else:
                n7_unique = 0

            article_quality = self.safe_mean(df30_u["__ni_rule"].tolist()) if n30_unique else None
            noise_ratio = float(df30_u["__ni_noise"].mean()) if n30_unique else 0.0
            high_ratio = float(df30_u["__ni_high"].mean()) if n30_unique else 0.0
            novelty = float(n30_unique / n30_raw) if n30_raw > 0 else None

            volume_score = self.clip01(np.log1p(n7_unique) / np.log1p(20))

            feat = feat_map.get(tk, {}) or {}
            feat_news_cnt = None

            for key in ["news_cnt", "newsCount", "cnt", "count", "article_cnt", "n_articles", "news_count"]:
                if key in feat:
                    feat_news_cnt = self.safe_float(feat.get(key))
                    break

            feat_boost = (
                self.clip01(np.log1p(feat_news_cnt) / np.log1p(100)) * 0.10
                if feat_news_cnt is not None
                else 0.0
            )

            novelty_val = novelty if novelty is not None else 0.5

            raw_score = (
                0.55 * (article_quality if article_quality is not None else 0.15)
                + 0.25 * volume_score
                + 0.10 * novelty_val
                + 0.10 * self.clip01(high_ratio)
                + 0.05 * feat_boost
            )

            raw_score = self.clip01(raw_score)
            news_signal_score = float(raw_score * 100)

            confidence = "medium"

            # 기사 수가 너무 적으면 점수 상한 적용
            if n30_unique < 3:
                news_signal_score = min(news_signal_score, 44.9)
                confidence = "low"
            elif n30_unique < 5:
                news_signal_score = min(news_signal_score, 59.9)
                confidence = "medium"

            verdict = "low_signal"
            if news_signal_score >= self.thresh_high:
                verdict = "high_signal"
            elif news_signal_score >= self.thresh_med:
                verdict = "medium_signal"

            if len(df30_u):
                df30_u["__ni_rec_bonus"] = 0.0
                if df30_u["__ni_dt"].notna().any():
                    d = (as_of_dt - df30_u["__ni_dt"]).dt.days.clip(lower=0, upper=7)
                    df30_u["__ni_rec_bonus"] = (7 - d) / 7 * 0.05

                df30_u["__ni_article_score"] = df30_u["__ni_rule"] + df30_u["__ni_rec_bonus"]
                top_df = df30_u.sort_values(["__ni_article_score", "__ni_dt"], ascending=[False, False]).head(3)
            else:
                top_df = df30_u.head(0)

            top_articles = []
            for _, rr in top_df.iterrows():
                url = rr[url_col] if url_col else None

                press = None
                if press_col:
                    press = rr.get(press_col, None)
                    if isinstance(press, float) and np.isnan(press):
                        press = None

                if not press:
                    press = self.press_from_url(url)

                top_articles.append({
                    "date": str(rr["__ni_dt"].date()) if pd.notna(rr["__ni_dt"]) else None,
                    "title": rr["__ni_title"][:200],
                    "description": rr["__ni_desc"][:240],
                    "url": url,
                    "press": press,
                    "article_score": float(rr["__ni_article_score"]),
                    "rule_score_0_1": float(rr["__ni_rule"]),
                })

            reasons = []
            if high_ratio >= 0.30:
                reasons.append("고임팩트 이벤트성 뉴스(실적/공시/계약/소송 등) 비중이 높음")
            if noise_ratio >= 0.50:
                reasons.append("주가설명/테마성 기사 비중이 높아 노이즈 가능성")
            if n7_unique >= 3:
                reasons.append("최근 7일 유니크 기사량 증가")

            results.append({
                "ticker": tk,
                "as_of_date": as_of_date,
                "news_summary": {
                    "n_articles_7d_unique": int(n7_unique),
                    "n_articles_30d_unique": int(n30_unique),
                    "n_articles_30d_raw": int(n30_raw),
                    "article_quality_score_0_1": article_quality,
                    "novelty_ratio_unique_over_raw_0_1": novelty,
                    "noise_ratio_0_1": noise_ratio,
                    "high_impact_ratio_0_1": high_ratio,
                },
                "news_signal_score": news_signal_score,
                "confidence_level": confidence,
                "verdict": verdict,
                "reasons": reasons[:6],
                "top_articles": top_articles,
            })

        out = {
            "meta": {
                "agent": "NewsInvestigationAgent",
                "version": "v1.4_news_only_no_market",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "as_of_date": as_of_date,
                "inputs": {
                    "news_raw": os.path.basename(self.news_raw_path),
                    "news_features_by_stock": os.path.basename(self.news_features_path),
                },
                "thresholds": {
                    "high_signal": self.thresh_high,
                    "medium_signal": self.thresh_med,
                },
                "score_formula": {
                    "article_quality": 0.55,
                    "recent_volume": 0.25,
                    "novelty": 0.10,
                    "high_impact_ratio": 0.10,
                    "feature_news_count_boost": 0.05,
                    "market_context": 0.0,
                },
            },
            "universe_summary": {},
            "tickers": sorted(results, key=lambda x: x["news_signal_score"], reverse=True),
        }

        verdicts = [x["verdict"] for x in out["tickers"]]

        out["universe_summary"] = {
            "n_tickers_with_news": int(len(out["tickers"])),
            "avg_news_signal_score": float(np.mean([x["news_signal_score"] for x in out["tickers"]])) if out["tickers"] else None,
            "count_high_signal": int(sum(v == "high_signal" for v in verdicts)),
            "count_medium_signal": int(sum(v == "medium_signal" for v in verdicts)),
            "count_low_signal": int(sum(v == "low_signal" for v in verdicts)),
            "share_high_signal": float(np.mean([1 if v == "high_signal" else 0 for v in verdicts])) if verdicts else 0.0,
        }

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return {
            "news_invest_rag_result": self.output_path,
            "news_invest_rag_result_json": self.output_path,
            "result": out,
        }