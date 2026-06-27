from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple


class LocalNewsCorpusRAG:
    """
    수집 뉴스 corpus 기반 RAG.
    
    1. Local news corpus loading
    2. Ticker-level query generation
    3. Candidate retrieval
    4. Evidence verification / reranking
    5. Verified evidence attachment
    """

    def __init__(
        self,
        corpus_dir: str,
        output_json_path: Optional[str] = None,
        fallback_news_raw_path: Optional[str] = None,
        top_k: int = 3,
        candidate_k: Optional[int] = None,
        max_rows_per_file: int = 500,
        max_docs: int = 30000,
        min_score: float = 0.01,
        min_directness_score: float = 0.45,
        allow_indirect_fallback: bool = False,
    ):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.output_json_path = Path(output_json_path) if output_json_path else None
        self.fallback_news_raw_path = Path(fallback_news_raw_path) if fallback_news_raw_path else None

        self.top_k = int(top_k or 3)
        self.candidate_k = int(candidate_k or max(self.top_k * 5, 15))

        self.max_rows_per_file = int(max_rows_per_file or 500)
        self.max_docs = int(max_docs or 30000)

        self.min_score = float(min_score or 0.01)
        self.min_directness_score = float(min_directness_score or 0.45)

        # False 권장:
        # verified evidence가 없으면 억지로 indirect mention을 넣지 않음.
        self.allow_indirect_fallback = bool(allow_indirect_fallback)

    # =========================================================
    # Public API
    # =========================================================

    def enrich_news_investigation(self, news_investigation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        NewsInvestigationAgent 결과에 RAG evidence를 추가한다.
        """

        docs = self.load_corpus()
        queries = self.build_queries(news_investigation_result)

        retrieval_result = self.retrieve(
            docs=docs,
            queries=queries,
        )

        retrieved_evidence = retrieval_result["retrieved_evidence"]
        candidate_evidence = retrieval_result["candidate_evidence"]
        rejected_evidence = retrieval_result["rejected_evidence"]

        rag_sources = self.build_sources(retrieved_evidence)

        verification_result = self.build_verification_summary(
            docs=docs,
            queries=queries,
            retrieved_evidence=retrieved_evidence,
            candidate_evidence=candidate_evidence,
            rejected_evidence=rejected_evidence,
        )

        rag_payload = {
            "enabled": True,
            "method": "local_news_corpus_tfidf_retrieval_with_evidence_verification",
            "corpus_dir": str(self.corpus_dir) if self.corpus_dir else None,
            "fallback_news_raw_path": str(self.fallback_news_raw_path) if self.fallback_news_raw_path else None,
            "corpus_size": len(docs),
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "min_score": self.min_score,
            "min_directness_score": self.min_directness_score,
            "allow_indirect_fallback": self.allow_indirect_fallback,
            "retrieval_queries": queries,
            "retrieved_evidence": retrieved_evidence,
            "candidate_evidence": candidate_evidence,
            "rejected_evidence": rejected_evidence,
            "rag_sources": rag_sources,
            "verification_result": verification_result,
        }

        enriched = self.attach_to_result(
            news_investigation_result=news_investigation_result,
            rag_payload=rag_payload,
        )

        if self.output_json_path:
            self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_json_path, "w", encoding="utf-8") as f:
                json.dump(rag_payload, f, ensure_ascii=False, indent=2)

        return enriched

    # =========================================================
    # Corpus Loading
    # =========================================================

    def load_corpus(self) -> List[Dict[str, Any]]:
        """
        RAG 검색 대상 문서 corpus를 읽는다.

        우선순위:
        1. corpus_dir/*.csv
        2. fallback_news_raw_path
        """

        files: List[Path] = []

        if self.corpus_dir and self.corpus_dir.exists():
            files.extend(sorted(self.corpus_dir.glob("*.csv")))

        if not files and self.fallback_news_raw_path and self.fallback_news_raw_path.exists():
            files.append(self.fallback_news_raw_path)

        docs: List[Dict[str, Any]] = []
        seen = set()

        for file_path in files:
            if len(docs) >= self.max_docs:
                break

            file_ticker, file_company = self.infer_ticker_company_from_file(file_path)
            rows = self.read_csv_dicts(file_path, max_rows=self.max_rows_per_file)

            for idx, row in enumerate(rows):
                if len(docs) >= self.max_docs:
                    break

                title = self.first_non_empty(
                    row,
                    ["title", "제목", "news_title", "headline", "기사제목"],
                )

                desc = self.first_non_empty(
                    row,
                    [
                        "description",
                        "desc",
                        "summary",
                        "요약",
                        "content",
                        "본문",
                        "본문요약",
                        "news_summary",
                        "news_content",
                        "article",
                        "text",
                    ],
                )

                date = self.first_non_empty(
                    row,
                    [
                        "date",
                        "날짜",
                        "pubDate",
                        "published_at",
                        "publishedAt",
                        "datetime",
                        "created_at",
                        "기사일자",
                    ],
                )

                url = self.first_non_empty(
                    row,
                    ["link", "url", "originallink", "original_link", "news_url", "article_url"],
                )

                press = self.first_non_empty(
                    row,
                    ["press", "media", "source", "publisher", "언론사"],
                )

                if not press:
                    press = self.press_from_url(url)

                ticker = self.first_non_empty(
                    row,
                    ["ticker", "종목코드", "stock_code", "code", "symbol"],
                ) or file_ticker or ""

                company = self.first_non_empty(
                    row,
                    ["company", "company_name", "기업명", "종목명", "name"],
                ) or file_company or ""

                ticker = self.zfill6(ticker) if ticker else ""

                # 검색용 text에는 company/ticker를 앞에 붙인다.
                # 단, evidence directness 판단에서는 raw title/description만 사용한다.
                search_text = " ".join(
                    [
                        str(company),
                        str(ticker),
                        str(title),
                        str(desc),
                        str(press),
                        str(date),
                    ]
                ).strip()

                if not search_text:
                    continue

                dedup_key = (
                    self.normalize_for_dedupe(title),
                    str(url).strip(),
                    self.normalize_for_dedupe(search_text[:200]),
                )

                if dedup_key in seen:
                    continue

                seen.add(dedup_key)

                docs.append(
                    {
                        "doc_id": f"{file_path.stem}_{idx}",
                        "ticker": ticker,
                        "company": company,
                        "title": title,
                        "description": desc,
                        "text": search_text,
                        "date": date,
                        "press": press,
                        "url": url,
                        "source_path": str(file_path),
                    }
                )

        return docs

    @staticmethod
    def read_csv_dicts(path: Path, max_rows: int = 500) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        encodings = ["utf-8-sig", "utf-8", "cp949"]

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.DictReader(f)

                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        rows.append(dict(row))

                return rows

            except UnicodeDecodeError:
                continue
            except Exception:
                return rows

        return rows

    @staticmethod
    def infer_ticker_company_from_file(path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        예:
        000660_SK하이닉스_naver_news.csv
        -> ticker=000660, company=SK하이닉스
        """

        stem = path.stem
        parts = stem.split("_")

        if len(parts) >= 2 and re.fullmatch(r"\d{6}", parts[0]):
            return parts[0], parts[1]

        return None, None

    # =========================================================
    # Query Building
    # =========================================================

    def build_queries(self, news_investigation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        NewsInvestigationAgent 결과를 바탕으로 ticker별 RAG query 생성.
        """

        queries: List[Dict[str, Any]] = []
        seen = set()

        tickers = news_investigation_result.get("tickers", [])

        if not isinstance(tickers, list):
            tickers = []

        for item in tickers:
            if not isinstance(item, dict):
                continue

            ticker = str(item.get("ticker") or "").strip()
            ticker = self.zfill6(ticker) if ticker else ""

            company = str(
                item.get("company")
                or item.get("company_name")
                or item.get("name")
                or ""
            ).strip()

            reasons = item.get("reasons") or []

            if not isinstance(reasons, list):
                reasons = [str(reasons)]

            event_keywords = self.extract_event_keywords_from_reasons(reasons)

            top_articles = item.get("top_articles") or []
            title_texts = []

            if isinstance(top_articles, list):
                for article in top_articles[:3]:
                    if not isinstance(article, dict):
                        continue

                    title = self.compact_space(article.get("title") or "")

                    if title:
                        title_texts.append(title)

            title_text = " ".join(title_texts)

            query = " ".join(
                [
                    ticker,
                    company,
                    title_text,
                    " ".join(event_keywords),
                    "뉴스",
                ]
            ).strip()

            query = self.compact_space(query)
            query = query[:500]

            if not query:
                continue

            key = query.lower()

            if key in seen:
                continue

            seen.add(key)

            queries.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "query_type": "ticker_news_evidence_retrieval",
                    "is_fallback": False,
                    "query": query,
                }
            )

        if not queries:
            queries.append(
                {
                    "ticker": "",
                    "company": "",
                    "query_type": "fallback_market_news_retrieval",
                    "is_fallback": True,
                    "query": "시장 실적 공시 수주 투자 계약 규제 리스크 뉴스",
                }
            )

        return queries

    @staticmethod
    def extract_event_keywords_from_reasons(reasons: List[Any]) -> List[str]:
        reason_text = " ".join(str(r) for r in reasons)

        event_keywords: List[str] = []

        if "고임팩트" in reason_text or "실적" in reason_text or "공시" in reason_text:
            event_keywords.extend(["실적", "공시", "수주", "투자", "계약", "규제", "리스크"])

        if "노이즈" in reason_text or "테마성" in reason_text:
            event_keywords.extend(["노이즈", "테마", "주가설명"])

        if "최근 7일" in reason_text or "기사량" in reason_text:
            event_keywords.extend(["최근뉴스", "뉴스증가"])

        if not event_keywords:
            event_keywords.extend(["실적", "공시", "수주", "투자", "계약", "리스크"])

        return list(dict.fromkeys(event_keywords))

    # =========================================================
    # Retrieval
    # =========================================================

    def retrieve(
        self,
        docs: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:

        if not docs or not queries:
            return {
                "retrieved_evidence": [],
                "candidate_evidence": [],
                "rejected_evidence": [],
            }

        index = self.build_tfidf_index(docs)
        doc_tokens = index["doc_tokens"]
        idf = index["idf"]

        retrieved_evidence: List[Dict[str, Any]] = []
        candidate_evidence: List[Dict[str, Any]] = []
        rejected_evidence: List[Dict[str, Any]] = []

        seen_final = set()
        seen_candidate = set()
        seen_rejected = set()

        for query_obj in queries:
            query = query_obj.get("query", "")
            query_ticker = str(query_obj.get("ticker") or "").strip()
            query_ticker = self.zfill6(query_ticker) if query_ticker else ""

            query_tokens = self.tokenize(query)

            scored_candidates = []

            for i, doc in enumerate(docs):
                doc_ticker = str(doc.get("ticker") or "").strip()
                doc_ticker = self.zfill6(doc_ticker) if doc_ticker else ""

                # 종목별 query에서는 같은 ticker 문서만 retrieval 대상으로 사용
                if query_ticker:
                    if not doc_ticker or query_ticker != doc_ticker:
                        continue

                retrieval_score = self.score_doc(
                    query_tokens=query_tokens,
                    doc_tf=doc_tokens[i],
                    idf=idf,
                )

                if query_ticker and doc_ticker == query_ticker:
                    retrieval_score = retrieval_score * 1.35 + 0.05

                if retrieval_score < self.min_score:
                    continue

                scored_candidates.append((retrieval_score, doc))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            verified_for_query: List[Dict[str, Any]] = []
            rejected_for_query: List[Dict[str, Any]] = []

            for candidate_rank, (retrieval_score, doc) in enumerate(
                scored_candidates[: self.candidate_k],
                start=1,
            ):
                verification = self.verify_evidence_candidate(
                    query_obj=query_obj,
                    doc=doc,
                    retrieval_score=retrieval_score,
                )

                evidence_item = self.build_evidence_item(
                    query_obj=query_obj,
                    doc=doc,
                    retrieval_score=retrieval_score,
                    candidate_rank=candidate_rank,
                    verification=verification,
                )

                candidate_key = (
                    query_obj.get("query_type"),
                    query_obj.get("ticker"),
                    doc.get("doc_id"),
                )

                if candidate_key not in seen_candidate:
                    seen_candidate.add(candidate_key)
                    candidate_evidence.append(evidence_item)

                if evidence_item["keep_as_evidence"]:
                    verified_for_query.append(evidence_item)
                else:
                    rejected_for_query.append(evidence_item)

            # verified evidence 우선 정렬
            verified_for_query.sort(
                key=lambda x: (
                    x.get("final_score", 0.0),
                    x.get("directness_score", 0.0),
                    x.get("retrieval_score", 0.0),
                ),
                reverse=True,
            )

            selected = verified_for_query[: self.top_k]

            # 선택지가 하나도 없고 indirect fallback을 허용하는 경우만 indirect를 채움
            if not selected and self.allow_indirect_fallback:
                indirect_candidates = [
                    x for x in rejected_for_query
                    if x.get("evidence_relevance") == "indirect_mention"
                ]

                indirect_candidates.sort(
                    key=lambda x: (
                        x.get("final_score", 0.0),
                        x.get("directness_score", 0.0),
                        x.get("retrieval_score", 0.0),
                    ),
                    reverse=True,
                )

                selected = indirect_candidates[: self.top_k]

                for item in selected:
                    item["keep_as_evidence"] = True
                    item["kept_by_indirect_fallback"] = True

            for rank, item in enumerate(selected, start=1):
                final_key = (
                    item.get("query_type"),
                    item.get("query_ticker"),
                    item.get("doc_id"),
                )

                if final_key in seen_final:
                    continue

                seen_final.add(final_key)
                item["rank"] = rank
                retrieved_evidence.append(item)

            selected_doc_ids = {x.get("doc_id") for x in selected}

            for item in rejected_for_query:
                if item.get("doc_id") in selected_doc_ids:
                    continue

                rejected_key = (
                    item.get("query_type"),
                    item.get("query_ticker"),
                    item.get("doc_id"),
                )

                if rejected_key in seen_rejected:
                    continue

                seen_rejected.add(rejected_key)
                rejected_evidence.append(item)

        return {
            "retrieved_evidence": retrieved_evidence,
            "candidate_evidence": candidate_evidence,
            "rejected_evidence": rejected_evidence,
        }

    def build_tfidf_index(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        doc_tokens = []
        df = Counter()

        for doc in docs:
            tokens = self.tokenize(doc.get("text", ""))
            token_counts = Counter(tokens)

            doc_tokens.append(token_counts)

            for token in token_counts.keys():
                df[token] += 1

        n_docs = max(len(docs), 1)

        idf = {}
        for token, freq in df.items():
            idf[token] = math.log((1 + n_docs) / (1 + freq)) + 1

        return {
            "doc_tokens": doc_tokens,
            "idf": idf,
            "n_docs": n_docs,
        }

    @staticmethod
    def score_doc(query_tokens: List[str], doc_tf: Counter, idf: Dict[str, float]) -> float:
        if not query_tokens or not doc_tf:
            return 0.0

        query_tf = Counter(query_tokens)
        score = 0.0

        for token, q_count in query_tf.items():
            if token not in doc_tf:
                continue

            score += q_count * doc_tf[token] * idf.get(token, 1.0)

        norm = math.sqrt(sum(v * v for v in doc_tf.values())) or 1.0

        return score / norm

    # =========================================================
    # Evidence Verification / Reranking
    # =========================================================

    def verify_evidence_candidate(
        self,
        query_obj: Dict[str, Any],
        doc: Dict[str, Any],
        retrieval_score: float,
    ) -> Dict[str, Any]:
        """
        후보 evidence가 실제로 해당 기업/종목의 직접 근거인지 검증한다.

        핵심 판단:
        - 회사명이 title에 직접 등장하는가
        - 회사명이 description에 직접 등장하는가
        - 실적/공시/수주/투자/계약/규제/사고 등 이벤트가 있는가
        - 단순 시장 수급/브랜드평판/특징주/리포트성 기사인가
        """

        query_ticker = self.zfill6(query_obj.get("ticker") or "")
        query_company = self.compact_space(query_obj.get("company") or "")

        doc_ticker = self.zfill6(doc.get("ticker") or "")
        doc_company = self.compact_space(doc.get("company") or "")

        target_company = query_company or doc_company

        title = self.compact_space(doc.get("title") or "")
        desc = self.compact_space(doc.get("description") or "")
        raw_text = f"{title} {desc}"

        title_norm = self.normalize_for_match(title)
        desc_norm = self.normalize_for_match(desc)
        raw_norm = self.normalize_for_match(raw_text)

        company_aliases = self.build_company_aliases(target_company)

        title_has_company = any(alias and alias in title_norm for alias in company_aliases)
        desc_has_company = any(alias and alias in desc_norm for alias in company_aliases)
        raw_has_company = any(alias and alias in raw_norm for alias in company_aliases)

        title_has_ticker = bool(query_ticker and query_ticker in title)
        desc_has_ticker = bool(query_ticker and query_ticker in desc)

        event_hits = self.find_event_hits(raw_text)
        title_event_hits = self.find_event_hits(title)

        weak_context_hits = self.find_weak_context_hits(raw_text)
        list_article_hits = self.find_list_article_hits(raw_text)
        analyst_context_hits = self.find_analyst_context_hits(raw_text)

        directness_score = 0.0
        reasons: List[str] = []

        # -----------------------------------------------------
        # Positive signals
        # -----------------------------------------------------
        if title_has_company:
            directness_score += 0.45
            reasons.append("target company appears in title")

        if title_has_ticker:
            directness_score += 0.40
            reasons.append("target ticker appears in title")

        if desc_has_company:
            directness_score += 0.25
            reasons.append("target company appears in description")

        if desc_has_ticker:
            directness_score += 0.20
            reasons.append("target ticker appears in description")

        if title_event_hits:
            directness_score += 0.20
            reasons.append(f"event keywords appear in title: {title_event_hits[:5]}")

        if event_hits:
            directness_score += 0.10
            reasons.append(f"event keywords appear in article: {event_hits[:5]}")

        company_mention_count = self.count_company_mentions(raw_norm, company_aliases)

        if company_mention_count >= 2:
            directness_score += 0.10
            reasons.append("target company appears multiple times")

        # -----------------------------------------------------
        # Negative signals
        # -----------------------------------------------------
        if not raw_has_company and not title_has_ticker and not desc_has_ticker:
            directness_score -= 0.45
            reasons.append("target company/ticker not found in raw title or description")

        if list_article_hits:
            directness_score -= 0.25
            reasons.append(f"list/market roundup article pattern: {list_article_hits[:5]}")

        if weak_context_hits:
            directness_score -= 0.20
            reasons.append(f"weak market context pattern: {weak_context_hits[:5]}")

        if analyst_context_hits and not title_has_company:
            directness_score -= 0.15
            reasons.append(f"analyst/report context pattern: {analyst_context_hits[:5]}")

        # 회사가 description에만 한 번 나오고 제목 주체가 다른 경우는 indirect 가능성이 큼
        if desc_has_company and not title_has_company and not title_has_ticker:
            if list_article_hits or analyst_context_hits or weak_context_hits:
                directness_score -= 0.15
                reasons.append("company appears only as indirect mention")

        directness_score = self.clip01(directness_score)

        evidence_relevance = self.classify_evidence_relevance(
            directness_score=directness_score,
            title_has_company=title_has_company,
            title_has_ticker=title_has_ticker,
            desc_has_company=desc_has_company,
            event_hits=event_hits,
            weak_context_hits=weak_context_hits,
            list_article_hits=list_article_hits,
            analyst_context_hits=analyst_context_hits,
        )

        keep_as_evidence = directness_score >= self.min_directness_score

        if evidence_relevance in ["market_context", "low_relevance"]:
            keep_as_evidence = False

        if evidence_relevance == "indirect_mention" and not self.allow_indirect_fallback:
            keep_as_evidence = False

        # retrieval_score가 크더라도 directness가 낮으면 final_score는 낮아짐
        final_score = float(retrieval_score) * (0.45 + 0.55 * directness_score)

        return {
            "target_company": target_company,
            "query_ticker": query_ticker,
            "doc_ticker": doc_ticker,
            "title_has_company": title_has_company,
            "desc_has_company": desc_has_company,
            "title_has_ticker": title_has_ticker,
            "desc_has_ticker": desc_has_ticker,
            "company_mention_count": company_mention_count,
            "event_hits": event_hits,
            "title_event_hits": title_event_hits,
            "weak_context_hits": weak_context_hits,
            "list_article_hits": list_article_hits,
            "analyst_context_hits": analyst_context_hits,
            "directness_score": round(float(directness_score), 6),
            "final_score": round(float(final_score), 6),
            "evidence_relevance": evidence_relevance,
            "keep_as_evidence": keep_as_evidence,
            "verification_reasons": reasons,
        }

    @staticmethod
    def classify_evidence_relevance(
        directness_score: float,
        title_has_company: bool,
        title_has_ticker: bool,
        desc_has_company: bool,
        event_hits: List[str],
        weak_context_hits: List[str],
        list_article_hits: List[str],
        analyst_context_hits: List[str],
    ) -> str:
        if directness_score >= 0.75:
            return "direct_company_news"

        if directness_score >= 0.45:
            if title_has_company or title_has_ticker:
                return "direct_company_news"

            if desc_has_company and event_hits:
                return "direct_business_relation"

            return "direct_business_relation"

        if directness_score >= 0.25:
            return "indirect_mention"

        if weak_context_hits or list_article_hits or analyst_context_hits:
            return "market_context"

        return "low_relevance"

    def build_evidence_item(
        self,
        query_obj: Dict[str, Any],
        doc: Dict[str, Any],
        retrieval_score: float,
        candidate_rank: int,
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = str(doc.get("text") or "")
        snippet = text[:320] + ("..." if len(text) > 320 else "")

        press = doc.get("press") or self.press_from_url(doc.get("url"))

        return {
            "query": query_obj.get("query"),
            "query_type": query_obj.get("query_type"),
            "query_ticker": self.zfill6(query_obj.get("ticker") or ""),
            "query_company": query_obj.get("company") or "",
            "candidate_rank": candidate_rank,

            "retrieval_score": round(float(retrieval_score), 6),
            "directness_score": verification.get("directness_score"),
            "final_score": verification.get("final_score"),
            "evidence_relevance": verification.get("evidence_relevance"),
            "keep_as_evidence": verification.get("keep_as_evidence"),
            "verification_reasons": verification.get("verification_reasons"),

            "target_company": verification.get("target_company"),
            "title_has_company": verification.get("title_has_company"),
            "desc_has_company": verification.get("desc_has_company"),
            "title_has_ticker": verification.get("title_has_ticker"),
            "desc_has_ticker": verification.get("desc_has_ticker"),
            "company_mention_count": verification.get("company_mention_count"),
            "event_hits": verification.get("event_hits"),
            "weak_context_hits": verification.get("weak_context_hits"),
            "list_article_hits": verification.get("list_article_hits"),
            "analyst_context_hits": verification.get("analyst_context_hits"),

            "ticker": doc.get("ticker"),
            "company": doc.get("company"),
            "title": doc.get("title"),
            "description": doc.get("description"),
            "snippet": snippet,
            "date": doc.get("date"),
            "press": press,
            "url": doc.get("url"),
            "source_path": doc.get("source_path"),
            "doc_id": doc.get("doc_id"),
        }

    # =========================================================
    # Attach RAG
    # =========================================================

    def attach_to_result(
        self,
        news_investigation_result: Dict[str, Any],
        rag_payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        retrieved_evidence = rag_payload.get("retrieved_evidence") or []
        candidate_evidence = rag_payload.get("candidate_evidence") or []
        rejected_evidence = rag_payload.get("rejected_evidence") or []

        evidence_by_ticker = defaultdict(list)
        candidate_by_ticker = defaultdict(list)
        rejected_by_ticker = defaultdict(list)

        for ev in retrieved_evidence:
            ticker = str(ev.get("query_ticker") or ev.get("ticker") or "").strip()
            ticker = self.zfill6(ticker) if ticker else ""

            if ticker:
                evidence_by_ticker[ticker].append(ev)

        for ev in candidate_evidence:
            ticker = str(ev.get("query_ticker") or ev.get("ticker") or "").strip()
            ticker = self.zfill6(ticker) if ticker else ""

            if ticker:
                candidate_by_ticker[ticker].append(ev)

        for ev in rejected_evidence:
            ticker = str(ev.get("query_ticker") or ev.get("ticker") or "").strip()
            ticker = self.zfill6(ticker) if ticker else ""

            if ticker:
                rejected_by_ticker[ticker].append(ev)

        tickers = news_investigation_result.get("tickers", [])

        if isinstance(tickers, list):
            for item in tickers:
                if not isinstance(item, dict):
                    continue

                ticker = str(item.get("ticker") or "").strip()
                ticker = self.zfill6(ticker) if ticker else ""

                ticker_evidence = evidence_by_ticker.get(ticker, [])
                ticker_candidates = candidate_by_ticker.get(ticker, [])
                ticker_rejected = rejected_by_ticker.get(ticker, [])

                relevance_counts = Counter(
                    ev.get("evidence_relevance") for ev in ticker_candidates
                )

                item["rag"] = {
                    "enabled": True,
                    "method": rag_payload.get("method"),
                    "evidence_count": len(ticker_evidence),
                    "candidate_count": len(ticker_candidates),
                    "rejected_count": len(ticker_rejected),
                    "relevance_counts": dict(relevance_counts),
                    "retrieved_evidence": ticker_evidence,
                    "verification_status": "supported" if ticker_evidence else "no_verified_evidence",
                }

        meta = news_investigation_result.setdefault("meta", {})
        meta["rag_enabled"] = True
        meta["rag_method"] = rag_payload.get("method")
        meta["rag_corpus_size"] = rag_payload.get("corpus_size")
        meta["rag_evidence_count"] = len(retrieved_evidence)
        meta["rag_candidate_count"] = len(candidate_evidence)
        meta["rag_rejected_count"] = len(rejected_evidence)
        meta["rag_generated_at_utc"] = datetime.now(timezone.utc).isoformat()

        news_investigation_result["rag"] = rag_payload
        news_investigation_result["retrieval_queries"] = rag_payload.get("retrieval_queries", [])
        news_investigation_result["retrieved_evidence"] = retrieved_evidence
        news_investigation_result["rag_sources"] = rag_payload.get("rag_sources", [])
        news_investigation_result["verification_result"] = rag_payload.get("verification_result", {})
        news_investigation_result["rag_enabled"] = True

        return news_investigation_result

    @staticmethod
    def build_sources(retrieved_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen = set()

        for ev in retrieved_evidence:
            key = (
                ev.get("title"),
                ev.get("url"),
                ev.get("source_path"),
                ev.get("doc_id"),
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "title": ev.get("title"),
                    "date": ev.get("date"),
                    "press": ev.get("press"),
                    "url": ev.get("url"),
                    "source_path": ev.get("source_path"),
                    "doc_id": ev.get("doc_id"),
                    "evidence_relevance": ev.get("evidence_relevance"),
                    "directness_score": ev.get("directness_score"),
                }
            )

        return sources

    @staticmethod
    def build_verification_summary(
        docs: List[Dict[str, Any]],
        queries: List[Dict[str, Any]],
        retrieved_evidence: List[Dict[str, Any]],
        candidate_evidence: List[Dict[str, Any]],
        rejected_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        relevance_counts = Counter(ev.get("evidence_relevance") for ev in candidate_evidence)
        kept_relevance_counts = Counter(ev.get("evidence_relevance") for ev in retrieved_evidence)

        fallback_query_count = sum(1 for q in queries if q.get("is_fallback"))

        query_count = len(queries)
        supported_query_count = len(set(ev.get("query_ticker") for ev in retrieved_evidence if ev.get("query_ticker")))

        return {
            "status": "completed" if candidate_evidence else "no_candidate_found",
            "method": "candidate_retrieval_then_directness_verification",
            "corpus_size": len(docs),
            "query_count": query_count,
            "fallback_query_count": fallback_query_count,
            "candidate_count": len(candidate_evidence),
            "verified_evidence_count": len(retrieved_evidence),
            "rejected_evidence_count": len(rejected_evidence),
            "supported_query_count": supported_query_count,
            "unsupported_query_count": max(query_count - supported_query_count, 0),
            "candidate_relevance_counts": dict(relevance_counts),
            "kept_relevance_counts": dict(kept_relevance_counts),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "The RAG module first retrieves candidate articles from the local news corpus, "
                "then verifies whether each article is direct company evidence, direct business relation, "
                "indirect mention, or weak market context. Only verified evidence is attached as retrieved_evidence."
            ),
        }

    # =========================================================
    # Verification patterns
    # =========================================================

    @staticmethod
    def find_event_hits(text: Any) -> List[str]:
        text = str(text or "")

        event_terms = [
            "실적",
            "잠정",
            "영업이익",
            "순이익",
            "매출",
            "흑자전환",
            "적자",
            "공시",
            "수주",
            "계약",
            "공급계약",
            "양수도",
            "인수",
            "매각",
            "합병",
            "M&A",
            "유상증자",
            "무상증자",
            "증자",
            "감자",
            "투자",
            "출자",
            "시설자금",
            "자사주",
            "배당",
            "분할",
            "상장",
            "상장예비심사",
            "상장예심",
            "거래정지",
            "제재",
            "규제",
            "소송",
            "피소",
            "리콜",
            "사고",
            "사망",
            "중대재해",
            "승인",
            "허가",
            "임상",
            "기술이전",
            "라이선스",
            "국책과제",
            "프로젝트",
            "생산",
            "증설",
        ]

        return [term for term in event_terms if term in text]

    @staticmethod
    def find_weak_context_hits(text: Any) -> List[str]:
        text = str(text or "")

        weak_terms = [
            "주가",
            "급등",
            "급락",
            "강세",
            "약세",
            "상승세",
            "하락세",
            "특징주",
            "관련주",
            "테마주",
            "수혜주",
            "대장주",
            "차익실현",
            "투자심리",
            "브랜드평판",
            "이격도과열",
            "순매수",
            "순매도",
            "외국인",
            "기관",
            "개미",
            "초고수",
            "마켓뷰",
            "시황",
            "증시",
            "코스피",
            "코스닥",
            "지수선물",
            "옵션",
            "서킷브레이커",
            "사이드카",
        ]

        return [term for term in weak_terms if term in text]

    @staticmethod
    def find_list_article_hits(text: Any) -> List[str]:
        text = str(text or "")

        list_patterns = [
            "[거래소 외국인]",
            "[거래소 기관]",
            "[코스닥 외국인]",
            "[코스닥 기관]",
            "[이격도과열 종목]",
            "[시총 100대기업]",
            "[특징주]",
            "[마켓뷰]",
            "[1% 초고수",
            "브랜드평판",
            "순매수 상위",
            "순매도 상위",
            "수익률 상위",
            "상승률 상위",
            "하락률 상위",
        ]

        return [term for term in list_patterns if term in text]

    @staticmethod
    def find_analyst_context_hits(text: Any) -> List[str]:
        text = str(text or "")

        analyst_terms = [
            "목표주가",
            "투자의견",
            "매수 의견",
            "리포트",
            "보고서",
            "증권은",
            "증권 연구원",
            "연구원은",
            "전망했다",
            "분석했다",
            "내다봤다",
            "예상했다",
            "컨센서스",
        ]

        return [term for term in analyst_terms if term in text]

    # =========================================================
    # Text Utilities
    # =========================================================

    @staticmethod
    def tokenize(text: Any) -> List[str]:
        if text is None:
            return []

        text = str(text).lower()
        tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)

        stopwords = {
            "the", "and", "or", "of", "to", "in", "for", "on", "with",
            "및", "그리고", "또는", "관련", "기반", "대한", "통해", "있는",
            "없는", "에서", "으로", "하며", "했다", "한다", "된다", "하는",
            "있다", "이번", "지난", "오늘", "전일", "최근",
        }

        return [t for t in tokens if len(t) >= 2 and t not in stopwords]

    @staticmethod
    def first_non_empty(row: Dict[str, Any], candidates: List[str]) -> str:
        for key in candidates:
            value = row.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    @staticmethod
    def normalize_for_dedupe(text: Any) -> str:
        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"[^0-9a-z가-힣\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def normalize_for_match(text: Any) -> str:
        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"[^0-9a-z가-힣]", "", text)

        return text

    @staticmethod
    def compact_space(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def zfill6(x: Any) -> str:
        s = str(x).strip()

        if s.endswith(".0"):
            s = s[:-2]

        if not s:
            return ""

        return s.zfill(6)

    @staticmethod
    def clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def build_company_aliases(company: Any) -> List[str]:
        company = str(company or "").strip()

        if not company:
            return []

        aliases = set()

        raw = company
        no_space = re.sub(r"\s+", "", raw)
        normalized = re.sub(r"[^0-9a-zA-Z가-힣]", "", no_space)

        aliases.add(raw)
        aliases.add(no_space)
        aliases.add(normalized)

        # 자주 나오는 축약명 보정
        manual_aliases = {
            "한국전력": ["한전", "KEPCO"],
            "삼성전자": ["삼전", "Samsung Electronics"],
            "삼성SDI": ["삼성SDI"],
            "SK하이닉스": ["하이닉스", "SK hynix"],
            "SK텔레콤": ["SKT", "SK텔레콤"],
            "현대건설": ["현대건설"],
            "HD현대중공업": ["HD현대중공업", "HD 현대중공업", "현대중공업"],
            "HD한국조선해양": ["HD한국조선해양", "HD 한국조선해양", "한국조선해양"],
            "HD현대일렉트릭": ["HD현대일렉트릭", "HD 현대일렉트릭", "현대일렉트릭"],
            "한화에어로스페이스": ["한화에어로", "한화 에어로", "한화에어로스페이스"],
            "LIG디펜스앤에어로스페이스": [
                "LIG디펜스앤에어로스페이스",
                "LIG D&A",
                "LIGD&A",
                "LIG디펜스",
                "LIG D",
            ],
            "리가켐바이오": ["리가켐", "리가켐바이오"],
            "하나금융지주": ["하나금융", "하나금융지주"],
            "우리금융지주": ["우리금융", "우리금융지주"],
            "미래에셋증권": ["미래에셋증권", "미래에셋"],
        }

        if raw in manual_aliases:
            for alias in manual_aliases[raw]:
                aliases.add(alias)

        # 긴 회사명의 일부 핵심 토큰도 alias로 추가
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", raw)

        for token in tokens:
            if len(token) >= 3:
                aliases.add(token)

        normalized_aliases = []

        for alias in aliases:
            alias_norm = re.sub(r"[^0-9a-zA-Z가-힣]", "", str(alias).lower())

            if len(alias_norm) >= 2:
                normalized_aliases.append(alias_norm)

        return list(dict.fromkeys(normalized_aliases))

    @staticmethod
    def count_company_mentions(text_norm: str, aliases: List[str]) -> int:
        if not text_norm or not aliases:
            return 0

        count = 0

        for alias in aliases:
            if not alias:
                continue

            count += text_norm.count(alias)

        return count

    @staticmethod
    def press_from_url(url: Any) -> str:
        if url is None:
            return ""

        u = str(url).strip()

        if not u:
            return ""

        try:
            parsed = urlparse(u)
            netloc = parsed.netloc.lower().replace("www.", "")

            if not netloc:
                return ""

            mapping = {
                "news.naver.com": "Naver News",
                "n.news.naver.com": "Naver News",
                "m.news.naver.com": "Naver News",
                "finance.naver.com": "Naver Finance",

                "hankyung.com": "한국경제",
                "mk.co.kr": "매일경제",
                "yna.co.kr": "연합뉴스",
                "yonhapnewstv.co.kr": "연합뉴스TV",
                "edaily.co.kr": "이데일리",
                "sedaily.com": "서울경제",
                "biz.chosun.com": "조선비즈",
                "chosun.com": "조선일보",
                "joongang.co.kr": "중앙일보",
                "donga.com": "동아일보",
                "fnnews.com": "파이낸셜뉴스",
                "news.mt.co.kr": "머니투데이",
                "mt.co.kr": "머니투데이",
                "etnews.com": "전자신문",
                "zdnet.co.kr": "지디넷코리아",
                "bloter.net": "블로터",
                "thebell.co.kr": "더벨",
                "businesspost.co.kr": "비즈니스포스트",
                "newspim.com": "뉴스핌",
                "newsis.com": "뉴시스",
                "ajunews.com": "아주경제",
                "heraldcorp.com": "헤럴드경제",
                "khan.co.kr": "경향신문",
                "hani.co.kr": "한겨레",
                "seoul.co.kr": "서울신문",
                "ytn.co.kr": "YTN",
                "sbs.co.kr": "SBS",
                "kbs.co.kr": "KBS",
                "imbc.com": "MBC",
            }

            if netloc in mapping:
                return mapping[netloc]

            for domain, press in mapping.items():
                if netloc.endswith(domain):
                    return press

            return netloc

        except Exception:
            return ""