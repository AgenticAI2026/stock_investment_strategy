from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


@dataclass
class EvidenceBuilderConfig:
    top_k: int = 10
    chart_window: int = 30
    output_dir_name: str = "report_evidence"
    output_json_name: str = "report_evidence_result.json"
    manifest_name: str = "manifest.json"

    # RAG evidence handling
    rag_top_k_per_candidate: int = 3
    rag_allowed_relevance: Tuple[str, ...] = (
        "direct_company_news",
        "direct_business_relation",
    )


@dataclass
class ReportEvidenceBuilderArtifacts:
    run_dir: Path
    output_dir: Path
    report_evidence_result_path: Path
    manifest_path: Path
    output_files: List[str]


class ReportEvidenceBuilderAgent(BaseAgent):
    """
    Candidate Scoring Agent가 만든 TOP10 후보를 기반으로
    Report Generative Agent가 사용할 종목별 근거 패키지를 생성한다.

    핵심 원칙:
    - LLM 사용 안 함
    - 입력 데이터에 없는 수치/뉴스/근거 생성 안 함
    - 누락 필드는 null과 data_quality.missing_fields에 기록
    - Ours B에서는 NewsInvestigationAgent의 RAG evidence를
      source_grounded_evidence로 명시적으로 연결
    """

    stage = "report_evidence_builder"
    version = "1.1.0"

    def __init__(
        self,
        config: Optional[EvidenceBuilderConfig] = None,
        encoding: str = "utf-8",
    ):
        self.config = config or EvidenceBuilderConfig()
        self.encoding = encoding

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.config.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = ReportEvidenceBuilderArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            report_evidence_result_path=output_dir / self.config.output_json_name,
            manifest_path=output_dir / self.config.manifest_name,
            output_files=[],
        )

        ctx.logger.info("[ReportEvidenceBuilderAgent] Starting report evidence building.")

        paths = self._resolve_input_paths(run_dir, ctx)

        candidate_result = self._load_json_required(paths["candidate_scoring_result"], ctx)
        market_flow_result = self._load_json_optional(paths.get("market_flow_result"), ctx)
        market_analysis_result = self._load_json_optional(paths.get("market_analysis_result"), ctx)

        # ----------------------------------------------------
        # NewsInvestigation 결과 로드
        # 1. ctx.analysis_outputs.news_investigation 우선
        # 2. news_invest_result.json
        # 3. news_rag_result.json은 RAG payload로 별도 병합
        # ----------------------------------------------------
        analysis_outputs = self._ctx_get(ctx, "analysis_outputs", {}) or {}

        news_result_from_ctx = {}
        if isinstance(analysis_outputs, dict):
            maybe_news = analysis_outputs.get("news_investigation")
            if isinstance(maybe_news, dict):
                news_result_from_ctx = maybe_news

        news_result_file = self._load_json_optional(paths.get("news_invest_result"), ctx)
        news_rag_payload_file = self._load_json_optional(paths.get("news_rag_result"), ctx)

        news_result = self._merge_news_and_rag_payload(
            primary_news_result=news_result_from_ctx or news_result_file,
            rag_payload=news_rag_payload_file,
        )

        risk_result = self._load_json_optional(paths.get("risk_score_result"), ctx)

        ohlcv_df = self._load_csv_optional(paths.get("ohlcv"), ctx)
        finance_df = self._load_csv_optional(paths.get("finance"), ctx)
        foreign_df = self._load_csv_optional(paths.get("foreign"), ctx)
        raw_news_df = self._load_csv_optional(paths.get("news_raw"), ctx)

        market_analysis_map = self._build_ticker_map(
            market_analysis_result.get("ticker_analyses", []) if market_analysis_result else []
        )
        news_map = self._build_ticker_map(news_result.get("tickers", []) if news_result else [])
        risk_map = self._build_ticker_map(risk_result.get("tickers", []) if risk_result else [])
        finance_map = self._build_latest_row_map(finance_df, date_col="year")
        foreign_map = self._build_latest_row_map(foreign_df, date_col="date")

        rag_evidence_map = self._build_rag_evidence_map(news_result)

        top_candidates = candidate_result.get("top10_candidates", [])[: self.config.top_k]
        evidence_items = []

        for candidate in top_candidates:
            ticker = self._normalize_ticker(candidate.get("ticker"))

            market_item = market_analysis_map.get(ticker, {})
            news_item = news_map.get(ticker, {})
            risk_item = risk_map.get(ticker, {})
            finance_item = finance_map.get(ticker, {})
            foreign_item = foreign_map.get(ticker, {})

            rag_evidence = self._select_rag_evidence_for_ticker(
                ticker=ticker,
                rag_evidence_map=rag_evidence_map,
                max_evidence=self.config.rag_top_k_per_candidate,
            )

            source_grounded_evidence = self._build_source_grounded_evidence(
                ticker=ticker,
                candidate=candidate,
                rag_evidence=rag_evidence,
            )

            rag_supported_claims = self._build_rag_supported_claims(
                ticker=ticker,
                candidate=candidate,
                source_grounded_evidence=source_grounded_evidence,
            )

            chart_data = self._build_chart_data(ticker, ohlcv_df, self.config.chart_window)
            latest_price_row = chart_data.get("latest_row", {})

            actual_data = self._build_actual_data(
                candidate=candidate,
                latest_price_row=latest_price_row,
                finance_item=finance_item,
                foreign_item=foreign_item,
                risk_item=risk_item,
            )

            top_news_3 = self._select_top_news(
                ticker=ticker,
                candidate=candidate,
                news_item=news_item,
                raw_news_df=raw_news_df,
                rag_evidence=rag_evidence,
                max_news=3,
            )

            ranking_reason = self._build_ranking_reason(
                candidate=candidate,
                market_item=market_item,
                risk_item=risk_item,
                rag_supported_claims=rag_supported_claims,
            )

            data_quality = self._build_data_quality(
                actual_data=actual_data,
                chart_data=chart_data,
                top_news_3=top_news_3,
                source_grounded_evidence=source_grounded_evidence,
            )

            evidence_items.append(
                self._clean_for_json(
                    {
                        "rank": candidate.get("rank"),
                        "ticker": ticker,
                        "company_name": candidate.get("company_name"),
                        "as_of_date": candidate.get("as_of_date") or candidate_result.get("as_of_date"),
                        "market_flow_candidate_score": candidate.get("market_flow_candidate_score"),
                        "score_before_risk_cap": candidate.get("score_before_risk_cap"),
                        "risk_score_cap_note": candidate.get("risk_score_cap_note"),
                        "component_scores": candidate.get("component_scores", {}),
                        "chart_data": self._strip_internal_chart_fields(chart_data),
                        "actual_data": actual_data,

                        "top_news_3": top_news_3,

                        "source_grounded_evidence": source_grounded_evidence,
                        "rag_supported_claims": rag_supported_claims,
                        "rag_evidence_snapshot": self._build_rag_evidence_snapshot(
                            rag_evidence=rag_evidence,
                            source_grounded_evidence=source_grounded_evidence,
                        ),

                        "ranking_reason": ranking_reason,
                        "positive_evidence": candidate.get("positive_evidence", []),
                        "negative_evidence": candidate.get("negative_evidence", []),
                        "risk_notes": self._merge_unique_strings(
                            candidate.get("risk_notes", []),
                            risk_item.get("evidence", []),
                            market_item.get("risk_factors", []),
                            limit=8,
                        ),
                        "market_analysis_snapshot": self._build_market_analysis_snapshot(market_item),
                        "news_signal_snapshot": self._build_news_signal_snapshot(news_item),
                        "risk_snapshot": self._build_risk_snapshot(risk_item),
                        "data_quality": data_quality,
                    }
                )
            )

        all_source_grounded_evidence = self._flatten_source_grounded_evidence(evidence_items)

        result = self._clean_for_json(
            {
                "meta": {
                    "agent": self.__class__.__name__,
                    "stage": self.stage,
                    "version": self.version,
                    "created_at_utc": self._now_utc(),
                    "purpose": "TOP10 후보별 리포트 근거 구조화 및 RAG 기반 source-grounded evidence 연결",
                },
                "as_of_date": candidate_result.get("as_of_date")
                or self._extract_market_flow(market_flow_result).get("as_of_date"),
                "report_context": candidate_result.get("report_context", {}),
                "market_context": self._build_market_context(
                    candidate_result,
                    market_flow_result,
                    market_analysis_result,
                ),

                # 핵심 output
                "top10_evidence": evidence_items,

                # Ours B에서 ReportGenerativeAgent가 바로 참조할 수 있는 top-level RAG evidence
                "source_grounded_evidence": all_source_grounded_evidence,
                "rag_evidence_summary": self._build_rag_evidence_summary(
                    news_result=news_result,
                    evidence_items=evidence_items,
                    source_grounded_evidence=all_source_grounded_evidence,
                ),

                "glossary_terms": self._build_glossary_terms(),
                "data_limitations": self._build_data_limitations(candidate_result, evidence_items),
                "source_trace": self._build_source_trace(paths),
            }
        )

        self._write_json(artifacts.report_evidence_result_path, result)

        manifest = self._build_manifest(
            result=result,
            paths=paths,
            result_path=artifacts.report_evidence_result_path,
        )
        self._write_json(artifacts.manifest_path, manifest)

        artifacts.output_files = [
            str(artifacts.report_evidence_result_path),
            str(artifacts.manifest_path),
        ]

        rag_summary = result.get("rag_evidence_summary", {}) or {}

        metrics = {
            "n_candidates": len(evidence_items),
            "as_of_date": result.get("as_of_date"),
            "has_candidate_scoring_result": paths["candidate_scoring_result"] is not None,
            "has_ohlcv": paths["ohlcv"] is not None,
            "has_finance": paths["finance"] is not None,
            "has_foreign": paths["foreign"] is not None,
            "has_news_result": paths["news_invest_result"] is not None,
            "has_news_rag_result": paths["news_rag_result"] is not None,
            "has_risk_result": paths["risk_score_result"] is not None,
            "has_market_analysis_result": paths["market_analysis_result"] is not None,
            "n_candidates_with_news_3": sum(
                1
                for item in evidence_items
                if item.get("data_quality", {}).get("news_count", 0) >= 3
            ),
            "n_candidates_with_chart": sum(
                1
                for item in evidence_items
                if item.get("data_quality", {}).get("chart_points", 0) > 0
            ),
            "n_candidates_with_rag_evidence": sum(
                1
                for item in evidence_items
                if item.get("data_quality", {}).get("rag_evidence_count", 0) > 0
            ),
            "n_source_grounded_evidence": len(all_source_grounded_evidence),
            "rag_enabled": rag_summary.get("rag_enabled"),
            "rag_retrieved_evidence_count": rag_summary.get("retrieved_evidence_count"),
            "rag_source_grounded_evidence_count": rag_summary.get("source_grounded_evidence_count"),
        }

        outputs = {
            "output_dir": str(output_dir),
            "report_evidence_result": str(artifacts.report_evidence_result_path),
            "manifest": str(artifacts.manifest_path),
        }

        ctx.logger.info(
            "[ReportEvidenceBuilderAgent] Completed report evidence building. "
            f"source_grounded_evidence={len(all_source_grounded_evidence)}"
        )

        return self._make_stage_result(
            status="success",
            message="Report evidence building completed.",
            metrics=metrics,
            outputs=outputs,
        )

    # ============================================================
    # Input Path Resolution
    # ============================================================

    def _resolve_input_paths(self, run_dir: Path, ctx: RunContext) -> Dict[str, Optional[Path]]:
        paths = {
            "candidate_scoring_result": self._find_single_file(
                run_dir,
                "candidate_scoring_result.json",
                ctx,
            ),
            "market_flow_result": self._find_single_file(
                run_dir,
                "market_flow_result.json",
                ctx,
            ),
            "market_analysis_result": self._find_single_file(
                run_dir,
                "market_analysis_result.json",
                ctx,
            ),
            "news_invest_result": (
                self._find_single_file(run_dir, "news_invest_result.json", ctx)
                or self._find_single_file(run_dir, "news_investigation_result.json", ctx)
            ),
            "news_rag_result": (
                self._find_single_file(run_dir, "news_rag_result.json", ctx)
                or self._find_single_file(run_dir, "news_invest_rag_result.json", ctx)
            ),
            "risk_score_result": self._find_single_file(
                run_dir,
                "risk_score_result.json",
                ctx,
            ),
            "ohlcv": self._find_single_file(
                run_dir,
                "preprocessed__*__price__ohlcv_last365.csv",
                ctx,
            ),
            "finance": self._find_single_file(
                run_dir,
                "preprocessed__*__finance__financial_features.csv",
                ctx,
            ),
            "foreign": self._find_single_file(
                run_dir,
                "preprocessed__*__price__foreign_snapshot_today.csv",
                ctx,
            ),
            "news_raw": self._find_single_file(
                run_dir,
                "preprocessed__*__news__news_raw_merged.csv",
                ctx,
            ),
            "evidence_builder_contract": self._find_single_file(
                run_dir,
                "evidence_builder_contract.json",
                ctx,
            ),
        }

        if paths["candidate_scoring_result"] is None:
            raise FileNotFoundError(
                "필수 입력 파일이 없습니다: candidate_scoring_result.json\n"
                "Candidate Scoring Agent 실행 결과가 먼저 필요합니다."
            )

        return paths

    # ============================================================
    # Loaders
    # ============================================================

    def _load_json_required(self, path: Optional[Path], ctx: RunContext) -> Dict[str, Any]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"필수 JSON 파일이 없습니다: {path}")

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)

    def _load_json_optional(self, path: Optional[Path], ctx: RunContext) -> Dict[str, Any]:
        if path is None:
            return {}

        if not path.exists():
            ctx.logger.warning(f"[ReportEvidenceBuilderAgent] Missing JSON: {path}")
            return {}

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            ctx.logger.warning(
                f"[ReportEvidenceBuilderAgent] Failed to read JSON: {path} ({e})"
            )
            return {}

    def _load_csv_optional(self, path: Optional[Path], ctx: RunContext) -> pd.DataFrame:
        if path is None:
            return pd.DataFrame()

        if not path.exists():
            ctx.logger.warning(f"[ReportEvidenceBuilderAgent] Missing CSV: {path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(path, encoding=self.encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            ctx.logger.warning(
                f"[ReportEvidenceBuilderAgent] Failed to read CSV: {path} ({e})"
            )
            return pd.DataFrame()

        if "종목코드" in df.columns:
            df["ticker"] = df["종목코드"].apply(self._normalize_ticker)
        elif "ticker" in df.columns:
            df["ticker"] = df["ticker"].apply(self._normalize_ticker)
        elif "code" in df.columns:
            df["ticker"] = df["code"].apply(self._normalize_ticker)
        elif "symbol" in df.columns:
            df["ticker"] = df["symbol"].apply(self._normalize_ticker)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "pubDate" in df.columns:
            df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")

        return df

    def _find_single_file(
        self,
        run_dir: Path,
        pattern: str,
        ctx: RunContext,
    ) -> Optional[Path]:
        matches = sorted(run_dir.rglob(pattern))

        if not matches:
            ctx.logger.info(f"[ReportEvidenceBuilderAgent] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            ctx.logger.warning(
                f"[ReportEvidenceBuilderAgent] Multiple files matched: {pattern}. "
                f"Using first: {matches[0]}"
            )

        return matches[0]

    # ============================================================
    # RAG Evidence Handling
    # ============================================================

    def _merge_news_and_rag_payload(
        self,
        primary_news_result: Dict[str, Any],
        rag_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        primary_news_result = primary_news_result or {}
        rag_payload = rag_payload or {}

        if not isinstance(primary_news_result, dict):
            primary_news_result = {}

        if not isinstance(rag_payload, dict):
            rag_payload = {}

        merged = dict(primary_news_result)

        if rag_payload:
            # news_rag_result.json은 보통 RAG payload 자체임
            if "rag" not in merged:
                merged["rag"] = rag_payload
            elif isinstance(merged.get("rag"), dict):
                tmp = dict(rag_payload)
                tmp.update(merged["rag"])
                merged["rag"] = tmp

            for key in [
                "retrieval_queries",
                "retrieved_evidence",
                "rag_sources",
                "verification_result",
            ]:
                if key not in merged and key in rag_payload:
                    merged[key] = rag_payload[key]

            merged["rag_enabled"] = bool(
                merged.get("rag_enabled")
                or rag_payload.get("enabled")
                or rag_payload.get("rag_enabled")
            )

        return merged

    def _build_rag_evidence_map(self, news_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        evidence_by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen = set()

        if not isinstance(news_result, dict):
            return {}

        payloads = [news_result]

        if isinstance(news_result.get("rag"), dict):
            payloads.append(news_result["rag"])

        # 1) top-level RAG evidence
        for payload in payloads:
            for ev in payload.get("retrieved_evidence", []) or []:
                self._add_rag_evidence_to_map(ev, evidence_by_ticker, seen)

        # 2) ticker-level RAG evidence
        for ticker_item in news_result.get("tickers", []) or []:
            if not isinstance(ticker_item, dict):
                continue

            ticker_level_rag = ticker_item.get("rag", {}) or {}
            if not isinstance(ticker_level_rag, dict):
                continue

            for ev in ticker_level_rag.get("retrieved_evidence", []) or []:
                if isinstance(ev, dict) and not ev.get("ticker"):
                    ev = dict(ev)
                    ev["ticker"] = ticker_item.get("ticker")

                self._add_rag_evidence_to_map(ev, evidence_by_ticker, seen)

        # 정렬
        for ticker, items in evidence_by_ticker.items():
            items.sort(
                key=lambda x: (
                    self._safe_float(x.get("final_score")) or 0.0,
                    self._safe_float(x.get("directness_score")) or 0.0,
                    self._safe_float(x.get("retrieval_score")) or self._safe_float(x.get("score")) or 0.0,
                ),
                reverse=True,
            )

        return dict(evidence_by_ticker)

    def _add_rag_evidence_to_map(
        self,
        ev: Any,
        evidence_by_ticker: Dict[str, List[Dict[str, Any]]],
        seen: set,
    ) -> None:
        if not isinstance(ev, dict):
            return

        relevance = ev.get("evidence_relevance")
        if relevance and relevance not in self.config.rag_allowed_relevance:
            return

        if ev.get("keep_as_evidence") is False:
            return

        ticker = self._normalize_ticker(
            ev.get("query_ticker")
            or ev.get("ticker")
        )

        if not ticker:
            return

        key = (
            ticker,
            ev.get("doc_id"),
            ev.get("url"),
            ev.get("title"),
        )

        if key in seen:
            return

        seen.add(key)
        evidence_by_ticker[ticker].append(ev)

    def _select_rag_evidence_for_ticker(
        self,
        ticker: str,
        rag_evidence_map: Dict[str, List[Dict[str, Any]]],
        max_evidence: int = 3,
    ) -> List[Dict[str, Any]]:
        ticker = self._normalize_ticker(ticker)

        if not ticker:
            return []

        items = rag_evidence_map.get(ticker, []) or []

        return items[:max_evidence]

    def _build_source_grounded_evidence(
        self,
        ticker: str,
        candidate: Dict[str, Any],
        rag_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        company_name = (
            candidate.get("company_name")
            or candidate.get("company")
            or candidate.get("name")
        )

        for idx, ev in enumerate(rag_evidence, start=1):
            title = ev.get("title")
            description = ev.get("description") or ev.get("summary_optional") or ev.get("snippet")
            relevance = ev.get("evidence_relevance")

            if not title:
                continue

            claim = self._build_claim_from_rag_evidence(
                company_name=company_name or ev.get("company"),
                title=title,
                relevance=relevance,
                event_hits=ev.get("event_hits", []),
            )

            result.append(
                self._clean_for_json(
                    {
                        "rank": idx,
                        "support_type": "rag_verified_news",
                        "ticker": ticker,
                        "company_name": company_name or ev.get("company"),
                        "claim": claim,

                        "evidence_relevance": relevance,
                        "directness_score": ev.get("directness_score"),
                        "final_score": ev.get("final_score"),
                        "retrieval_score": ev.get("retrieval_score") or ev.get("score"),
                        "verification_reasons": ev.get("verification_reasons", []),
                        "event_hits": ev.get("event_hits", []),

                        "title": title,
                        "summary_optional": description,
                        "press": ev.get("press") or ev.get("source"),
                        "published_at": ev.get("date") or ev.get("published_at") or ev.get("pubDate"),
                        "url": ev.get("url"),
                        "source_path": ev.get("source_path"),
                        "doc_id": ev.get("doc_id"),
                    }
                )
            )

        return result

    def _build_claim_from_rag_evidence(
        self,
        company_name: Any,
        title: str,
        relevance: Optional[str],
        event_hits: Any,
    ) -> str:
        company = str(company_name or "").strip()
        title = str(title or "").strip()

        event_hits_list = []
        if isinstance(event_hits, list):
            event_hits_list = [str(x) for x in event_hits if x]
        elif event_hits:
            event_hits_list = [str(event_hits)]

        event_text = ""
        if event_hits_list:
            event_text = f" 주요 이벤트 키워드는 {', '.join(event_hits_list[:3])}입니다."

        if relevance == "direct_company_news":
            if company:
                return f"{company}와 직접 관련된 뉴스로 '{title}' 기사가 확인되었습니다.{event_text}"
            return f"해당 종목과 직접 관련된 뉴스로 '{title}' 기사가 확인되었습니다.{event_text}"

        if relevance == "direct_business_relation":
            if company:
                return f"{company}은/는 사업 관계 또는 시장 이벤트 맥락에서 언급된 '{title}' 기사가 확인되었습니다.{event_text}"
            return f"해당 종목이 사업 관계 또는 시장 이벤트 맥락에서 언급된 '{title}' 기사가 확인되었습니다.{event_text}"

        if company:
            return f"{company} 관련 뉴스로 '{title}' 기사가 확인되었습니다.{event_text}"

        return f"관련 뉴스로 '{title}' 기사가 확인되었습니다.{event_text}"

    def _build_rag_supported_claims(
        self,
        ticker: str,
        candidate: Dict[str, Any],
        source_grounded_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        claims = []

        for ev in source_grounded_evidence:
            claims.append(
                self._clean_for_json(
                    {
                        "ticker": ticker,
                        "company_name": candidate.get("company_name") or ev.get("company_name"),
                        "claim": ev.get("claim"),
                        "support_type": ev.get("support_type"),
                        "evidence_relevance": ev.get("evidence_relevance"),
                        "source_title": ev.get("title"),
                        "source_press": ev.get("press"),
                        "source_url": ev.get("url"),
                        "directness_score": ev.get("directness_score"),
                    }
                )
            )

        return claims

    def _build_rag_evidence_snapshot(
        self,
        rag_evidence: List[Dict[str, Any]],
        source_grounded_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        relevance_counts = Counter(
            ev.get("evidence_relevance")
            for ev in rag_evidence
            if ev.get("evidence_relevance")
        )

        grounded_relevance_counts = Counter(
            ev.get("evidence_relevance")
            for ev in source_grounded_evidence
            if ev.get("evidence_relevance")
        )

        return {
            "available_rag_evidence_count": len(rag_evidence),
            "source_grounded_evidence_count": len(source_grounded_evidence),
            "available_relevance_counts": dict(relevance_counts),
            "grounded_relevance_counts": dict(grounded_relevance_counts),
            "status": "supported" if source_grounded_evidence else "no_verified_rag_evidence",
        }

    def _flatten_source_grounded_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        flattened = []

        for item in evidence_items:
            ticker = item.get("ticker")
            company_name = item.get("company_name")
            rank = item.get("rank")

            for ev in item.get("source_grounded_evidence", []) or []:
                record = dict(ev)
                record["candidate_rank"] = rank
                record["candidate_ticker"] = ticker
                record["candidate_company_name"] = company_name
                flattened.append(record)

        return flattened

    def _build_rag_evidence_summary(
        self,
        news_result: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        source_grounded_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        news_result = news_result or {}
        rag_payload = news_result.get("rag", {}) if isinstance(news_result.get("rag"), dict) else {}

        verification_result = (
            news_result.get("verification_result")
            if isinstance(news_result.get("verification_result"), dict)
            else rag_payload.get("verification_result", {})
        )

        retrieved_evidence = (
            news_result.get("retrieved_evidence")
            if isinstance(news_result.get("retrieved_evidence"), list)
            else rag_payload.get("retrieved_evidence", [])
        )

        retrieval_queries = (
            news_result.get("retrieval_queries")
            if isinstance(news_result.get("retrieval_queries"), list)
            else rag_payload.get("retrieval_queries", [])
        )

        source_relevance_counts = Counter(
            ev.get("evidence_relevance")
            for ev in source_grounded_evidence
            if ev.get("evidence_relevance")
        )

        candidate_coverage = []
        for item in evidence_items:
            candidate_coverage.append(
                {
                    "rank": item.get("rank"),
                    "ticker": item.get("ticker"),
                    "company_name": item.get("company_name"),
                    "source_grounded_evidence_count": len(item.get("source_grounded_evidence", []) or []),
                    "rag_status": item.get("rag_evidence_snapshot", {}).get("status"),
                }
            )

        return self._clean_for_json(
            {
                "rag_enabled": bool(
                    news_result.get("rag_enabled")
                    or news_result.get("meta", {}).get("rag_enabled", False)
                    or rag_payload.get("enabled", False)
                    or source_grounded_evidence
                ),
                "rag_method": rag_payload.get("method") or news_result.get("meta", {}).get("rag_method"),
                "retrieval_query_count": len(retrieval_queries) if isinstance(retrieval_queries, list) else None,
                "retrieved_evidence_count": len(retrieved_evidence) if isinstance(retrieved_evidence, list) else None,
                "source_grounded_evidence_count": len(source_grounded_evidence),
                "source_grounded_relevance_counts": dict(source_relevance_counts),
                "verification_result": verification_result,
                "candidate_coverage": candidate_coverage,
            }
        )

    # ============================================================
    # Mapping / Data Building
    # ============================================================

    def _build_ticker_map(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result = {}

        for item in items:
            ticker = self._normalize_ticker(item.get("ticker"))

            if ticker:
                result[ticker] = item

        return result

    def _build_latest_row_map(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Dict[str, Dict[str, Any]]:
        if df.empty or "ticker" not in df.columns:
            return {}

        work = df.copy()

        if date_col in work.columns:
            work = work.sort_values(["ticker", date_col])
        else:
            work = work.sort_values(["ticker"])

        result = {}
        for ticker, group in work.groupby("ticker"):
            result[ticker] = self._clean_for_json(group.tail(1).iloc[0].to_dict())

        return result

    def _build_chart_data(self, ticker: str, ohlcv_df: pd.DataFrame, window: int) -> Dict[str, Any]:
        empty = {
            "period": {"start": None, "end": None, "n_points": 0},
            "recent_price_series": [],
            "recent_volume_series": [],
            "moving_average_series_optional": [],
            "chart_summary": {},
            "latest_row": {},
        }

        if ohlcv_df.empty or "ticker" not in ohlcv_df.columns:
            return empty

        stock_df = ohlcv_df[ohlcv_df["ticker"] == ticker].copy()
        if stock_df.empty:
            return empty

        if "date" in stock_df.columns:
            stock_df = stock_df.sort_values("date")

        recent = stock_df.tail(window).copy()
        latest = recent.tail(1).iloc[0].to_dict() if not recent.empty else {}

        price_cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "daily_return",
            "ma20_close",
            "ma60_close",
            "rsi_14",
            "market_regime_20",
            "market_regime_60",
        ]
        price_cols = [c for c in price_cols if c in recent.columns]

        volume_cols = [
            "date",
            "volume",
            "volume_zscore_20d",
            "volume_zscore_60d",
        ]
        volume_cols = [c for c in volume_cols if c in recent.columns]

        ma_cols = [
            "date",
            "close",
            "ma20_close",
            "ma60_close",
            "ma120_close",
        ]
        ma_cols = [c for c in ma_cols if c in recent.columns]

        recent_price_series = self._df_records(recent[price_cols])
        recent_volume_series = self._df_records(recent[volume_cols])
        moving_average_series = self._df_records(recent[ma_cols])

        start_date = self._date_to_str(recent["date"].iloc[0]) if "date" in recent.columns and len(recent) else None
        end_date = self._date_to_str(recent["date"].iloc[-1]) if "date" in recent.columns and len(recent) else None

        latest_close = self._safe_float(latest.get("close"))
        first_close = self._safe_float(recent["close"].iloc[0]) if "close" in recent.columns and len(recent) else None
        latest_volume = self._safe_float(latest.get("volume"))

        avg_volume_20d = None
        avg_volume_60d = None

        if "volume" in stock_df.columns and len(stock_df) > 0:
            avg_volume_20d = self._safe_float(stock_df["volume"].tail(20).mean())
            avg_volume_60d = self._safe_float(stock_df["volume"].tail(60).mean())

        close_change = None
        if latest_close is not None and first_close not in (None, 0):
            close_change = latest_close / first_close - 1

        volume_vs_20d_avg = None
        if latest_volume is not None and avg_volume_20d not in (None, 0):
            volume_vs_20d_avg = latest_volume / avg_volume_20d

        chart_summary = {
            "latest_date": end_date,
            "latest_close": latest_close,
            f"close_change_{len(recent)}d": close_change,
            "latest_volume": latest_volume,
            "avg_volume_20d": avg_volume_20d,
            "avg_volume_60d": avg_volume_60d,
            "volume_vs_20d_avg": volume_vs_20d_avg,
            "ma20_close": self._safe_float(latest.get("ma20_close")),
            "ma60_close": self._safe_float(latest.get("ma60_close")),
            "rsi_14": self._safe_float(latest.get("rsi_14")),
            "market_regime_20": latest.get("market_regime_20"),
            "market_regime_60": latest.get("market_regime_60"),
        }

        return self._clean_for_json(
            {
                "period": {
                    "start": start_date,
                    "end": end_date,
                    "n_points": int(len(recent)),
                },
                "recent_price_series": recent_price_series,
                "recent_volume_series": recent_volume_series,
                "moving_average_series_optional": moving_average_series,
                "chart_summary": chart_summary,
                "latest_row": latest,
            }
        )

    def _build_actual_data(
        self,
        candidate: Dict[str, Any],
        latest_price_row: Dict[str, Any],
        finance_item: Dict[str, Any],
        foreign_item: Dict[str, Any],
        risk_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate_actual = candidate.get("actual_data", {}) or {}
        risk_scores = risk_item.get("risk_scores", {}) or {}

        latest_close = self._first_non_null(
            latest_price_row.get("close"),
            candidate_actual.get("latest_close"),
        )
        volume = self._first_non_null(
            latest_price_row.get("volume"),
            candidate_actual.get("volume"),
        )

        trading_value = None
        latest_close_num = self._safe_float(latest_close)
        volume_num = self._safe_float(volume)

        if latest_close_num is not None and volume_num is not None:
            trading_value = latest_close_num * volume_num

        actual_data = {
            "latest_date": self._date_to_str(latest_price_row.get("date")),
            "latest_close": latest_close,
            "daily_return": self._first_non_null(
                latest_price_row.get("daily_return"),
                candidate_actual.get("daily_return"),
            ),
            "volume": volume,
            "trading_value": trading_value,
            "foreign_net_flow_ratio": self._first_non_null(
                foreign_item.get("foreign_net_flow_ratio"),
                candidate_actual.get("foreign_net_flow_ratio"),
            ),
            "foreign_ownership_level": foreign_item.get("foreign_ownership_level"),
            "roe": self._first_non_null(
                finance_item.get("roe"),
                candidate_actual.get("roe"),
            ),
            "roa": finance_item.get("roa"),
            "revenue_yoy": self._first_non_null(
                finance_item.get("revenue_yoy"),
                candidate_actual.get("revenue_yoy"),
            ),
            "operating_income_yoy": self._first_non_null(
                finance_item.get("operating_income_yoy"),
                candidate_actual.get("operating_income_yoy"),
            ),
            "debt_ratio": finance_item.get("debt_ratio"),
            "current_ratio": finance_item.get("current_ratio"),
            "overall_risk_score": self._first_non_null(
                risk_scores.get("overall_risk_score"),
                candidate_actual.get("overall_risk_score"),
            ),
            "risk_level": self._first_non_null(
                risk_scores.get("risk_level"),
                candidate_actual.get("risk_level"),
            ),
            "market_cap": self._first_non_null(
                candidate_actual.get("market_cap"),
                finance_item.get("market_cap"),
            ),
            "per": self._first_non_null(
                candidate_actual.get("per"),
                finance_item.get("per"),
            ),
        }

        return self._clean_for_json(actual_data)

    def _select_top_news(
        self,
        ticker: str,
        candidate: Dict[str, Any],
        news_item: Dict[str, Any],
        raw_news_df: pd.DataFrame,
        rag_evidence: Optional[List[Dict[str, Any]]] = None,
        max_news: int = 3,
    ) -> List[Dict[str, Any]]:
        collected = []

        def add_article(article: Dict[str, Any], source_type: str = "internal_news") -> None:
            if not isinstance(article, dict):
                return

            title = article.get("title")
            url = article.get("url") or article.get("link") or article.get("originallink")

            if not title:
                return

            key = (str(title).strip(), str(url or "").strip())
            existing_keys = {
                (str(x.get("title", "")).strip(), str(x.get("url", "")).strip())
                for x in collected
            }

            if key in existing_keys:
                return

            collected.append(
                self._clean_for_json(
                    {
                        "title": title,
                        "source": article.get("source") or article.get("press"),
                        "published_at": article.get("published_at")
                        or article.get("date")
                        or article.get("pubDate"),
                        "url": url,
                        "summary_optional": article.get("summary")
                        or article.get("description")
                        or article.get("snippet"),
                        "article_score": article.get("article_score"),
                        "rule_score_0_1": article.get("rule_score_0_1"),

                        # RAG source metadata
                        "source_type": source_type,
                        "support_type": "rag_verified_news" if source_type == "rag_verified_news" else None,
                        "evidence_relevance": article.get("evidence_relevance"),
                        "directness_score": article.get("directness_score"),
                        "verification_reasons": article.get("verification_reasons"),
                    }
                )
            )

        for ev in rag_evidence or []:
            add_article(ev, source_type="rag_verified_news")

            if len(collected) >= max_news:
                return collected[:max_news]

        for article in candidate.get("top_articles", []) or []:
            add_article(article)
            if len(collected) >= max_news:
                return collected[:max_news]

        for article in news_item.get("top_articles", []) or []:
            add_article(article)
            if len(collected) >= max_news:
                return collected[:max_news]

        if not raw_news_df.empty and "ticker" in raw_news_df.columns:
            stock_news = raw_news_df[raw_news_df["ticker"] == ticker].copy()

            if not stock_news.empty:
                if "pubDate" in stock_news.columns:
                    stock_news = stock_news.sort_values("pubDate", ascending=False)

                for _, row in stock_news.iterrows():
                    add_article(row.to_dict())
                    if len(collected) >= max_news:
                        break

        return collected[:max_news]

    def _build_ranking_reason(
        self,
        candidate: Dict[str, Any],
        market_item: Dict[str, Any],
        risk_item: Dict[str, Any],
        rag_supported_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        component_scores = candidate.get("component_scores", {}) or {}

        sorted_components = sorted(
            [
                (key, self._safe_float(value))
                for key, value in component_scores.items()
                if self._safe_float(value) is not None
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        top_components = sorted_components[:2]

        labels = {
            "market_flow_alignment_score": "시장 흐름 연관도",
            "price_volume_momentum_score": "가격·거래량 모멘텀",
            "news_momentum_score": "뉴스 모멘텀",
            "news_direct_relevance_score": "뉴스 직접 관련도",
            "foreign_flow_score": "외국인 수급",
            "fundamental_score": "재무 지표",
            "risk_penalty_score": "리스크 관리",
            "user_interest_boost": "사용자 관심도",
        }

        if top_components:
            top_text = ", ".join(
                f"{labels.get(key, key)}({value:.1f})"
                for key, value in top_components
            )
            main_reason = f"{top_text} 신호가 상대적으로 높아 시장 흐름 기반 후보로 선정되었습니다."
        else:
            main_reason = candidate.get("ranking_reason_short") or "입력된 점수 근거를 기반으로 후보에 포함되었습니다."

        supporting_reasons = self._merge_unique_strings(
            candidate.get("positive_evidence", []),
            market_item.get("positive_factors", []),
            [
                claim.get("claim")
                for claim in (rag_supported_claims or [])
                if isinstance(claim, dict) and claim.get("claim")
            ],
            limit=6,
        )

        if not supporting_reasons and candidate.get("ranking_reason_short"):
            supporting_reasons = [candidate["ranking_reason_short"]]

        risk_evidence = self._merge_unique_strings(
            risk_item.get("evidence", []),
            market_item.get("risk_factors", []),
            candidate.get("negative_evidence", []),
            limit=3,
        )

        risk_level = None
        if risk_item.get("risk_scores"):
            risk_level = risk_item["risk_scores"].get("risk_level")
        risk_level = risk_level or candidate.get("actual_data", {}).get("risk_level")

        if risk_evidence:
            caution_note = f"다만 {risk_evidence[0]}"
        elif risk_level:
            caution_note = f"다만 risk_level={risk_level}로 분류되어 리스크 확인이 필요합니다."
        else:
            caution_note = "다만 세부 수치와 리스크 요인은 추가 확인이 필요합니다."

        beginner_explanation = self._make_beginner_explanation(
            top_components,
            labels,
            risk_level,
        )

        return self._clean_for_json(
            {
                "main_reason": main_reason,
                "supporting_reasons": supporting_reasons,
                "beginner_explanation": beginner_explanation,
                "caution_note": caution_note,
                "score_basis": {
                    "top_component_scores": [
                        {
                            "name": key,
                            "label": labels.get(key, key),
                            "score": value,
                        }
                        for key, value in top_components
                    ],
                    "all_component_scores": component_scores,
                },
                "rag_grounding_note": (
                    "RAG verified news evidence is available."
                    if rag_supported_claims else
                    "No verified RAG evidence was attached for this candidate."
                ),
                "original_ranking_reason_short": candidate.get("ranking_reason_short"),
            }
        )

    def _build_data_quality(
        self,
        actual_data: Dict[str, Any],
        chart_data: Dict[str, Any],
        top_news_3: List[Dict[str, Any]],
        source_grounded_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_grounded_evidence = source_grounded_evidence or []
        missing_fields = []

        for key, value in actual_data.items():
            if value is None:
                missing_fields.append(f"actual_data.{key}")

        n_chart_points = chart_data.get("period", {}).get("n_points", 0)

        if not n_chart_points:
            missing_fields.append("chart_data.recent_price_series")

        if len(top_news_3) < 3:
            missing_fields.append("top_news_3")

        return {
            "chart_points": n_chart_points,
            "news_count": len(top_news_3),
            "rag_evidence_count": len(source_grounded_evidence),
            "missing_fields": missing_fields,
            "news_status": "ok" if len(top_news_3) >= 3 else "insufficient_articles_available",
            "rag_status": "supported" if source_grounded_evidence else "no_verified_rag_evidence",
        }

    def _build_market_context(
        self,
        candidate_result: Dict[str, Any],
        market_flow_result: Dict[str, Any],
        market_analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate_market = candidate_result.get("market_flow_summary", {}) or {}
        market_flow = self._extract_market_flow(market_flow_result)
        overview = market_analysis_result.get("market_overview", {}) if market_analysis_result else {}

        return self._clean_for_json(
            {
                "as_of_date": self._first_non_null(
                    candidate_market.get("as_of_date"),
                    market_flow.get("as_of_date"),
                    overview.get("as_of_date"),
                ),
                "market_one_line_summary": self._first_non_null(
                    candidate_market.get("market_one_line_summary"),
                    market_flow.get("market_one_line_summary"),
                    overview.get("summary"),
                ),
                "market_brief": self._first_non_null(
                    candidate_market.get("market_brief"),
                    market_flow.get("market_brief"),
                    overview.get("summary"),
                ),
                "market_keywords": self._first_non_null(
                    candidate_market.get("market_keywords"),
                    market_flow.get("market_keywords"),
                    [],
                ),
                "core_market_drivers": self._first_non_null(
                    candidate_market.get("core_market_drivers"),
                    market_flow.get("core_market_drivers"),
                    [],
                ),
                "market_phase": overview.get("market_phase"),
                "market_tone": overview.get("market_tone"),
                "market_risk_notes": overview.get("risk_notes", []),
                "report_summary_box": self._first_non_null(
                    candidate_market.get("report_summary_box"),
                    market_flow.get("report_summary_box"),
                    {},
                ),
            }
        )

    def _extract_market_flow(self, market_flow_result: Dict[str, Any]) -> Dict[str, Any]:
        if not market_flow_result:
            return {}
        return market_flow_result.get("market_flow", {}) or {}

    def _build_market_analysis_snapshot(self, market_item: Dict[str, Any]) -> Dict[str, Any]:
        if not market_item:
            return {}

        return self._clean_for_json(
            {
                "market": market_item.get("market"),
                "ticker_type": market_item.get("ticker_type"),
                "phase": market_item.get("phase"),
                "tone": market_item.get("tone"),
                "rsi_state": market_item.get("rsi_state"),
                "return_state": market_item.get("return_state"),
                "market_context": market_item.get("market_context"),
                "positive_factors": market_item.get("positive_factors", []),
                "risk_factors": market_item.get("risk_factors", []),
                "evidence": market_item.get("evidence", []),
                "compact_signals": market_item.get("compact_signals", {}),
            }
        )

    def _build_news_signal_snapshot(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        if not news_item:
            return {}

        rag_info = news_item.get("rag", {}) if isinstance(news_item.get("rag"), dict) else {}

        return self._clean_for_json(
            {
                "news_signal_score": news_item.get("news_signal_score"),
                "confidence_level": news_item.get("confidence_level"),
                "verdict": news_item.get("verdict"),
                "reasons": news_item.get("reasons", []),
                "news_summary": news_item.get("news_summary", {}),
                "rag_verification_status": rag_info.get("verification_status"),
                "rag_evidence_count": rag_info.get("evidence_count"),
                "rag_relevance_counts": rag_info.get("relevance_counts", {}),
            }
        )

    def _build_risk_snapshot(self, risk_item: Dict[str, Any]) -> Dict[str, Any]:
        if not risk_item:
            return {}

        return self._clean_for_json(
            {
                "risk_scores": risk_item.get("risk_scores", {}),
                "dominant_risk_factors": risk_item.get("dominant_risk_factors", []),
                "evidence": risk_item.get("evidence", []),
            }
        )

    # ============================================================
    # Static Report Helpers
    # ============================================================

    def _build_glossary_terms(self) -> List[Dict[str, str]]:
        return [
            {
                "term": "거래대금",
                "plain_korean_definition": "하루 동안 해당 주식이 거래된 금액입니다. 보통 종가와 거래량을 곱해 참고합니다.",
                "why_it_matters": "거래대금이 크면 시장 참여자들의 관심이 높고 매매가 활발하다고 볼 수 있습니다.",
            },
            {
                "term": "가격·거래량 모멘텀",
                "plain_korean_definition": "최근 주가와 거래량이 함께 강해지는 흐름입니다.",
                "why_it_matters": "단기적으로 시장 관심이 붙었는지 확인할 때 사용합니다.",
            },
            {
                "term": "외국인 순매수 비율",
                "plain_korean_definition": "거래량 대비 외국인이 순매수한 정도를 나타낸 값입니다.",
                "why_it_matters": "외국인 수급이 들어오는지 확인하는 보조 지표입니다.",
            },
            {
                "term": "ROE",
                "plain_korean_definition": "자기자본 대비 순이익을 얼마나 냈는지 보여주는 수익성 지표입니다.",
                "why_it_matters": "기업이 가진 자본을 얼마나 효율적으로 활용하는지 볼 때 사용합니다.",
            },
            {
                "term": "리스크 점수",
                "plain_korean_definition": "변동성, 낙폭, 재무 부담, 뉴스 이벤트 등을 종합한 위험도 점수입니다.",
                "why_it_matters": "점수가 높을수록 가격 흔들림이나 불확실성에 더 주의해야 합니다.",
            },
        ]

    def _build_data_limitations(
        self,
        candidate_result: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
    ) -> List[str]:
        limitations = list(candidate_result.get("limitations", []) or [])

        base = [
            "본 결과는 투자 추천이 아니라 시장 흐름 기반 리포트 생성을 위한 근거 패키지입니다.",
            "candidate_score는 기대수익률이나 수익 보장 확률이 아닙니다.",
            "입력 데이터에 없는 market_cap, PER 등은 추정하지 않고 null로 처리합니다.",
            "뉴스가 3개 미만인 종목은 확인된 기사 범위 안에서만 제공합니다.",
            "RAG 기반 source_grounded_evidence는 수집된 뉴스 corpus 안에서 검색·검증된 근거이며, 실시간 외부 웹 검색 결과가 아닙니다.",
            "direct_business_relation evidence는 해당 기업 자체 뉴스가 아니라 사업 관계 또는 시장 이벤트 맥락의 근거일 수 있습니다.",
        ]

        return self._merge_unique_strings(limitations, base, limit=20)

    def _build_source_trace(self, paths: Dict[str, Optional[Path]]) -> Dict[str, Optional[str]]:
        return {
            key: str(path) if path else None
            for key, path in paths.items()
        }

    def _build_manifest(
        self,
        result: Dict[str, Any],
        paths: Dict[str, Optional[Path]],
        result_path: Path,
    ) -> Dict[str, Any]:
        evidence_items = result.get("top10_evidence", [])
        rag_summary = result.get("rag_evidence_summary", {}) or {}

        return self._clean_for_json(
            {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc_iso(),
                "input_files": self._build_source_trace(paths),
                "output_files": {
                    "report_evidence_result": str(result_path),
                },
                "summary": {
                    "n_candidates": len(evidence_items),
                    "as_of_date": result.get("as_of_date"),
                    "rag_enabled": rag_summary.get("rag_enabled"),
                    "source_grounded_evidence_count": rag_summary.get("source_grounded_evidence_count"),
                    "candidates": [
                        {
                            "rank": item.get("rank"),
                            "ticker": item.get("ticker"),
                            "company_name": item.get("company_name"),
                            "news_count": item.get("data_quality", {}).get("news_count"),
                            "rag_evidence_count": item.get("data_quality", {}).get("rag_evidence_count"),
                            "chart_points": item.get("data_quality", {}).get("chart_points"),
                            "missing_fields": item.get("data_quality", {}).get("missing_fields", []),
                        }
                        for item in evidence_items
                    ],
                },
            }
        )

    # ============================================================
    # Utility
    # ============================================================

    def _ctx_get(self, ctx: RunContext, key: str, default: Any = None) -> Any:
        if isinstance(ctx, dict):
            return ctx.get(key, default)

        return getattr(ctx, key, default)

    def _normalize_ticker(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, float) and math.isnan(value):
            return ""

        text = str(value).strip()

        if not text:
            return ""

        try:
            if text.replace(".", "", 1).isdigit():
                text = str(int(float(text)))
        except Exception:
            pass

        return text.zfill(6)

    def _df_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return [
            self._clean_for_json(record)
            for record in df.to_dict(orient="records")
        ]

    def _strip_internal_chart_fields(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(chart_data)
        cleaned.pop("latest_row", None)
        return cleaned

    def _first_non_null(self, *values: Any) -> Any:
        for value in values:
            if value is None:
                continue

            if isinstance(value, float) and math.isnan(value):
                continue

            if isinstance(value, str) and value.strip() == "":
                continue

            return value

        return None

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            result = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(result) or math.isinf(result):
            return None

        return result

    def _date_to_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, float) and math.isnan(value):
            return None

        try:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return str(value)

    def _clean_for_json(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                str(key): self._clean_for_json(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                self._clean_for_json(value)
                for value in obj
            ]

        if isinstance(obj, tuple):
            return [
                self._clean_for_json(value)
                for value in obj
            ]

        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None

            if obj.time().isoformat() == "00:00:00":
                return obj.strftime("%Y-%m-%d")

            return obj.strftime("%Y-%m-%d %H:%M:%S")

        try:
            if not isinstance(obj, (list, dict, tuple)) and pd.isna(obj):
                return None
        except Exception:
            pass

        if hasattr(obj, "item"):
            try:
                return self._clean_for_json(obj.item())
            except Exception:
                pass

        return obj

    def _merge_unique_strings(self, *groups: Any, limit: Optional[int] = None) -> List[str]:
        merged = []
        seen = set()

        for group in groups:
            if not group:
                continue

            if isinstance(group, str):
                iterable = [group]
            else:
                iterable = group

            for item in iterable:
                if item is None:
                    continue

                text = str(item).strip()

                if not text or text in seen:
                    continue

                seen.add(text)
                merged.append(text)

                if limit is not None and len(merged) >= limit:
                    return merged

        return merged

    def _make_beginner_explanation(
        self,
        top_components: List[Tuple[str, float]],
        labels: Dict[str, str],
        risk_level: Optional[str],
    ) -> str:
        if not top_components:
            base = "이 종목은 여러 점수 지표를 종합했을 때 TOP10 후보에 포함되었습니다."
        else:
            names = [
                labels.get(key, key)
                for key, _ in top_components
            ]

            if len(names) == 1:
                base = f"이 종목은 {names[0]} 신호가 상대적으로 좋아 후보에 포함되었습니다."
            else:
                base = f"이 종목은 {names[0]}와 {names[1]} 신호가 상대적으로 좋아 후보에 포함되었습니다."

        if risk_level:
            return f"{base} 다만 리스크 등급은 {risk_level}이므로, 가격 변동 가능성도 함께 확인해야 합니다."

        return f"{base} 다만 후보 선정은 수익을 보장하는 의미가 아니므로, 리스크도 함께 확인해야 합니다."

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _now_utc_iso(self) -> str:
        return self._now_utc()

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _make_stage_result(
        self,
        status: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        sig = inspect.signature(StageResult)

        candidate_kwargs = {
            "stage": self.stage,
            "status": status,
            "message": message,
            "metrics": metrics or {},
            "outputs": outputs or {},
            "artifacts": outputs or {},
        }

        kwargs = {
            k: v
            for k, v in candidate_kwargs.items()
            if k in sig.parameters
        }

        return StageResult(**kwargs)