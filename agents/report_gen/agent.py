from __future__ import annotations

import copy
import inspect
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


class ReportGenerativeAgent(BaseAgent):
    stage = "report_generation"
    version = "1.1.0"

    COMPONENT_LABELS = {
        "market_flow_alignment_score": "시장 흐름 연관도",
        "news_direct_relevance_score": "뉴스 직접 관련도",
        "news_momentum_score": "뉴스 모멘텀",
        "price_volume_momentum_score": "가격·거래량 모멘텀",
        "foreign_flow_score": "외국인 수급",
        "fundamental_score": "재무 지표",
        "risk_penalty_score": "리스크 부담",
        "user_interest_boost": "사용자 관심 반영",
    }

    RELEVANCE_LABELS = {
        "direct_company_news": "기업 직접 뉴스",
        "direct_business_relation": "사업 관계 뉴스",
        "indirect_mention": "간접 언급",
        "market_context": "시장 맥락",
        "low_relevance": "낮은 관련도",
    }

    def __init__(
        self,
        gemini_model_name: str = "gemini-2.5-flash",
        openai_model_name: str = "gpt-4.1-mini",
        llm_provider: str = "auto",
        use_llm: bool = True,
        encoding: str = "utf-8",
        max_news_per_candidate: int = 3,
        max_source_grounded_evidence_per_candidate: int = 3,
        model_name: Optional[str] = None,
    ):
        if model_name is not None:
            gemini_model_name = model_name

        self.gemini_model_name = gemini_model_name
        self.openai_model_name = openai_model_name
        self.llm_provider = llm_provider
        self.use_llm = use_llm
        self.encoding = encoding
        self.max_news_per_candidate = max_news_per_candidate
        self.max_source_grounded_evidence_per_candidate = max_source_grounded_evidence_per_candidate

    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        load_dotenv()

        run_dir = Path(ctx.artifact_root)
        output_dir = run_dir / self.stage
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = output_dir / "report_output.json"
        manifest_path = output_dir / "manifest.json"

        ctx.logger.info("[ReportGenerativeAgent] Starting report generation.")

        evidence_path = self._find_single_file(
            run_dir,
            "report_evidence_result.json",
            ctx,
        )

        if evidence_path is None:
            raise FileNotFoundError(
                "[ReportGenerativeAgent] report_evidence_result.json을 찾을 수 없습니다."
            )

        evidence_manifest_path = evidence_path.parent / "manifest.json"
        if not evidence_manifest_path.exists():
            evidence_manifest_path = None

        report_output_contract_path = self._find_single_file(
            run_dir,
            "report_output_contract.json",
            ctx,
        )

        evidence = self._safe_read_json(evidence_path, ctx)
        evidence_manifest = self._safe_read_json(evidence_manifest_path, ctx)
        report_output_contract = self._safe_read_json(report_output_contract_path, ctx)

        if not evidence:
            raise ValueError(
                "[ReportGenerativeAgent] report_evidence_result.json이 비어 있거나 읽을 수 없습니다."
            )

        llm_pack, llm_status = self._generate_text_pack(
            evidence=evidence,
            ctx=ctx,
        )

        if llm_pack is None:
            llm_pack = {}

        report = self._build_report_output(
            evidence=evidence,
            evidence_path=evidence_path,
            evidence_manifest=evidence_manifest,
            evidence_manifest_path=evidence_manifest_path,
            report_output_contract=report_output_contract,
            report_output_contract_path=report_output_contract_path,
            llm_pack=llm_pack,
            llm_status=llm_status,
            ctx=ctx,
        )

        is_valid, warnings = self._validate_report_output(report)

        report["quality_checks"] = {
            "is_valid": is_valid,
            "warnings": warnings,
        }

        self._save_json(report, result_path)

        manifest = self._build_manifest(
            evidence_path=evidence_path,
            evidence_manifest_path=evidence_manifest_path,
            report_output_contract_path=report_output_contract_path,
            result_path=result_path,
            report=report,
            is_valid=is_valid,
            warnings=warnings,
            llm_status=llm_status,
        )

        self._save_json(manifest, manifest_path)

        candidates = self._safe_get(report, "candidate_section.candidates", default=[])

        metrics = {
            "is_valid": is_valid,
            "n_warnings": len(warnings),
            "n_candidates": len(candidates),
            "as_of_date": self._safe_get(report, "report_meta.as_of_date"),
            "llm_used": llm_status.get("used"),
            "llm_provider": llm_status.get("provider"),
            "n_candidates_with_source_grounding": sum(
                1
                for c in candidates
                if isinstance(c, dict)
                and self._safe_get(c, "source_grounding.evidence_count", default=0) > 0
            ),
            "n_source_grounded_evidence": self._safe_get(
                report,
                "source_grounding_summary.source_grounded_evidence_count",
                default=0,
            ),
            "rag_enabled": self._safe_get(
                report,
                "source_grounding_summary.rag_enabled",
                default=False,
            ),
        }

        outputs = {
            "output_dir": str(output_dir),
            "report_output": str(result_path),
            "manifest": str(manifest_path),
        }

        if warnings:
            for warning in warnings:
                ctx.logger.warning(f"[ReportGenerativeAgent] {warning}")

        ctx.logger.info(
            "[ReportGenerativeAgent] Completed report generation. "
            f"source_grounded_evidence={metrics['n_source_grounded_evidence']}"
        )

        return self._make_stage_result(
            status="success",
            message="Report generation completed.",
            metrics=metrics,
            outputs=outputs,
        )

    # ============================================================
    # Report builder
    # ============================================================

    def _build_report_output(
        self,
        evidence: Dict[str, Any],
        evidence_path: Path,
        evidence_manifest: Optional[Dict[str, Any]],
        evidence_manifest_path: Optional[Path],
        report_output_contract: Optional[Dict[str, Any]],
        report_output_contract_path: Optional[Path],
        llm_pack: Dict[str, Any],
        llm_status: Dict[str, Any],
        ctx: RunContext,
    ) -> Dict[str, Any]:
        report_context = evidence.get("report_context", {}) or {}
        market_context = evidence.get("market_context", {}) or {}
        top10 = evidence.get("top10_evidence", []) or []

        as_of_date = (
            evidence.get("as_of_date")
            or market_context.get("as_of_date")
            or self._safe_get(evidence_manifest, "summary.as_of_date")
        )

        report_title = self._first_non_empty(
            report_context.get("report_title"),
            default="시장 흐름 기반 유망 종목 후보 리포트",
        )

        llm_market_text = llm_pack.get("market_summary")
        llm_candidates = llm_pack.get("candidates", {})

        if not isinstance(llm_candidates, dict):
            llm_candidates = {}

        source_grounding_summary = self._build_source_grounding_summary(evidence)

        candidate_cards = []

        for item in sorted(top10, key=lambda x: x.get("rank", 999)):
            ticker = str(item.get("ticker", ""))

            llm_candidate_text = llm_candidates.get(ticker)
            if not isinstance(llm_candidate_text, dict):
                llm_candidate_text = self._fallback_candidate_text(item)

            candidate_cards.append(
                self._build_candidate_card(
                    item=item,
                    llm_candidate_text=llm_candidate_text,
                )
            )

        report = {
            "meta": {
                "agent": self.__class__.__name__,
                "stage": self.stage,
                "version": self.version,
                "created_at_utc": self._now_utc(),
                "purpose": "프론트엔드 카드형 UI 렌더링을 위한 최종 리포트 JSON 생성",
                "llm_status": llm_status,
            },
            "report_meta": {
                "report_type": report_context.get(
                    "report_type",
                    "market_flow_candidate_report",
                ),
                "title": report_title,
                "subtitle": (
                    "시장 상황을 먼저 요약하고, 그 흐름에 맞는 관심 종목 후보를 제시하는 "
                    "초보 투자자용 리포트"
                ),
                "as_of_date": as_of_date,
                "ranking_target": report_context.get("ranking_target"),
                "ranking_scope": report_context.get("ranking_scope"),
                "candidate_definition": report_context.get("candidate_definition", {}),
                "evidence_grounding": {
                    "rag_enabled": source_grounding_summary.get("rag_enabled"),
                    "source_grounded_evidence_count": source_grounding_summary.get(
                        "source_grounded_evidence_count"
                    ),
                    "description": (
                        "Ours B에서는 수집 뉴스 corpus 기반 RAG evidence가 "
                        "종목별 근거로 함께 반영됩니다."
                        if source_grounding_summary.get("rag_enabled")
                        else "RAG 기반 source-grounded evidence는 사용되지 않았습니다."
                    ),
                },
            },
            "ui_contract": {
                "layout": "market_summary_then_candidate_cards_then_glossary",
                "components": [
                    "MarketSummaryCard",
                    "CandidateStockCard",
                    "StockPriceChart",
                    "ActualDataTable",
                    "NewsList",
                    "EntryReasonBox",
                    "SourceGroundedEvidenceList",
                    "GlossarySection",
                ],
                "frontend_notes": [
                    "chart.series.price는 차트 렌더링에 사용합니다.",
                    "actual_data.items는 표 형태로 표시합니다.",
                    "top_news.url은 뉴스 링크 버튼에 연결합니다.",
                    "source_grounded_evidence는 RAG로 검증된 뉴스 근거입니다.",
                    "source_grounding.evidence_count가 0이면 근거 검증 배지를 숨길 수 있습니다.",
                    "is_missing=true인 데이터는 '-' 또는 '데이터 없음'으로 표시합니다.",
                    "이 리포트는 투자 추천이 아니라 관심 후보 설명 리포트입니다.",
                ],
            },
            "disclaimer": {
                "short": (
                    "본 리포트는 투자 추천이 아니라 시장 흐름 기반 관심 후보를 정리한 참고 자료입니다."
                ),
                "long": (
                    "본 결과는 시세, 뉴스, 수급, 재무, 리스크 데이터를 기반으로 생성된 "
                    "정보성 리포트입니다. 특정 종목의 매수·매도 추천, 수익률 예측, "
                    "수익 보장을 의미하지 않습니다."
                ),
                "source_grounding_note": (
                    "뉴스 근거는 수집된 뉴스 corpus 안에서 검색·검증된 자료를 기반으로 하며, "
                    "실시간 외부 웹 검색 결과가 아닙니다."
                ),
            },
            "market_summary": self._build_market_summary(
                evidence=evidence,
                llm_market_text=llm_market_text,
                source_grounding_summary=source_grounding_summary,
            ),
            "source_grounding_summary": source_grounding_summary,
            "candidate_section": {
                "title": "유망 + 관심 종목 후보 TOP 10",
                "subtitle": "오늘 시장 흐름과 데이터상 관심 있게 확인할 만한 종목 후보입니다.",
                "candidate_count": len(candidate_cards),
                "candidates": candidate_cards,
            },
            "glossary": {
                "title": "용어 풀이",
                "items": self._build_glossary(evidence),
            },
            "data_limitations": evidence.get("data_limitations", []),
            "source_trace": {
                "report_evidence_result": str(evidence_path),
                "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path else None,
                "report_output_contract": (
                    str(report_output_contract_path) if report_output_contract_path else None
                ),
                "evidence_builder_manifest": evidence_manifest,
                "report_output_contract_object": report_output_contract,
            },
        }

        return report

    def _build_market_summary(
        self,
        evidence: Dict[str, Any],
        llm_market_text: Optional[Dict[str, Any]] = None,
        source_grounding_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        market_context = evidence.get("market_context", {}) or {}
        summary_box = market_context.get("report_summary_box", {}) or {}

        if not isinstance(llm_market_text, dict):
            llm_market_text = self._fallback_market_text(market_context)

        source_grounding_summary = source_grounding_summary or self._build_source_grounding_summary(evidence)

        return {
            "title": summary_box.get("title", "오늘 시장 한 줄 요약"),
            "headline": self._truncate_text(llm_market_text.get("headline"), 100),
            "body": self._truncate_text(llm_market_text.get("body"), 180),
            "source_note": summary_box.get(
                "source_note",
                "시세, 뉴스, 외국인 수급, 리스크 데이터 기반 추출",
            ),
            "market_phase": market_context.get("market_phase"),
            "market_tone": market_context.get("market_tone"),
            "market_tone_label": llm_market_text.get(
                "tone_label",
                self._tone_label_ko(market_context.get("market_tone")),
            ),
            "keywords": market_context.get("market_keywords", []),
            "core_drivers": market_context.get("core_market_drivers", []),
            "risk_notes": market_context.get("market_risk_notes", []),
            "raw_market_brief": market_context.get("market_brief"),
            "source_grounding": {
                "rag_enabled": source_grounding_summary.get("rag_enabled"),
                "source_grounded_evidence_count": source_grounding_summary.get(
                    "source_grounded_evidence_count"
                ),
                "candidate_coverage_count": source_grounding_summary.get(
                    "candidate_coverage_count"
                ),
                "evidence_relevance_counts": source_grounding_summary.get(
                    "evidence_relevance_counts",
                    {},
                ),
                "note": (
                    "일부 종목 설명에는 RAG로 검증된 뉴스 근거가 함께 연결되어 있습니다."
                    if source_grounding_summary.get("source_grounded_evidence_count", 0) > 0
                    else "이번 리포트에는 RAG로 연결된 뉴스 근거가 없습니다."
                ),
            },
        }

    def _build_candidate_card(
        self,
        item: Dict[str, Any],
        llm_candidate_text: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(llm_candidate_text, dict):
            llm_candidate_text = self._fallback_candidate_text(item)

        rank = item.get("rank")
        score = item.get("market_flow_candidate_score")
        actual_data = item.get("actual_data", {}) or {}
        chart_data = item.get("chart_data", {}) or {}
        component_scores = item.get("component_scores", {}) or {}

        source_grounded_evidence = self._build_source_grounded_evidence_items(
            item.get("source_grounded_evidence", []) or []
        )
        rag_supported_claims = self._build_rag_supported_claim_items(
            item.get("rag_supported_claims", []) or []
        )
        source_grounding_payload = self._build_candidate_source_grounding_payload(
            item=item,
            source_grounded_evidence=source_grounded_evidence,
            rag_supported_claims=rag_supported_claims,
        )

        return {
            "rank": rank,
            "badge_label": f"종목 {rank}",
            "ticker": item.get("ticker"),
            "company_name": item.get("company_name"),
            "as_of_date": item.get("as_of_date"),
            "score": self._round_or_none(score, 3),
            "score_display": self._format_score(score, 1),
            "quick_summary": self._truncate_text(
                llm_candidate_text.get("quick_summary"),
                120,
            ),
            "chart": self._build_chart_payload(chart_data),
            "actual_data": self._build_actual_data_table(actual_data),
            "component_scores": self._build_component_score_items(component_scores),
            "top_component_highlights": self._build_top_component_highlights(item),
            "top_news": self._build_news_items(
                news_list=item.get("top_news_3", []),
                llm_news_summaries=llm_candidate_text.get("news_summaries", []),
            ),
            "entry_reason": {
                "title": "순위 진입 이유",
                "body": self._truncate_text(
                    llm_candidate_text.get("entry_reason"),
                    240,
                ),
                "raw_main_reason": self._safe_get(item, "ranking_reason.main_reason"),
                "supporting_reasons": self._safe_get(
                    item,
                    "ranking_reason.supporting_reasons",
                    default=[],
                ),
                "source_grounded_note": self._truncate_text(
                    llm_candidate_text.get("grounded_evidence_note")
                    or source_grounding_payload.get("headline"),
                    180,
                ),
            },
            "source_grounding": source_grounding_payload,
            "source_grounded_evidence": source_grounded_evidence,
            "rag_supported_claims": rag_supported_claims,
            "positive_evidence": item.get("positive_evidence", [])[:5],
            "negative_evidence": item.get("negative_evidence", [])[:5],
            "caution": {
                "title": "주의할 점",
                "body": self._truncate_text(
                    llm_candidate_text.get("caution_note"),
                    180,
                ),
            },
            "risk": self._build_risk_payload(item),
            "snapshots": {
                "market_analysis": item.get("market_analysis_snapshot", {}),
                "news_signal": item.get("news_signal_snapshot", {}),
                "risk": item.get("risk_snapshot", {}),
                "rag_evidence": item.get("rag_evidence_snapshot", {}),
            },
            "data_quality": item.get("data_quality", {}),
        }

    # ============================================================
    # Source-grounded evidence
    # ============================================================

    def _build_source_grounding_summary(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        rag_summary = evidence.get("rag_evidence_summary", {}) or {}
        source_grounded_evidence = evidence.get("source_grounded_evidence", []) or []
        top10 = evidence.get("top10_evidence", []) or []

        relevance_counts = Counter()

        for ev in source_grounded_evidence:
            if isinstance(ev, dict) and ev.get("evidence_relevance"):
                relevance_counts[ev.get("evidence_relevance")] += 1

        candidate_coverage = []

        for item in top10:
            if not isinstance(item, dict):
                continue

            sg = item.get("source_grounded_evidence", []) or []

            candidate_coverage.append(
                {
                    "rank": item.get("rank"),
                    "ticker": item.get("ticker"),
                    "company_name": item.get("company_name"),
                    "source_grounded_evidence_count": len(sg),
                    "status": "supported" if sg else "no_verified_rag_evidence",
                }
            )

        return {
            "rag_enabled": bool(
                rag_summary.get("rag_enabled")
                or source_grounded_evidence
            ),
            "rag_method": rag_summary.get("rag_method"),
            "retrieval_query_count": rag_summary.get("retrieval_query_count"),
            "retrieved_evidence_count": rag_summary.get("retrieved_evidence_count"),
            "source_grounded_evidence_count": len(source_grounded_evidence),
            "candidate_coverage_count": sum(
                1
                for item in candidate_coverage
                if item.get("source_grounded_evidence_count", 0) > 0
            ),
            "candidate_count": len(candidate_coverage),
            "evidence_relevance_counts": dict(relevance_counts),
            "evidence_relevance_labels": {
                key: self.RELEVANCE_LABELS.get(key, key)
                for key in relevance_counts.keys()
            },
            "candidate_coverage": candidate_coverage,
            "raw_rag_evidence_summary": rag_summary,
        }

    def _build_source_grounded_evidence_items(
        self,
        raw_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        if not isinstance(raw_evidence, list):
            return result

        for idx, ev in enumerate(raw_evidence[: self.max_source_grounded_evidence_per_candidate]):
            if not isinstance(ev, dict):
                continue

            relevance = ev.get("evidence_relevance")

            result.append(
                {
                    "order": idx + 1,
                    "support_type": ev.get("support_type") or "rag_verified_news",
                    "support_type_label": "뉴스 근거",
                    "ticker": ev.get("ticker"),
                    "company_name": ev.get("company_name"),
                    "claim": ev.get("claim"),
                    "evidence_relevance": relevance,
                    "evidence_relevance_label": self.RELEVANCE_LABELS.get(
                        relevance,
                        relevance or "확인 필요",
                    ),
                    "directness_score": self._round_or_none(ev.get("directness_score"), 3),
                    "final_score": self._round_or_none(ev.get("final_score"), 3),
                    "retrieval_score": self._round_or_none(ev.get("retrieval_score"), 3),
                    "event_hits": ev.get("event_hits", []) or [],
                    "verification_reasons": ev.get("verification_reasons", []) or [],
                    "source": {
                        "title": ev.get("title"),
                        "press": ev.get("press"),
                        "published_at": ev.get("published_at"),
                        "url": ev.get("url"),
                        "source_path": ev.get("source_path"),
                        "doc_id": ev.get("doc_id"),
                    },
                    "summary": self._truncate_text(
                        self._first_non_empty(
                            ev.get("summary_optional"),
                            ev.get("claim"),
                            ev.get("title"),
                            default="뉴스 근거가 확인되었습니다.",
                        ),
                        180,
                    ),
                }
            )

        return result

    def _build_rag_supported_claim_items(
        self,
        raw_claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        if not isinstance(raw_claims, list):
            return result

        for idx, claim in enumerate(raw_claims[: self.max_source_grounded_evidence_per_candidate]):
            if not isinstance(claim, dict):
                continue

            relevance = claim.get("evidence_relevance")

            result.append(
                {
                    "order": idx + 1,
                    "ticker": claim.get("ticker"),
                    "company_name": claim.get("company_name"),
                    "claim": claim.get("claim"),
                    "support_type": claim.get("support_type"),
                    "evidence_relevance": relevance,
                    "evidence_relevance_label": self.RELEVANCE_LABELS.get(
                        relevance,
                        relevance or "확인 필요",
                    ),
                    "source_title": claim.get("source_title"),
                    "source_press": claim.get("source_press"),
                    "source_url": claim.get("source_url"),
                    "directness_score": self._round_or_none(claim.get("directness_score"), 3),
                }
            )

        return result

    def _build_candidate_source_grounding_payload(
        self,
        item: Dict[str, Any],
        source_grounded_evidence: List[Dict[str, Any]],
        rag_supported_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence_count = len(source_grounded_evidence)
        relevance_counts = Counter(
            ev.get("evidence_relevance")
            for ev in source_grounded_evidence
            if ev.get("evidence_relevance")
        )

        if evidence_count > 0:
            first = source_grounded_evidence[0]
            headline = self._first_non_empty(
                first.get("claim"),
                self._safe_get(first, "source.title"),
                default="검증된 뉴스 근거가 확인되었습니다.",
            )
        else:
            headline = "검증된 뉴스 근거는 별도로 확인되지 않았습니다."

        return {
            "status": "supported" if evidence_count > 0 else "no_verified_rag_evidence",
            "evidence_count": evidence_count,
            "claim_count": len(rag_supported_claims),
            "headline": self._truncate_text(headline, 180),
            "relevance_counts": dict(relevance_counts),
            "relevance_labels": {
                key: self.RELEVANCE_LABELS.get(key, key)
                for key in relevance_counts.keys()
            },
            "display_badge": (
                "근거 확인"
                if evidence_count > 0
                else "근거 제한"
            ),
            "frontend_hint": {
                "show_source_grounding_box": evidence_count > 0,
                "show_source_links": evidence_count > 0,
                "preferred_section_title": "확인된 뉴스 근거",
            },
        }

    # ============================================================
    # UI payload builders
    # ============================================================

    def _build_chart_payload(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        price_series = chart_data.get("recent_price_series", []) or []
        volume_series = chart_data.get("recent_volume_series", []) or []
        moving_average_series = chart_data.get("moving_average_series_optional", []) or []

        return {
            "type": "price_volume_ma",
            "title": "주가 흐름 차트",
            "period": chart_data.get("period", {}),
            "series": {
                "price": [
                    {
                        "date": row.get("date"),
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "volume": row.get("volume"),
                        "daily_return": row.get("daily_return"),
                        "ma20_close": row.get("ma20_close"),
                        "ma60_close": row.get("ma60_close"),
                        "rsi_14": row.get("rsi_14"),
                        "market_regime_20": row.get("market_regime_20"),
                        "market_regime_60": row.get("market_regime_60"),
                    }
                    for row in price_series
                    if isinstance(row, dict)
                ],
                "volume": [
                    {
                        "date": row.get("date"),
                        "volume": row.get("volume"),
                        "volume_zscore_20d": row.get("volume_zscore_20d"),
                        "volume_zscore_60d": row.get("volume_zscore_60d"),
                    }
                    for row in volume_series
                    if isinstance(row, dict)
                ],
                "moving_average": [
                    {
                        "date": row.get("date"),
                        "close": row.get("close"),
                        "ma20_close": row.get("ma20_close"),
                        "ma60_close": row.get("ma60_close"),
                        "ma120_close": row.get("ma120_close"),
                    }
                    for row in moving_average_series
                    if isinstance(row, dict)
                ],
            },
            "summary": chart_data.get("chart_summary", {}),
            "frontend_hint": {
                "x_axis": "date",
                "primary_y_axis": "close",
                "secondary_y_axis": "volume",
                "recommended_lines": ["close", "ma20_close", "ma60_close"],
            },
        }

    def _build_actual_data_table(self, actual_data: Dict[str, Any]) -> Dict[str, Any]:
        market_cap = actual_data.get("market_cap")
        per = actual_data.get("per")

        return {
            "title": "실제 데이터",
            "latest_date": actual_data.get("latest_date"),
            "raw": copy.deepcopy(actual_data),
            "items": [
                {
                    "key": "latest_close",
                    "label": "종가",
                    "value": self._format_krw(actual_data.get("latest_close")),
                    "raw_value": actual_data.get("latest_close"),
                },
                {
                    "key": "daily_return",
                    "label": "일간 등락률",
                    "value": self._format_percent_ratio(
                        actual_data.get("daily_return"),
                        signed=True,
                    ),
                    "raw_value": actual_data.get("daily_return"),
                },
                {
                    "key": "trading_value",
                    "label": "거래대금",
                    "value": self._format_krw(actual_data.get("trading_value")),
                    "raw_value": actual_data.get("trading_value"),
                },
                {
                    "key": "volume",
                    "label": "거래량",
                    "value": self._format_number(actual_data.get("volume")),
                    "raw_value": actual_data.get("volume"),
                },
                {
                    "key": "foreign_net_flow_ratio",
                    "label": "외국인 순매수 비율",
                    "value": self._format_percent_ratio(
                        actual_data.get("foreign_net_flow_ratio")
                    ),
                    "raw_value": actual_data.get("foreign_net_flow_ratio"),
                },
                {
                    "key": "foreign_ownership_level",
                    "label": "외국인 보유 비중",
                    "value": self._format_percent_point(
                        actual_data.get("foreign_ownership_level")
                    ),
                    "raw_value": actual_data.get("foreign_ownership_level"),
                },
                {
                    "key": "roe",
                    "label": "ROE",
                    "value": self._format_percent_ratio(actual_data.get("roe")),
                    "raw_value": actual_data.get("roe"),
                },
                {
                    "key": "operating_income_yoy",
                    "label": "영업이익 YoY",
                    "value": self._format_percent_ratio(
                        actual_data.get("operating_income_yoy")
                    ),
                    "raw_value": actual_data.get("operating_income_yoy"),
                },
                {
                    "key": "risk_level",
                    "label": "리스크 등급",
                    "value": self._risk_level_ko(actual_data.get("risk_level")),
                    "raw_value": actual_data.get("risk_level"),
                },
                {
                    "key": "market_cap",
                    "label": "시가총액",
                    "value": self._format_krw(market_cap) if market_cap is not None else "-",
                    "raw_value": market_cap,
                    "is_missing": market_cap is None,
                },
                {
                    "key": "per",
                    "label": "PER",
                    "value": self._format_multiple(per) if per is not None else "-",
                    "raw_value": per,
                    "is_missing": per is None,
                },
            ],
        }

    def _build_component_score_items(
        self,
        component_scores: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "key": key,
                "label": self.COMPONENT_LABELS.get(key, key),
                "score": self._round_or_none(value, 3),
                "display_value": self._format_score(value, 1),
            }
            for key, value in component_scores.items()
        ]

    def _build_top_component_highlights(
        self,
        item: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        top_components = self._safe_get(
            item,
            "ranking_reason.score_basis.top_component_scores",
            default=[],
        )

        result = []

        if isinstance(top_components, list):
            for c in top_components:
                if not isinstance(c, dict):
                    continue

                key = c.get("name")

                result.append(
                    {
                        "key": key,
                        "label": c.get("label") or self.COMPONENT_LABELS.get(key, key),
                        "score": self._round_or_none(c.get("score"), 3),
                        "display_value": self._format_score(c.get("score"), 1),
                    }
                )

        if result:
            return result

        component_scores = item.get("component_scores", {}) or {}

        sorted_items = sorted(
            component_scores.items(),
            key=lambda kv: kv[1] if self._is_number(kv[1]) else -999,
            reverse=True,
        )

        for key, value in sorted_items[:2]:
            result.append(
                {
                    "key": key,
                    "label": self.COMPONENT_LABELS.get(key, key),
                    "score": self._round_or_none(value, 3),
                    "display_value": self._format_score(value, 1),
                }
            )

        return result

    def _build_news_items(
        self,
        news_list: List[Dict[str, Any]],
        llm_news_summaries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        result = []
        llm_news_summaries = llm_news_summaries or []

        if not isinstance(news_list, list):
            news_list = []

        for idx, news in enumerate(news_list[: self.max_news_per_candidate]):
            if not isinstance(news, dict):
                continue

            llm_summary = (
                llm_news_summaries[idx]
                if idx < len(llm_news_summaries)
                else None
            )

            summary = self._first_non_empty(
                llm_summary,
                news.get("summary_optional"),
                news.get("title"),
                default="관련 뉴스가 확인되었습니다.",
            )

            relevance = news.get("evidence_relevance")

            result.append(
                {
                    "order": idx + 1,
                    "title": news.get("title"),
                    "source": news.get("source"),
                    "published_at": news.get("published_at"),
                    "url": news.get("url"),
                    "summary": self._truncate_text(summary, 120),
                    "article_score": news.get("article_score"),
                    "rule_score_0_1": news.get("rule_score_0_1"),

                    # RAG metadata
                    "source_type": news.get("source_type") or "internal_news",
                    "support_type": news.get("support_type"),
                    "is_source_grounded": news.get("support_type") == "rag_verified_news",
                    "evidence_relevance": relevance,
                    "evidence_relevance_label": self.RELEVANCE_LABELS.get(
                        relevance,
                        relevance,
                    ),
                    "directness_score": self._round_or_none(news.get("directness_score"), 3),
                    "verification_reasons": news.get("verification_reasons"),
                }
            )

        return result

    def _build_risk_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
        actual_data = item.get("actual_data", {}) or {}
        risk_snapshot = item.get("risk_snapshot", {}) or {}
        risk_scores = risk_snapshot.get("risk_scores", {}) or {}

        level = actual_data.get("risk_level") or risk_scores.get("risk_level")

        return {
            "level": self._normalize_risk_level(level),
            "level_label": self._risk_level_ko(level),
            "overall_risk_score": self._round_or_none(
                actual_data.get("overall_risk_score")
                or risk_scores.get("overall_risk_score"),
                3,
            ),
            "dominant_risk_factors": risk_snapshot.get("dominant_risk_factors", []),
            "risk_scores": risk_scores,
            "notes": item.get("risk_notes", [])[:5],
            "evidence": risk_snapshot.get("evidence", [])[:5],
        }

    def _build_glossary(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_glossary = (
            evidence.get("glossary")
            or evidence.get("terms_glossary")
            or evidence.get("glossary_terms")
            or []
        )
        result = []

        if isinstance(raw_glossary, list):
            for item in raw_glossary:
                if not isinstance(item, dict):
                    continue

                term = item.get("term")
                if not term:
                    continue

                result.append(
                    {
                        "term": term,
                        "description": self._first_non_empty(
                            item.get("plain_korean_definition"),
                            item.get("description"),
                            item.get("definition"),
                            default="리포트 이해를 돕기 위한 용어입니다.",
                        ),
                        "why_it_matters": self._first_non_empty(
                            item.get("why_it_matters"),
                            default="해당 지표를 함께 보면 종목을 더 입체적으로 이해할 수 있습니다.",
                        ),
                    }
                )

        default_terms = [
            {
                "term": "외국인 순매수",
                "description": "외국인 투자자가 판 금액보다 산 금액이 더 많은 상태를 의미합니다.",
                "why_it_matters": "수급 관심이 어느 종목에 몰리는지 확인할 때 참고할 수 있습니다.",
            },
            {
                "term": "PER",
                "description": "주가가 기업의 이익에 비해 얼마나 높거나 낮게 평가되어 있는지 보여주는 지표입니다.",
                "why_it_matters": "기업의 가격 부담을 간단히 비교할 때 사용됩니다.",
            },
            {
                "term": "ROE",
                "description": "자기자본 대비 순이익을 얼마나 냈는지 보여주는 수익성 지표입니다.",
                "why_it_matters": "기업이 가진 자본을 얼마나 효율적으로 활용하는지 볼 때 사용합니다.",
            },
            {
                "term": "리스크 점수",
                "description": "변동성, 낙폭, 재무 부담, 뉴스 이벤트 등을 종합한 위험도 점수입니다.",
                "why_it_matters": "점수가 높을수록 가격 흔들림이나 불확실성에 더 주의해야 합니다.",
            },
            {
                "term": "뉴스 근거",
                "description": "수집된 뉴스 데이터 안에서 해당 종목과 관련성이 확인된 기사입니다.",
                "why_it_matters": "종목 설명이 단순 추정이 아니라 어떤 기사 흐름에 기반하는지 확인할 때 참고합니다.",
            },
        ]

        existing = {x["term"] for x in result}

        for term in default_terms:
            if term["term"] not in existing:
                result.append(term)

        return result

    # ============================================================
    # LLM generation
    # ============================================================

    def _generate_text_pack(
        self,
        evidence: Dict[str, Any],
        ctx: RunContext,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not self.use_llm or self.llm_provider == "none":
            return None, {
                "used": False,
                "provider": self.llm_provider,
                "model": "",
                "note": "LLM disabled. Rule-based fallback text used.",
            }

        prompt = self._build_llm_prompt(evidence)

        providers = self._resolve_llm_providers()

        if not providers:
            return None, {
                "used": False,
                "provider": self.llm_provider,
                "model": "",
                "note": "No LLM API key found. Rule-based fallback text used.",
            }

        last_error = None

        for provider in providers:
            try:
                if provider == "gemini":
                    ctx.logger.info("[ReportGenerativeAgent] Gemini text generation start.")
                    raw = self._call_gemini(prompt)
                    parsed = self._parse_json_response(raw)
                    return parsed, {
                        "used": True,
                        "provider": "gemini",
                        "model": self.gemini_model_name,
                        "note": "LLM-generated report text used.",
                    }

                if provider == "openai":
                    ctx.logger.info("[ReportGenerativeAgent] OpenAI text generation start.")
                    raw = self._call_openai(prompt)
                    parsed = self._parse_json_response(raw)
                    return parsed, {
                        "used": True,
                        "provider": "openai",
                        "model": self.openai_model_name,
                        "note": "LLM-generated report text used.",
                    }

            except Exception as e:
                last_error = str(e)
                ctx.logger.warning(
                    f"[ReportGenerativeAgent] {provider} failed. Error: {last_error}"
                )

        return None, {
            "used": False,
            "provider": self.llm_provider,
            "model": "",
            "note": "All LLM providers failed. Rule-based fallback text used.",
            "error": last_error,
        }

    def _resolve_llm_providers(self) -> List[str]:
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if self.llm_provider == "gemini":
            return ["gemini"] if google_key else []

        if self.llm_provider == "openai":
            return ["openai"] if openai_key else []

        providers = []

        if google_key:
            providers.append("gemini")

        if openai_key:
            providers.append("openai")

        return providers

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 없습니다.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=self.gemini_model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        return response.text or ""

    def _call_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY가 없습니다.")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=self.openai_model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a Korean financial report generation agent. "
                        "Return only valid JSON. Do not invent facts. "
                        "Do not change rankings, scores, numbers, news titles, or URLs."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text={
                "format": {
                    "type": "json_object",
                }
            },
            temperature=0.2,
        )

        return response.output_text or ""

    def _build_llm_prompt(self, evidence: Dict[str, Any]) -> str:
        market_context = evidence.get("market_context", {}) or {}
        top10 = evidence.get("top10_evidence", []) or []
        source_grounding_summary = self._build_source_grounding_summary(evidence)

        compact_candidates = []

        for item in top10:
            ranking_reason = item.get("ranking_reason", {}) or {}
            actual_data = item.get("actual_data", {}) or {}
            chart_summary = self._safe_get(item, "chart_data.chart_summary", {})
            score_basis = self._safe_get(
                item,
                "ranking_reason.score_basis",
                {},
            )

            source_grounded_evidence = []

            for ev in item.get("source_grounded_evidence", [])[:3]:
                if not isinstance(ev, dict):
                    continue

                source_grounded_evidence.append(
                    {
                        "claim": ev.get("claim"),
                        "evidence_relevance": ev.get("evidence_relevance"),
                        "title": ev.get("title"),
                        "press": ev.get("press"),
                        "published_at": ev.get("published_at"),
                        "event_hits": ev.get("event_hits", [])[:5],
                        "directness_score": ev.get("directness_score"),
                    }
                )

            compact_candidates.append(
                {
                    "rank": item.get("rank"),
                    "ticker": item.get("ticker"),
                    "company_name": item.get("company_name"),
                    "score": item.get("market_flow_candidate_score"),
                    "risk_level": actual_data.get("risk_level"),
                    "main_reason": ranking_reason.get("main_reason"),
                    "supporting_reasons": ranking_reason.get("supporting_reasons", [])[:6],
                    "beginner_explanation": ranking_reason.get("beginner_explanation"),
                    "caution_note": ranking_reason.get("caution_note"),
                    "positive_evidence": item.get("positive_evidence", [])[:5],
                    "negative_evidence": item.get("negative_evidence", [])[:3],
                    "risk_notes": item.get("risk_notes", [])[:4],
                    "top_component_scores": score_basis.get("top_component_scores", []),
                    "chart_summary": {
                        "latest_close": chart_summary.get("latest_close"),
                        "daily_return": actual_data.get("daily_return"),
                        "close_change_30d": chart_summary.get("close_change_30d"),
                        "rsi_14": chart_summary.get("rsi_14"),
                        "market_regime_20": chart_summary.get("market_regime_20"),
                        "market_regime_60": chart_summary.get("market_regime_60"),
                    },
                    "top_news_3": [
                        {
                            "title": n.get("title"),
                            "summary_optional": n.get("summary_optional"),
                            "published_at": n.get("published_at"),
                            "source_type": n.get("source_type"),
                            "evidence_relevance": n.get("evidence_relevance"),
                        }
                        for n in item.get("top_news_3", [])[:3]
                        if isinstance(n, dict)
                    ],
                    "source_grounded_evidence": source_grounded_evidence,
                    "rag_evidence_snapshot": item.get("rag_evidence_snapshot", {}),
                }
            )

        compact_input = {
            "report_context": evidence.get("report_context", {}),
            "market_context": {
                "market_one_line_summary": market_context.get("market_one_line_summary"),
                "market_brief": market_context.get("market_brief"),
                "market_keywords": market_context.get("market_keywords", []),
                "core_market_drivers": market_context.get("core_market_drivers", []),
                "market_phase": market_context.get("market_phase"),
                "market_tone": market_context.get("market_tone"),
                "market_risk_notes": market_context.get("market_risk_notes", []),
            },
            "source_grounding_summary": {
                "rag_enabled": source_grounding_summary.get("rag_enabled"),
                "source_grounded_evidence_count": source_grounding_summary.get(
                    "source_grounded_evidence_count"
                ),
                "evidence_relevance_counts": source_grounding_summary.get(
                    "evidence_relevance_counts",
                    {},
                ),
            },
            "candidates": compact_candidates,
        }

        output_schema = {
            "market_summary": {
                "headline": "오늘 시장을 한 문장으로 요약",
                "body": "2문장 이내의 시장 요약",
                "tone_label": "긍정/중립/혼재/부정 중 하나",
            },
            "candidates": {
                "티커": {
                    "quick_summary": "종목 카드 상단 1문장 요약",
                    "entry_reason": "순위 진입 이유 1~2문장",
                    "caution_note": "주의사항 1문장",
                    "grounded_evidence_note": "뉴스 근거가 있으면 이를 자연스럽게 요약한 1문장",
                    "news_summaries": [
                        "뉴스 1 카드용 요약",
                        "뉴스 2 카드용 요약",
                        "뉴스 3 카드용 요약",
                    ],
                }
            },
        }

        return f"""
아래 데이터를 바탕으로 프론트 카드형 리포트에 들어갈 자연어 문장만 생성해줘.
반드시 JSON만 반환해.

중요 규칙:
1. 입력에 없는 사실을 만들지 마.
2. 순위, 점수, 수치, 뉴스 제목, 뉴스 URL은 변경하지 마.
3. 매수 추천, 수익 보장, 목표주가, 단기 급등 예측처럼 쓰지 마.
4. 후보 종목은 '관심 있게 확인할 만한 종목'으로만 표현해.
5. 리스크는 반드시 함께 언급해.
6. 초보 투자자가 이해하기 쉽게 써.
7. 후보별 candidates의 key는 반드시 ticker 값으로 사용해.
8. source_grounded_evidence가 있는 후보는 entry_reason 또는 grounded_evidence_note에 해당 뉴스 근거를 자연스럽게 반영해.
9. 내부 용어인 RAG, directness_score, retrieval_score는 사용자용 문장에 직접 쓰지 마. 대신 '확인된 뉴스 근거', '관련 기사', '사업 관계 뉴스'처럼 자연스럽게 표현해.
10. source_grounded_evidence가 없는 후보에 대해 근거가 확인됐다고 말하지 마.

반환 스키마:
{json.dumps(output_schema, ensure_ascii=False, indent=2)}

입력 데이터:
{json.dumps(compact_input, ensure_ascii=False, indent=2)}
""".strip()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            return json.loads(candidate)

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)

        raise ValueError("Failed to parse LLM response as JSON.")

    # ============================================================
    # Fallback text
    # ============================================================

    def _fallback_market_text(self, market_context: Dict[str, Any]) -> Dict[str, str]:
        summary_box = market_context.get("report_summary_box", {}) or {}

        headline = self._first_non_empty(
            market_context.get("market_one_line_summary"),
            summary_box.get("body"),
            default="오늘 시장은 종목별 차별화가 중요한 흐름입니다.",
        )

        phase = market_context.get("market_phase", "혼재 국면")
        keywords = market_context.get("market_keywords", []) or []

        if keywords:
            body = f"{phase} 속에서 {', '.join(keywords[:3])} 흐름을 함께 확인할 필요가 있습니다."
        else:
            body = self._first_non_empty(
                market_context.get("market_brief"),
                summary_box.get("body"),
                default=headline,
            )

        return {
            "headline": self._truncate_text(headline, 100),
            "body": self._truncate_text(body, 180),
            "tone_label": self._tone_label_ko(market_context.get("market_tone")),
        }

    def _fallback_candidate_text(self, item: Dict[str, Any]) -> Dict[str, Any]:
        company_name = item.get("company_name") or item.get("ticker")
        ranking_reason = item.get("ranking_reason", {}) or {}
        actual_data = item.get("actual_data", {}) or {}
        score_basis = self._safe_get(item, "ranking_reason.score_basis", {})
        top_components = score_basis.get("top_component_scores", []) or []
        source_grounded_evidence = item.get("source_grounded_evidence", []) or []

        if top_components:
            labels = [
                c.get("label")
                for c in top_components
                if isinstance(c, dict) and c.get("label")
            ]
            factor_text = "와 ".join(labels[:2]) if labels else "주요 데이터"
            quick_summary = f"{company_name}은 {factor_text} 신호가 상대적으로 높게 나타난 후보입니다."
        else:
            quick_summary = f"{company_name}은 현재 시장 흐름에서 관심 있게 확인할 만한 후보입니다."

        grounded_note = ""

        if source_grounded_evidence:
            first_ev = source_grounded_evidence[0]
            grounded_note = self._first_non_empty(
                first_ev.get("claim"),
                first_ev.get("title"),
                default="관련 뉴스 근거가 확인되었습니다.",
            )

        entry_reason = self._first_non_empty(
            ranking_reason.get("beginner_explanation"),
            ranking_reason.get("main_reason"),
            ranking_reason.get("original_ranking_reason_short"),
            grounded_note,
            default=quick_summary,
        )

        if grounded_note and grounded_note not in entry_reason:
            entry_reason = f"{entry_reason} 또한 {grounded_note}"

        caution_note = self._first_non_empty(
            ranking_reason.get("caution_note"),
            *(item.get("risk_notes", [])[:2]),
            default=(
                f"리스크 등급은 {self._risk_level_ko(actual_data.get('risk_level'))}이며, "
                "가격 변동 가능성을 함께 확인해야 합니다."
            ),
        )

        news_summaries = []

        for news in item.get("top_news_3", [])[: self.max_news_per_candidate]:
            if not isinstance(news, dict):
                continue

            prefix = ""

            if news.get("support_type") == "rag_verified_news":
                prefix = "확인된 뉴스 근거: "

            news_summaries.append(
                self._truncate_text(
                    prefix
                    + self._first_non_empty(
                        news.get("summary_optional"),
                        news.get("title"),
                        default="관련 뉴스가 확인되었습니다.",
                    ),
                    120,
                )
            )

        return {
            "quick_summary": self._truncate_text(quick_summary, 120),
            "entry_reason": self._truncate_text(entry_reason, 240),
            "caution_note": self._truncate_text(caution_note, 180),
            "grounded_evidence_note": self._truncate_text(grounded_note, 180),
            "news_summaries": news_summaries,
        }

    # ============================================================
    # Manifest / validation
    # ============================================================

    def _build_manifest(
        self,
        evidence_path: Path,
        evidence_manifest_path: Optional[Path],
        report_output_contract_path: Optional[Path],
        result_path: Path,
        report: Dict[str, Any],
        is_valid: bool,
        warnings: List[str],
        llm_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates = self._safe_get(report, "candidate_section.candidates", default=[])
        source_grounding_summary = report.get("source_grounding_summary", {}) or {}

        return {
            "agent": self.__class__.__name__,
            "stage": self.stage,
            "version": self.version,
            "created_at_utc": self._now_utc(),
            "input_files": {
                "report_evidence_result": str(evidence_path),
                "evidence_manifest": (
                    str(evidence_manifest_path) if evidence_manifest_path else None
                ),
                "report_output_contract": (
                    str(report_output_contract_path) if report_output_contract_path else None
                ),
            },
            "output_files": {
                "report_output": str(result_path),
            },
            "llm_status": llm_status,
            "summary": {
                "as_of_date": self._safe_get(report, "report_meta.as_of_date"),
                "report_title": self._safe_get(report, "report_meta.title"),
                "n_candidates": len(candidates),
                "is_valid": is_valid,
                "warnings": warnings,
                "rag_enabled": source_grounding_summary.get("rag_enabled"),
                "source_grounded_evidence_count": source_grounding_summary.get(
                    "source_grounded_evidence_count"
                ),
                "candidate_coverage_count": source_grounding_summary.get(
                    "candidate_coverage_count"
                ),
                "candidates": [
                    {
                        "rank": c.get("rank"),
                        "ticker": c.get("ticker"),
                        "company_name": c.get("company_name"),
                        "score": c.get("score"),
                        "risk_level": self._safe_get(c, "risk.level"),
                        "news_count": len(c.get("top_news", [])),
                        "source_grounded_evidence_count": self._safe_get(
                            c,
                            "source_grounding.evidence_count",
                            default=0,
                        ),
                        "chart_points": len(
                            self._safe_get(c, "chart.series.price", default=[])
                        ),
                    }
                    for c in candidates
                    if isinstance(c, dict)
                ],
            },
        }

    def _validate_report_output(self, report: Dict[str, Any]) -> Tuple[bool, List[str]]:
        warnings = []
        candidates = self._safe_get(report, "candidate_section.candidates", default=[])

        if not candidates:
            warnings.append("candidate_section.candidates가 비어 있습니다.")

        ranks = [c.get("rank") for c in candidates if isinstance(c, dict)]

        if len(ranks) != len(set(ranks)):
            warnings.append("후보 rank가 중복됩니다.")

        for c in candidates:
            if not isinstance(c, dict):
                continue

            ticker = c.get("ticker")

            if not ticker:
                warnings.append(f"rank={c.get('rank')} 후보에 ticker가 없습니다.")

            if not c.get("company_name"):
                warnings.append(f"rank={c.get('rank')} 후보에 company_name이 없습니다.")

            if not self._safe_get(c, "chart.series.price", default=[]):
                warnings.append(f"{ticker} 차트 price series가 비어 있습니다.")

            if len(c.get("top_news", [])) < self.max_news_per_candidate:
                warnings.append(f"{ticker} 뉴스가 {self.max_news_per_candidate}개 미만입니다.")

            if not self._safe_get(c, "actual_data.items", default=[]):
                warnings.append(f"{ticker} actual_data.items가 비어 있습니다.")

            source_grounding = c.get("source_grounding", {}) or {}
            sg_count = source_grounding.get("evidence_count", 0)

            if sg_count > 0 and not c.get("source_grounded_evidence"):
                warnings.append(f"{ticker} source_grounding count는 있으나 source_grounded_evidence가 비어 있습니다.")

        return len(warnings) == 0, warnings

    # ============================================================
    # File helpers
    # ============================================================

    def _find_single_file(
        self,
        run_dir: Path,
        pattern: str,
        ctx: RunContext,
    ) -> Optional[Path]:
        matches = sorted(run_dir.rglob(pattern))

        if not matches:
            ctx.logger.info(f"[ReportGenerativeAgent] No file matched: {pattern}")
            return None

        if len(matches) > 1:
            ctx.logger.warning(
                f"[ReportGenerativeAgent] Multiple files matched: {pattern}. "
                f"Using first: {matches[0]}"
            )

        return matches[0]

    def _safe_read_json(
        self,
        path: Optional[Path],
        ctx: RunContext,
    ) -> Optional[Dict[str, Any]]:
        if path is None:
            return None

        if not path.exists():
            ctx.logger.warning(f"[ReportGenerativeAgent] Missing JSON: {path}")
            return None

        try:
            with open(path, "r", encoding=self.encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            ctx.logger.warning(
                f"[ReportGenerativeAgent] Failed to read JSON: {path} ({e})"
            )
            return None

    @staticmethod
    def _save_json(obj: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ============================================================
    # Format helpers
    # ============================================================

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _round_or_none(self, value: Any, ndigits: int = 3) -> Optional[float]:
        if self._is_number(value):
            return round(float(value), ndigits)

        try:
            if value is not None and str(value).strip() != "":
                return round(float(value), ndigits)
        except Exception:
            pass

        return None

    @staticmethod
    def _safe_get(data: Any, path: str, default: Any = None) -> Any:
        cur = data

        for key in path.split("."):
            if not isinstance(cur, dict):
                return default

            cur = cur.get(key)

            if cur is None:
                return default

        return cur

    @staticmethod
    def _first_non_empty(*values: Any, default: str = "") -> str:
        for value in values:
            if value is None:
                continue

            s = str(value).strip()

            if s:
                return s

        return default

    @staticmethod
    def _truncate_text(text: Any, max_len: int = 160) -> str:
        if text is None:
            return ""

        s = str(text).strip()
        s = re.sub(r"\s+", " ", s)

        if len(s) <= max_len:
            return s

        return s[: max_len - 1].rstrip() + "…"

    def _format_krw(self, value: Any, ndigits: int = 2) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        v = float(value)
        av = abs(v)

        if av >= 1_0000_0000_0000:
            return f"{v / 1_0000_0000_0000:.{ndigits}f}조 원"

        if av >= 1_0000_0000:
            return f"{v / 1_0000_0000:.{ndigits}f}억 원"

        if av >= 1_0000:
            return f"{v / 1_0000:.{ndigits}f}만 원"

        return f"{v:,.0f}원"

    def _format_number(self, value: Any, ndigits: int = 0) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        if ndigits == 0:
            return f"{float(value):,.0f}"

        return f"{float(value):,.{ndigits}f}"

    def _format_percent_ratio(
        self,
        value: Any,
        ndigits: int = 2,
        signed: bool = False,
    ) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        pct = float(value) * 100
        sign = "+" if signed and pct > 0 else ""

        return f"{sign}{pct:.{ndigits}f}%"

    def _format_percent_point(self, value: Any, ndigits: int = 2) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        return f"{float(value):.{ndigits}f}%"

    def _format_multiple(self, value: Any, ndigits: int = 2) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        return f"{float(value):.{ndigits}f}배"

    def _format_score(self, value: Any, ndigits: int = 1) -> str:
        if not self._is_number(value):
            try:
                value = float(value)
            except Exception:
                return "-"

        return f"{float(value):.{ndigits}f}점"

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        s = str(value or "").strip().lower()

        if s in {"low", "medium", "high", "critical"}:
            return s

        if s in {"mid", "middle"}:
            return "medium"

        return "unknown"

    def _risk_level_ko(self, value: Any) -> str:
        mapping = {
            "low": "낮음",
            "medium": "중간",
            "high": "높음",
            "critical": "매우 높음",
            "unknown": "확인 필요",
        }

        return mapping.get(self._normalize_risk_level(value), "확인 필요")

    @staticmethod
    def _tone_label_ko(value: Any) -> str:
        mapping = {
            "positive": "긍정",
            "neutral": "중립",
            "mixed": "혼재",
            "negative": "부정",
        }

        return mapping.get(str(value or "").lower(), "혼재")

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