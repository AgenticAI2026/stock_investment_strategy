from __future__ import annotations

import argparse
import inspect
import json
import shutil
import copy
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================
# 0. Import Agents
# ============================================================

try:
    from agents.ingest.agent import DataIngestionAgent
    from agents.feature_table.agent import FeatureExtractionAgent
    from agents.prep_reco.agent import PreprocessingRecommenderAgent
    from agents.model_match_v1.agent import ModelDataMatcherAgent
    from agents.prep_apply.agent import PreprocessingImplementorAgent

    from agents.market_analysis.agent import MarketAnalysisAgent
    from agents.news_invest.agent import NewsInvestigationAgent
    from agents.risk_score.agent import RiskScoreAgent

    from agents.market_flow.agent import MarketFlowAgent
    from agents.report_target_planner.agent import ReportTargetPlannerAgent
    from agents.candidate_score.agent import CandidateScoringAgent
    from agents.report_evidence_builder.agent import ReportEvidenceBuilderAgent
    from agents.report_gen.agent import ReportGenerativeAgent

except ImportError as e:
    print("[IMPORT ERROR]")
    print("프로젝트 루트에서 실행하고 있는지 확인해줘.")
    print("예: python run_contribution2_experiment.py")
    print("현재 에러:", e)
    raise


# ============================================================
# 1. Experiment Config
# ============================================================

EXPERIMENT_CONFIGS = {
    "baseline": {
        "label": "Baseline",
        "use_analysis_layer": False,
        "use_rag": False,
        "market_flow_mode": "data_only_summary",
        "description": "No analysis layer, no RAG.",
    },
    "ours_a": {
        "label": "Ours A",
        "use_analysis_layer": True,
        "use_rag": False,
        "market_flow_mode": "agent_analysis_summary",
        "description": "Multi-agent analysis layer without RAG.",
    },
    "ours_b": {
        "label": "Ours B",
        "use_analysis_layer": True,
        "use_rag": True,
        "market_flow_mode": "rag_supported_analysis_summary",
        "description": "Multi-agent analysis layer with local news-corpus RAG evidence verification.",
    },
}


STALE_EXPERIMENT_STAGE_DIRS = [
    "market_analysis",
    "news_invest",
    "risk_score",
    "market_flow",
    "report_target_planner",
    "candidate_scoring",
    "candidate_score",
    "report_evidence_builder",
    "report_evidence",
    "report_gen",
    "report_generation",
]


# ============================================================
# 2. Context / Logger / Artifact Path Compatibility
# ============================================================

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


class SimpleLogger:
    def _format(self, msg: Any, *args: Any) -> str:
        msg = str(msg)

        if args:
            try:
                msg = msg % args
            except Exception:
                msg = " ".join([msg] + [str(a) for a in args])

        return msg

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        print(f"[INFO] {self._format(msg, *args)}")

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        print(f"[WARNING] {self._format(msg, *args)}")

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        print(f"[ERROR] {self._format(msg, *args)}")

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        print(f"[DEBUG] {self._format(msg, *args)}")

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        print(f"[EXCEPTION] {self._format(msg, *args)}")


class ArtifactPathContext(DotDict):
    """
    기존 Agent의 execute(ctx, ap) 호출 호환용.
    """

    def _artifact_run_dir(self) -> Path:
        artifact_run_dir = Path(self["artifact_run_dir"])
        artifact_run_dir.mkdir(parents=True, exist_ok=True)
        return artifact_run_dir

    def _make_dir(self, stage_name: str) -> Path:
        path = self._artifact_run_dir() / str(stage_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_dir(self, stage: str) -> Path:
        return self._make_dir(stage)

    def file(self, stage: str, filename: str) -> Path:
        path = self.stage_dir(stage) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, *parts: str) -> Path:
        path = self._artifact_run_dir().joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------------------------------------
    # Stage dirs
    # --------------------------------------------------------

    def ingest_dir(self) -> Path:
        return self._make_dir("ingest")

    def feature_table_dir(self) -> Path:
        return self._make_dir("feature_table")

    def prep_reco_dir(self) -> Path:
        return self._make_dir("prep_reco")

    def model_match_v1_dir(self) -> Path:
        return self._make_dir("model_match_v1")

    def prep_apply_dir(self) -> Path:
        return self._make_dir("prep_apply")

    def market_analysis_dir(self) -> Path:
        return self._make_dir("market_analysis")

    def news_invest_dir(self) -> Path:
        return self._make_dir("news_invest")

    def risk_score_dir(self) -> Path:
        return self._make_dir("risk_score")

    def market_flow_dir(self) -> Path:
        return self._make_dir("market_flow")

    def report_target_planner_dir(self) -> Path:
        return self._make_dir("report_target_planner")

    def candidate_score_dir(self) -> Path:
        return self._make_dir("candidate_scoring")

    def candidate_scoring_dir(self) -> Path:
        return self._make_dir("candidate_scoring")

    def report_evidence_builder_dir(self) -> Path:
        return self._make_dir("report_evidence_builder")

    def report_evidence_dir(self) -> Path:
        return self._make_dir("report_evidence")

    def report_gen_dir(self) -> Path:
        return self._make_dir("report_gen")

    def report_generation_dir(self) -> Path:
        return self._make_dir("report_generation")

    # --------------------------------------------------------
    # Ingest sub dirs
    # --------------------------------------------------------

    def finance_dir(self) -> Path:
        path = self.ingest_dir() / "finance"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def price_dir(self) -> Path:
        path = self.ingest_dir() / "price"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def news_dir(self) -> Path:
        path = self.ingest_dir() / "news"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def meta_dir(self) -> Path:
        path = self.ingest_dir() / "meta"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def user_dir(self) -> Path:
        path = self.ingest_dir() / "user"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------------------------------------
    # Specific files
    # --------------------------------------------------------

    def price_ohlcv(self) -> Path:
        return self.price_dir() / "price_ohlcv.csv"

    def price_foreign(self) -> Path:
        return self.price_dir() / "price_foreign.csv"

    def __getattr__(self, key):
        if key.endswith("_dir"):
            stage_name = key[:-4]

            def _dir_func():
                return self._make_dir(stage_name)

            return _dir_func

        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: to_dotdict(v) for k, v in obj.items()})

    if isinstance(obj, list):
        return [to_dotdict(v) for v in obj]

    return obj


# ============================================================
# 3. Basic Utilities
# ============================================================

def now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_json_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict(orient="records")
        except Exception:
            try:
                return obj.to_dict()
            except Exception:
                pass

    if hasattr(obj, "__dict__"):
        try:
            return make_json_safe(obj.__dict__)
        except Exception:
            pass

    return str(obj)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def try_init_agent(agent_class: Any, **kwargs) -> Any:
    try:
        return agent_class(**kwargs)
    except TypeError:
        return agent_class()


# ============================================================
# 4. Artifact Normalization
# ============================================================

def first_existing_file(paths):
    for path in paths:
        p = Path(path)

        if p.exists() and p.is_file():
            return p

    return None


def collect_artifact_files(artifact_run_dir: Path) -> Dict[str, str]:
    files: Dict[str, str] = {}

    for path in Path(artifact_run_dir).rglob("*"):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if "contribution2_experiment" in path.parts:
            continue

        suffix = path.suffix.lower()

        if suffix in [".csv", ".json", ".parquet", ".xlsx", ".md", ".txt"]:
            key = path.stem

            if key in files:
                key = str(path.relative_to(artifact_run_dir)).replace("\\", "/")

            files[key] = str(path)

    return files


def infer_asof_date_from_price_ohlcv(price_ohlcv_path: Path) -> Optional[str]:
    price_ohlcv_path = Path(price_ohlcv_path)

    if not price_ohlcv_path.exists():
        return None

    date_candidates = [
        "date",
        "Date",
        "DATE",
        "날짜",
        "trade_date",
        "TRD_DD",
        "기준일",
        "asof_date",
    ]

    try:
        max_date = None

        with open(price_ohlcv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                return None

            date_col = None

            for candidate in date_candidates:
                if candidate in reader.fieldnames:
                    date_col = candidate
                    break

            if date_col is None:
                return None

            for row in reader:
                raw = str(row.get(date_col, "")).strip()

                if not raw:
                    continue

                raw_norm = raw.replace(".", "-").replace("/", "-")

                if re.fullmatch(r"\d{8}", raw_norm):
                    raw_norm = f"{raw_norm[:4]}-{raw_norm[4:6]}-{raw_norm[6:8]}"

                try:
                    dt = datetime.fromisoformat(raw_norm[:10])
                except Exception:
                    continue

                if max_date is None or dt > max_date:
                    max_date = dt

        if max_date is None:
            return None

        return max_date.strftime("%Y-%m-%d")

    except Exception:
        return None


def infer_asof_date_from_run_id(artifact_run_dir: Path) -> Optional[str]:
    name = Path(artifact_run_dir).name

    match = re.search(r"(\d{8})", name)

    if not match:
        return None

    raw = match.group(1)

    try:
        dt = datetime.strptime(raw, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def build_artifact_aliases(artifact_run_dir: Path) -> Dict[str, Any]:
    artifact_run_dir = Path(artifact_run_dir)

    ingest_dir = artifact_run_dir / "ingest"
    finance_dir = ingest_dir / "finance"
    price_dir = ingest_dir / "price"
    news_dir = ingest_dir / "news"
    meta_dir = ingest_dir / "meta"
    user_dir = ingest_dir / "user"

    price_ohlcv = first_existing_file(
        [
            price_dir / "price_ohlcv.csv",
            ingest_dir / "price_ohlcv.csv",
            artifact_run_dir / "price_ohlcv.csv",
        ]
    )

    price_foreign = first_existing_file(
        [
            price_dir / "price_foreign.csv",
            ingest_dir / "price_foreign.csv",
            artifact_run_dir / "price_foreign.csv",
        ]
    )

    asof_date = None

    if price_ohlcv:
        asof_date = infer_asof_date_from_price_ohlcv(price_ohlcv)

    if asof_date is None:
        asof_date = infer_asof_date_from_run_id(artifact_run_dir)

    aliases: Dict[str, Any] = {
        "artifact_root": str(artifact_run_dir),
        "artifact_run_dir": str(artifact_run_dir),
        "run_dir": str(artifact_run_dir),

        "ingest_dir": str(ingest_dir),
        "finance_dir": str(finance_dir),
        "price_dir": str(price_dir),
        "news_dir": str(news_dir),
        "meta_dir": str(meta_dir),
        "user_dir": str(user_dir),
    }

    if price_ohlcv:
        aliases["price_ohlcv"] = str(price_ohlcv)
        aliases["price_ohlcv_path"] = str(price_ohlcv)

    if price_foreign:
        aliases["price_foreign"] = str(price_foreign)
        aliases["price_foreign_path"] = str(price_foreign)

    if asof_date:
        aliases["asof_date"] = asof_date
        aliases["base_date"] = asof_date
        aliases["target_date"] = asof_date
        aliases["trade_date"] = asof_date

    return aliases


def normalize_manifest(
    artifact_run_dir: Path,
    manifest: Optional[Dict[str, Any]],
    artifact_files: Dict[str, str],
) -> Dict[str, Any]:
    artifact_run_dir = Path(artifact_run_dir)
    manifest = manifest or {}
    aliases = build_artifact_aliases(artifact_run_dir)

    for key, value in aliases.items():
        manifest.setdefault(key, value)

    if "inputs" not in manifest or not isinstance(manifest["inputs"], dict):
        manifest["inputs"] = {}

    for key, value in aliases.items():
        manifest["inputs"].setdefault(key, value)

    if "ingest" not in manifest or not isinstance(manifest["ingest"], dict):
        manifest["ingest"] = {}

    for key, value in aliases.items():
        manifest["ingest"].setdefault(key, value)

    for key in ["price_ohlcv", "price_foreign"]:
        if key in aliases:
            artifact_files[key] = aliases[key]

    return manifest


def write_normalized_manifest(artifact_run_dir: Path, manifest: Dict[str, Any]) -> Path:
    artifact_run_dir = Path(artifact_run_dir)
    manifest_path = artifact_run_dir / "manifest.json"
    backup_path = artifact_run_dir / "manifest.original_before_contribution2.json"

    if manifest_path.exists() and not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    save_json(manifest, manifest_path)

    return manifest_path


def find_latest_artifact_run_dir(artifacts_root: str = "artifacts") -> Path:
    root = Path(artifacts_root)

    if not root.exists():
        raise FileNotFoundError(f"artifacts root가 없습니다: {root}")

    candidates = []
    candidates.extend(root.glob("run_id=*"))
    candidates.extend(root.glob("runid=*"))
    candidates.extend(root.glob("run_*"))

    for child in root.iterdir():
        if child.is_dir() and (child / "manifest.json").exists():
            candidates.append(child)

    candidates = list({p.resolve(): p for p in candidates if p.is_dir()}.values())

    if not candidates:
        raise FileNotFoundError(
            f"{artifacts_root} 아래에서 run_id/runid/run_* 또는 manifest.json 포함 폴더를 찾지 못했습니다."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


def resolve_ingestion_artifacts(
    ingested_output: Optional[Any] = None,
    artifacts_root: str = "artifacts",
    existing_artifact_run_dir: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_run_dir: Optional[Path] = None

    if existing_artifact_run_dir:
        artifact_run_dir = Path(existing_artifact_run_dir)

    elif isinstance(ingested_output, (str, Path)):
        candidate = Path(ingested_output)

        if candidate.exists() and candidate.is_dir():
            artifact_run_dir = candidate

    elif isinstance(ingested_output, dict):
        artifact_dir_keys = [
            "artifact_run_dir",
            "artifact_dir",
            "run_dir",
            "output_dir",
            "artifacts_dir",
            "saved_dir",
        ]

        for key in artifact_dir_keys:
            if ingested_output.get(key):
                candidate = Path(str(ingested_output[key]))

                if candidate.exists():
                    artifact_run_dir = candidate
                    break

        if artifact_run_dir is None and ingested_output.get("run_id"):
            run_id = str(ingested_output["run_id"])

            candidates = [
                Path(artifacts_root) / f"run_id={run_id}",
                Path(artifacts_root) / f"runid={run_id}",
                Path(artifacts_root) / run_id,
            ]

            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    artifact_run_dir = candidate
                    break

    if artifact_run_dir is None:
        artifact_run_dir = find_latest_artifact_run_dir(artifacts_root)

    if not artifact_run_dir.exists():
        raise FileNotFoundError(f"artifact_run_dir가 존재하지 않습니다: {artifact_run_dir}")

    manifest_path = artifact_run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    artifact_files = collect_artifact_files(artifact_run_dir)
    manifest = normalize_manifest(artifact_run_dir, manifest, artifact_files)
    manifest_path = write_normalized_manifest(artifact_run_dir, manifest)

    artifact_files = collect_artifact_files(artifact_run_dir)

    return {
        "artifact_run_dir": str(artifact_run_dir),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "artifact_files": artifact_files,
        **build_artifact_aliases(artifact_run_dir),
    }


def refresh_artifact_context(artifact_context: Dict[str, Any]) -> Dict[str, Any]:
    artifact_run_dir = Path(artifact_context["artifact_run_dir"])

    manifest_path = artifact_run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else artifact_context.get("manifest", {})

    artifact_files = collect_artifact_files(artifact_run_dir)
    manifest = normalize_manifest(artifact_run_dir, manifest, artifact_files)
    manifest_path = write_normalized_manifest(artifact_run_dir, manifest)

    artifact_context.update(
        {
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "artifact_files": collect_artifact_files(artifact_run_dir),
            **build_artifact_aliases(artifact_run_dir),
        }
    )

    return artifact_context


def build_common_artifact_payload(shared_outputs: Dict[str, Any]) -> Dict[str, Any]:
    artifact_context = shared_outputs["artifact_context"]
    aliases = build_artifact_aliases(Path(artifact_context["artifact_run_dir"]))

    return {
        **aliases,
        "artifact_files": artifact_context["artifact_files"],
        "manifest_path": artifact_context["manifest_path"],
        "manifest": artifact_context["manifest"],
        "input_paths": shared_outputs.get("input_paths", {}),
        "ingested_output": shared_outputs.get("ingested_output"),
        "artifacts_root": shared_outputs.get("artifacts_root", "artifacts"),
    }


def validate_ingestion_outputs(artifact_run_dir: str) -> None:
    artifact_run_dir_path = Path(artifact_run_dir)

    finance_dir = artifact_run_dir_path / "ingest" / "finance"
    price_dir = artifact_run_dir_path / "ingest" / "price"

    price_ohlcv = price_dir / "price_ohlcv.csv"
    price_foreign = price_dir / "price_foreign.csv"

    if not finance_dir.exists():
        raise FileNotFoundError(f"Missing finance directory: {finance_dir}")

    if not list(finance_dir.glob("*.csv")):
        raise FileNotFoundError(f"No financial csv files in: {finance_dir}")

    if not price_ohlcv.exists():
        raise FileNotFoundError(f"Missing price_ohlcv.csv: {price_ohlcv}")

    if not price_foreign.exists():
        print(f"[WARNING] price_foreign.csv not found: {price_foreign}")


# ============================================================
# 5. Execute Wrapper
# ============================================================

def build_agent_ctx_and_ap(input_data: Dict[str, Any]) -> tuple:
    input_data = input_data or {}

    artifacts_root = input_data.get("artifacts_root") or "artifacts"

    artifact_run_dir = (
        input_data.get("artifact_run_dir")
        or input_data.get("artifact_root")
        or input_data.get("run_dir")
        or input_data.get("artifact_dir")
        or input_data.get("artifacts_dir")
    )

    if artifact_run_dir is None:
        run_id = input_data.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_run_dir = str(Path(artifacts_root) / f"run_id={run_id}")

    artifact_run_dir = Path(artifact_run_dir)
    artifact_run_dir.mkdir(parents=True, exist_ok=True)

    aliases = build_artifact_aliases(artifact_run_dir)

    manifest_path = input_data.get("manifest_path") or str(artifact_run_dir / "manifest.json")
    artifact_files = input_data.get("artifact_files") or collect_artifact_files(artifact_run_dir)

    experiment_config = input_data.get("experiment_config") or input_data.get("config") or {}

    asof_date = (
        input_data.get("asof_date")
        or aliases.get("asof_date")
        or infer_asof_date_from_run_id(artifact_run_dir)
    )

    flags = input_data.get("flags") or {
        "use_analysis_layer": experiment_config.get("use_analysis_layer"),
        "use_rag": experiment_config.get("use_rag"),
        "market_flow_mode": experiment_config.get("market_flow_mode"),
        "experiment_name": input_data.get("experiment_name"),
        "asof_date": asof_date,
    }

    ctx_data = {
        **aliases,
        **input_data,

        "artifact_root": str(artifact_run_dir),
        "artifact_run_dir": str(artifact_run_dir),
        "run_dir": str(artifact_run_dir),
        "artifacts_root": str(artifacts_root),
        "project_root": str(Path.cwd()),

        "manifest_path": manifest_path,
        "artifact_files": artifact_files,
        "flags": flags,

        "asof_date": asof_date,
        "base_date": asof_date,
        "target_date": asof_date,
        "trade_date": asof_date,
    }

    ctx = to_dotdict(ctx_data)

    if "logger" not in ctx:
        ctx.logger = SimpleLogger()

    ap_payload = {
        **aliases,

        "artifacts_root": str(artifacts_root),
        "artifact_root": str(artifact_run_dir),
        "artifact_run_dir": str(artifact_run_dir),
        "artifact_dir": str(artifact_run_dir),
        "run_dir": str(artifact_run_dir),
        "project_root": str(Path.cwd()),

        "output_dir": str(
            input_data.get("agent_output_dir")
            or input_data.get("output_dir")
            or artifact_run_dir
        ),

        "manifest_path": manifest_path,
        "manifest": input_data.get("manifest") or {},
        "artifact_files": artifact_files,

        "input_paths": input_data.get("input_paths") or {},
        "ingested_output": input_data.get("ingested_output"),

        "experiment_name": input_data.get("experiment_name"),
        "experiment_config": experiment_config,
        "flags": flags,
        "run_id": input_data.get("run_id"),

        "asof_date": asof_date,
        "base_date": asof_date,
        "target_date": asof_date,
        "trade_date": asof_date,
    }

    ap = ArtifactPathContext(**ap_payload)

    return ctx, ap


def safe_execute(agent: Any, input_data: Dict[str, Any]) -> Any:
    input_data = input_data or {}
    ctx, ap = build_agent_ctx_and_ap(input_data)

    if not hasattr(ctx, "logger"):
        ctx.logger = SimpleLogger()

    method_names = ["execute", "run", "invoke", "generate", "process"]

    for method_name in method_names:
        if not hasattr(agent, method_name):
            continue

        method = getattr(agent, method_name)

        try:
            sig = inspect.signature(method)
            params = list(sig.parameters.values())

            positional_params = [
                p for p in params
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]

            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

            if has_varargs:
                return method(ctx, ap)

            if len(positional_params) >= 2:
                return method(ctx, ap)

            if len(positional_params) == 1:
                param_name = positional_params[0].name

                if param_name in ["ctx", "context", "state"]:
                    return method(ctx)

                return method(input_data)

            if len(positional_params) == 0:
                return method()

        except TypeError as e:
            try:
                return method(ctx, ap)
            except TypeError:
                try:
                    return method(ctx)
                except TypeError:
                    try:
                        return method(input_data)
                    except TypeError:
                        raise e

    if callable(agent):
        try:
            return agent(ctx, ap)
        except TypeError:
            try:
                return agent(ctx)
            except TypeError:
                return agent(input_data)

    raise AttributeError(
        f"{agent.__class__.__name__}에 execute/run/invoke/generate/process/__call__ 중 실행 가능한 메서드가 없습니다."
    )


# ============================================================
# 6. StageResult Utilities
# ============================================================

def stage_result_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)

    return getattr(result, key, default)


def is_stage_result(result: Any) -> bool:
    if isinstance(result, dict):
        return "stage" in result and "status" in result

    return hasattr(result, "stage") and hasattr(result, "status")


def stage_result_to_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return make_json_safe(result)

    data = {}

    for key in ["stage", "status", "inputs", "outputs", "metrics", "error"]:
        if hasattr(result, key):
            data[key] = make_json_safe(getattr(result, key))

    if data:
        return data

    if hasattr(result, "__dict__"):
        return make_json_safe(result.__dict__)

    return {"value": str(result)}


def assert_agent_success(result: Any, agent_name: str) -> None:
    if result is None:
        raise RuntimeError(f"{agent_name} returned None. Previous step failed.")

    status = str(stage_result_get(result, "status", "")).lower()
    error = stage_result_get(result, "error", None)

    if status in ["failed", "error", "exception"]:
        raise RuntimeError(f"{agent_name} failed: {error}")

    if error:
        raise RuntimeError(f"{agent_name} failed: {error}")

    if isinstance(result, str):
        lowered = result.lower()

        if "status='failed'" in lowered or "failed" in lowered or "error=" in lowered:
            raise RuntimeError(f"{agent_name} failed: {result}")

    if isinstance(result, dict):
        success = result.get("success")

        if success is False:
            raise RuntimeError(
                f"{agent_name} failed: {json.dumps(make_json_safe(result), ensure_ascii=False, indent=2)}"
            )

        for key in ["error", "exception", "traceback"]:
            if key in result and result[key]:
                raise RuntimeError(f"{agent_name} failed: {result[key]}")


def flatten_output_paths(outputs: Any) -> list[Any]:
    if outputs is None:
        return []

    if isinstance(outputs, dict):
        return list(outputs.values())

    if isinstance(outputs, (list, tuple, set)):
        result = []

        for item in outputs:
            result.extend(flatten_output_paths(item))

        return result

    return [outputs]


def read_output_file(path: Any) -> Any:
    if not path:
        return None

    path = Path(path)

    if not path.exists() or not path.is_file():
        return None

    suffix = path.suffix.lower()

    if suffix == ".json":
        return read_json(path)

    if suffix in [".md", ".txt"]:
        return path.read_text(encoding="utf-8")

    if suffix == ".csv":
        return str(path)

    return str(path)


def copy_stage_output_files(stage_result: Any, dest_dir: Path) -> None:
    outputs = stage_result_get(stage_result, "outputs", {}) or {}
    ensure_dir(dest_dir)

    if isinstance(outputs, dict):
        items = outputs.items()

    elif isinstance(outputs, list):
        items = []

        for value in outputs:
            if not value:
                continue

            src = Path(value)
            key = src.stem
            items.append((key, value))

    else:
        return

    for key, value in items:
        if not value:
            continue

        src = Path(value)

        if not src.exists():
            continue

        if src.is_file():
            target = dest_dir / src.name
            shutil.copy2(src, target)

        elif src.is_dir():
            target = dest_dir / key

            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(src, target)


def extract_primary_output(stage_result: Any, preferred_keys: list[str]) -> Any:
    if not is_stage_result(stage_result):
        return stage_result

    outputs = stage_result_get(stage_result, "outputs", {}) or {}

    if isinstance(outputs, dict):
        for key in preferred_keys:
            if key in outputs:
                loaded = read_output_file(outputs[key])

                if loaded is not None:
                    return loaded

        for _, value in outputs.items():
            loaded = read_output_file(value)

            if loaded is not None:
                return loaded

    elif isinstance(outputs, list):
        for preferred_key in preferred_keys:
            for value in outputs:
                if not value:
                    continue

                path = Path(value)

                if preferred_key in path.stem or preferred_key in path.name:
                    loaded = read_output_file(path)

                    if loaded is not None:
                        return loaded

        for value in outputs:
            loaded = read_output_file(value)

            if loaded is not None:
                return loaded

    return stage_result_to_dict(stage_result)


def materialize_agent_result(
    stage_result: Any,
    output_dir: Path,
    stage_name: str,
    preferred_keys: list[str],
) -> Any:
    stage_result_dir = output_dir / "_stage_results"
    stage_files_dir = output_dir / "_stage_files" / stage_name

    ensure_dir(stage_result_dir)
    ensure_dir(stage_files_dir)

    save_json(stage_result_to_dict(stage_result), stage_result_dir / f"{stage_name}_stage_result.json")
    copy_stage_output_files(stage_result, stage_files_dir)

    return extract_primary_output(stage_result, preferred_keys)


def to_markdown_report(report_output: Any) -> str:
    if isinstance(report_output, str):
        return report_output

    if is_stage_result(report_output):
        report_output = extract_primary_output(
            report_output,
            preferred_keys=["report_output", "report_md", "markdown", "final_report"],
        )

    if isinstance(report_output, str):
        return report_output

    if isinstance(report_output, dict):
        for key in [
            "markdown",
            "report_md",
            "content",
            "report",
            "final_report",
            "summary_markdown",
        ]:
            if key in report_output and isinstance(report_output[key], str):
                return report_output[key]

        return "```json\n" + json.dumps(make_json_safe(report_output), ensure_ascii=False, indent=2) + "\n```"

    return str(report_output)


# ============================================================
# 7. RAG Validation / Metrics
# ============================================================

def get_nested_dict(obj: Any, key: str) -> Optional[dict]:
    if isinstance(obj, dict) and isinstance(obj.get(key), dict):
        return obj[key]

    return None


def get_list(obj: Any, key: str) -> list:
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        return obj[key]

    return []


def extract_rag_metrics(news_investigation: Any) -> Dict[str, Any]:
    if not isinstance(news_investigation, dict):
        return {
            "rag_enabled": False,
            "rag_method": None,
            "retrieval_query_count": 0,
            "retrieved_evidence_count": 0,
            "candidate_evidence_count": 0,
            "rejected_evidence_count": 0,
            "verification_status": "not_dict",
        }

    rag = news_investigation.get("rag") if isinstance(news_investigation.get("rag"), dict) else {}

    meta = news_investigation.get("meta") if isinstance(news_investigation.get("meta"), dict) else {}

    verification_result = (
        news_investigation.get("verification_result")
        if isinstance(news_investigation.get("verification_result"), dict)
        else rag.get("verification_result", {})
    )

    retrieval_queries = (
        news_investigation.get("retrieval_queries")
        if isinstance(news_investigation.get("retrieval_queries"), list)
        else rag.get("retrieval_queries", [])
    )

    retrieved_evidence = (
        news_investigation.get("retrieved_evidence")
        if isinstance(news_investigation.get("retrieved_evidence"), list)
        else rag.get("retrieved_evidence", [])
    )

    candidate_evidence = rag.get("candidate_evidence", [])
    rejected_evidence = rag.get("rejected_evidence", [])

    return {
        "rag_enabled": bool(
            news_investigation.get("rag_enabled")
            or meta.get("rag_enabled")
            or rag.get("enabled")
        ),
        "rag_method": rag.get("method") or meta.get("rag_method"),
        "retrieval_query_count": len(retrieval_queries) if isinstance(retrieval_queries, list) else 0,
        "retrieved_evidence_count": len(retrieved_evidence) if isinstance(retrieved_evidence, list) else 0,
        "candidate_evidence_count": len(candidate_evidence) if isinstance(candidate_evidence, list) else 0,
        "rejected_evidence_count": len(rejected_evidence) if isinstance(rejected_evidence, list) else 0,
        "verification_status": verification_result.get("status") if isinstance(verification_result, dict) else None,
        "supported_query_count": verification_result.get("supported_query_count") if isinstance(verification_result, dict) else None,
        "unsupported_query_count": verification_result.get("unsupported_query_count") if isinstance(verification_result, dict) else None,
        "kept_relevance_counts": verification_result.get("kept_relevance_counts") if isinstance(verification_result, dict) else None,
        "candidate_relevance_counts": verification_result.get("candidate_relevance_counts") if isinstance(verification_result, dict) else None,
    }


def validate_rag_usage_for_config(
    experiment_name: str,
    config: Dict[str, Any],
    news_investigation: Any,
) -> None:
    if not config.get("use_analysis_layer"):
        return

    metrics = extract_rag_metrics(news_investigation)

    if config.get("use_rag"):
        if not metrics["rag_enabled"]:
            raise RuntimeError(
                f"{experiment_name}: use_rag=True인데 news_investigation 결과에서 rag_enabled를 확인하지 못했습니다."
            )

        if not metrics["rag_method"]:
            raise RuntimeError(
                f"{experiment_name}: use_rag=True인데 RAG method가 없습니다."
            )

        if metrics["retrieved_evidence_count"] <= 0:
            raise RuntimeError(
                f"{experiment_name}: use_rag=True인데 retrieved_evidence가 0개입니다. "
                f"RAG가 호출됐지만 usable evidence가 없는 상태입니다."
            )

    else:
        if metrics["retrieved_evidence_count"] > 0:
            raise RuntimeError(
                f"{experiment_name}: use_rag=False인데 retrieved_evidence가 존재합니다. "
                f"이전 RAG artifact가 섞였을 가능성이 큽니다."
            )


# ============================================================
# 8. Shared Front Pipeline
# ============================================================

def run_shared_front_pipeline(
    input_paths: Dict[str, str],
    shared_dir: Path,
    artifacts_root: str = "artifacts",
    existing_artifact_run_dir: Optional[str] = None,
) -> Dict[str, Any]:

    print("\n[1] Shared front pipeline started")

    if existing_artifact_run_dir:
        print(" - Existing artifact_run_dir provided. Data Ingestion Agent skipped.")

        ingested_output = {
            "skipped_data_ingestion": True,
            "artifact_run_dir": existing_artifact_run_dir,
            "artifact_root": existing_artifact_run_dir,
        }

    else:
        data_ingestion_agent = try_init_agent(DataIngestionAgent)

        ingestion_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_run_dir_for_ingestion = Path(artifacts_root) / f"run_id={ingestion_run_id}"
        artifact_run_dir_for_ingestion.mkdir(parents=True, exist_ok=True)

        ingested_output = safe_execute(
            data_ingestion_agent,
            {
                "input_paths": input_paths,
                "artifacts_root": artifacts_root,
                "artifact_run_dir": str(artifact_run_dir_for_ingestion),
                "artifact_root": str(artifact_run_dir_for_ingestion),
                "run_id": ingestion_run_id,
            },
        )

        if isinstance(ingested_output, dict):
            ingested_output.setdefault("artifact_run_dir", str(artifact_run_dir_for_ingestion))
            ingested_output.setdefault("artifact_root", str(artifact_run_dir_for_ingestion))
            ingested_output.setdefault("run_id", ingestion_run_id)
        else:
            ingested_output = {
                "data_ingestion_result": ingested_output,
                "artifact_run_dir": str(artifact_run_dir_for_ingestion),
                "artifact_root": str(artifact_run_dir_for_ingestion),
                "run_id": ingestion_run_id,
            }

        print(" - Data Ingestion Agent completed")

    save_json(ingested_output, shared_dir / "01_ingestion_output.json")

    artifact_context = resolve_ingestion_artifacts(
        ingested_output=ingested_output,
        artifacts_root=artifacts_root,
        existing_artifact_run_dir=existing_artifact_run_dir,
    )

    save_json(artifact_context, shared_dir / "02_artifact_context.json")

    print(f" - Artifact run dir resolved: {artifact_context['artifact_run_dir']}")

    validate_ingestion_outputs(artifact_context["artifact_run_dir"])

    common_payload = {
        **build_artifact_aliases(Path(artifact_context["artifact_run_dir"])),
        "artifact_files": artifact_context["artifact_files"],
        "manifest_path": artifact_context["manifest_path"],
        "manifest": artifact_context["manifest"],
        "input_paths": input_paths,
        "ingested_output": ingested_output,
        "artifacts_root": artifacts_root,
    }

    # --------------------------------------------------------
    # 1) Feature Extraction
    # --------------------------------------------------------

    feature_extraction_agent = try_init_agent(FeatureExtractionAgent)

    feature_data = safe_execute(
        feature_extraction_agent,
        {
            **common_payload,
        },
    )

    assert_agent_success(feature_data, "FeatureExtractionAgent")

    artifact_context = refresh_artifact_context(artifact_context)
    common_payload.update(
        {
            **build_artifact_aliases(Path(artifact_context["artifact_run_dir"])),
            "artifact_files": artifact_context["artifact_files"],
            "manifest_path": artifact_context["manifest_path"],
            "manifest": artifact_context["manifest"],
        }
    )

    save_json(feature_data, shared_dir / "03_feature_data.json")
    print(" - Feature Extraction Agent completed")

    # --------------------------------------------------------
    # 2) Preprocessing Recommender
    # --------------------------------------------------------

    preprocessing_recommender = try_init_agent(PreprocessingRecommenderAgent)

    preprocessing_plan = safe_execute(
        preprocessing_recommender,
        {
            **common_payload,
            "feature_data": feature_data,
        },
    )

    assert_agent_success(preprocessing_plan, "PreprocessingRecommenderAgent")

    artifact_context = refresh_artifact_context(artifact_context)
    common_payload.update(
        {
            **build_artifact_aliases(Path(artifact_context["artifact_run_dir"])),
            "artifact_files": artifact_context["artifact_files"],
            "manifest_path": artifact_context["manifest_path"],
            "manifest": artifact_context["manifest"],
        }
    )

    save_json(preprocessing_plan, shared_dir / "04_preprocessing_plan.json")
    print(" - Preprocessing Recommender Agent completed")

    # --------------------------------------------------------
    # 3) Model-Data Matcher
    # --------------------------------------------------------

    model_data_matcher = try_init_agent(ModelDataMatcherAgent)

    model_data_match_result = safe_execute(
        model_data_matcher,
        {
            **common_payload,
            "feature_data": feature_data,
            "preprocessing_plan": preprocessing_plan,
        },
    )

    assert_agent_success(model_data_match_result, "ModelDataMatcherAgent")

    artifact_context = refresh_artifact_context(artifact_context)
    common_payload.update(
        {
            **build_artifact_aliases(Path(artifact_context["artifact_run_dir"])),
            "artifact_files": artifact_context["artifact_files"],
            "manifest_path": artifact_context["manifest_path"],
            "manifest": artifact_context["manifest"],
        }
    )

    save_json(model_data_match_result, shared_dir / "05_model_data_match_result.json")
    print(" - Model-Data Matcher Agent completed")

    # --------------------------------------------------------
    # 4) Preprocessing Implementor
    # --------------------------------------------------------

    preprocessing_implementor = try_init_agent(PreprocessingImplementorAgent)

    preprocessed_data = safe_execute(
        preprocessing_implementor,
        {
            **common_payload,
            "feature_data": feature_data,
            "preprocessing_plan": preprocessing_plan,
            "model_data_match_result": model_data_match_result,
        },
    )

    assert_agent_success(preprocessed_data, "PreprocessingImplementorAgent")

    artifact_context = refresh_artifact_context(artifact_context)

    save_json(preprocessed_data, shared_dir / "06_preprocessed_data.json")
    print(" - Preprocessing Implementor Agent completed")

    shared_outputs = {
        "input_paths": input_paths,
        "artifacts_root": artifacts_root,
        "ingested_output": ingested_output,
        "artifact_context": artifact_context,
        "feature_data": feature_data,
        "preprocessing_plan": preprocessing_plan,
        "model_data_match_result": model_data_match_result,
        "preprocessed_data": preprocessed_data,
    }

    save_json(shared_outputs, shared_dir / "shared_outputs.json")

    print("[1] Shared front pipeline finished")

    return shared_outputs


# ============================================================
# 9. Analysis Layer
# ============================================================

def run_analysis_layer(
    experiment_name: str,
    shared_outputs: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:

    if not config["use_analysis_layer"]:
        print(" - Analysis layer skipped for Baseline")

        analysis_outputs = {
            "enabled": False,
            "use_rag": False,
            "market_analysis": None,
            "news_investigation": None,
            "risk_score": None,
            "risk_management": None,
            "rag_metrics": extract_rag_metrics(None),
        }

        save_json(analysis_outputs, output_dir / "analysis_outputs.json")

        return analysis_outputs

    print(" - Analysis layer started")

    use_rag = bool(config["use_rag"])
    common_payload = build_common_artifact_payload(shared_outputs)

    flags = {
        "experiment_name": experiment_name,
        "use_analysis_layer": config["use_analysis_layer"],
        "use_rag": config["use_rag"],
        "market_flow_mode": config["market_flow_mode"],
    }

    # --------------------------------------------------------
    # 1) Market Analysis Agent
    # --------------------------------------------------------

    market_analysis_agent = try_init_agent(MarketAnalysisAgent)

    market_analysis_stage = safe_execute(
        market_analysis_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "use_rag": False,
        },
    )

    assert_agent_success(market_analysis_stage, "MarketAnalysisAgent")

    market_analysis = materialize_agent_result(
        market_analysis_stage,
        output_dir,
        "market_analysis",
        preferred_keys=[
            "market_analysis_result",
            "market_analysis",
            "analysis_result",
            "result",
        ],
    )

    save_json(market_analysis, output_dir / "analysis_01_market_analysis.json")
    print("   - Market Analysis Agent completed")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 2) News Investigation Agent
    # --------------------------------------------------------

    news_investigation_agent = try_init_agent(
        NewsInvestigationAgent,
        use_rag=use_rag,
    )

    news_investigation_stage = safe_execute(
        news_investigation_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "market_analysis": market_analysis,
            "use_rag": use_rag,
        },
    )

    assert_agent_success(news_investigation_stage, "NewsInvestigationAgent")

    news_investigation = materialize_agent_result(
        news_investigation_stage,
        output_dir,
        "news_invest",
        preferred_keys=[
            "news_invest_result",
            "news_investigation_result",
            "news_investigation",
            "news_result",
            "result",
        ],
    )

    validate_rag_usage_for_config(
        experiment_name=experiment_name,
        config=config,
        news_investigation=news_investigation,
    )

    rag_metrics = extract_rag_metrics(news_investigation)

    save_json(news_investigation, output_dir / "analysis_02_news_investigation.json")
    save_json(rag_metrics, output_dir / "analysis_02_news_rag_metrics.json")

    print(
        f"   - News Investigation Agent completed | "
        f"RAG={use_rag} | evidence={rag_metrics['retrieved_evidence_count']}"
    )

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 3) Risk Score Agent
    # --------------------------------------------------------

    risk_score_agent = try_init_agent(RiskScoreAgent)

    risk_score_stage = safe_execute(
        risk_score_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "market_analysis": market_analysis,
            "news_investigation": news_investigation,
            "use_rag": use_rag,
        },
    )

    assert_agent_success(risk_score_stage, "RiskScoreAgent")

    risk_score = materialize_agent_result(
        risk_score_stage,
        output_dir,
        "risk_score",
        preferred_keys=[
            "risk_score_result",
            "risk_result",
            "risk_score",
            "result",
        ],
    )

    save_json(risk_score, output_dir / "analysis_03_risk_score.json")
    print("   - RiskScoreAgent completed")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])

    analysis_outputs = {
        "enabled": True,
        "use_rag": use_rag,
        "market_analysis": market_analysis,
        "news_investigation": news_investigation,
        "risk_score": risk_score,
        "risk_management": risk_score,
        "rag_metrics": rag_metrics,
        "stage_results": {
            "market_analysis": stage_result_to_dict(market_analysis_stage),
            "news_investigation": stage_result_to_dict(news_investigation_stage),
            "risk_score": stage_result_to_dict(risk_score_stage),
        },
    }

    save_json(analysis_outputs, output_dir / "analysis_outputs.json")

    print(" - Analysis layer finished")

    return analysis_outputs


# ============================================================
# 10. Report Pipeline
# ============================================================

def run_report_pipeline(
    experiment_name: str,
    config: Dict[str, Any],
    shared_outputs: Dict[str, Any],
    analysis_outputs: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:

    print(f"\n[2] Report pipeline started: {experiment_name}")

    common_payload = build_common_artifact_payload(shared_outputs)

    flags = {
        "experiment_name": experiment_name,
        "use_analysis_layer": config["use_analysis_layer"],
        "use_rag": config["use_rag"],
        "market_flow_mode": config["market_flow_mode"],
    }

    # --------------------------------------------------------
    # 1) Report Target Planner Agent
    # --------------------------------------------------------

    report_target_planner = try_init_agent(ReportTargetPlannerAgent)

    report_target_stage = safe_execute(
        report_target_planner,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "analysis_outputs": analysis_outputs,
        },
    )

    assert_agent_success(report_target_stage, "ReportTargetPlannerAgent")

    report_target_spec = materialize_agent_result(
        report_target_stage,
        output_dir,
        "report_target_planner",
        preferred_keys=[
            "report_target_spec",
            "report_feature_contract",
            "candidate_scoring_plan",
            "evidence_builder_contract",
            "report_output_contract",
        ],
    )

    save_json(report_target_spec, output_dir / "01_report_target_spec.json")
    print(" - Report Target Planner Agent completed")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 2) Market Flow Agent
    # --------------------------------------------------------

    market_flow_agent = try_init_agent(
        MarketFlowAgent,
        mode=config["market_flow_mode"],
    )

    market_flow_stage = safe_execute(
        market_flow_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "mode": config["market_flow_mode"],
            "market_flow_mode": config["market_flow_mode"],
            "use_analysis_layer": config["use_analysis_layer"],
            "use_rag": config["use_rag"],
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "analysis_outputs": analysis_outputs,
            "report_target_spec": report_target_spec,
            "instruction": build_market_flow_instruction(config),
        },
    )

    assert_agent_success(market_flow_stage, "MarketFlowAgent")

    market_flow_result = materialize_agent_result(
        market_flow_stage,
        output_dir,
        "market_flow",
        preferred_keys=[
            "market_flow_result",
            "market_flow_summary",
        ],
    )

    save_json(market_flow_result, output_dir / "02_market_flow_result.json")
    print(f" - Market Flow Agent completed | mode={config['market_flow_mode']}")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 3) Candidate Scoring Agent
    # --------------------------------------------------------

    candidate_scoring_agent = try_init_agent(CandidateScoringAgent)

    candidate_scoring_stage = safe_execute(
        candidate_scoring_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "analysis_outputs": analysis_outputs,
            "report_target_spec": report_target_spec,
            "market_flow_result": market_flow_result,
        },
    )

    assert_agent_success(candidate_scoring_stage, "CandidateScoringAgent")

    candidate_scoring_result = materialize_agent_result(
        candidate_scoring_stage,
        output_dir,
        "candidate_scoring",
        preferred_keys=[
            "candidate_scoring_result",
            "candidate_scoring_top10",
            "candidate_scoring_all_candidates",
        ],
    )

    save_json(candidate_scoring_result, output_dir / "03_candidate_scoring_result.json")
    print(" - Candidate Scoring Agent completed")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 4) Report Evidence Builder Agent
    # --------------------------------------------------------

    evidence_builder_agent = try_init_agent(ReportEvidenceBuilderAgent)

    evidence_stage = safe_execute(
        evidence_builder_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "preprocessed_data": shared_outputs["preprocessed_data"],
            "feature_data": shared_outputs["feature_data"],
            "preprocessing_plan": shared_outputs["preprocessing_plan"],
            "model_data_match_result": shared_outputs["model_data_match_result"],
            "analysis_outputs": analysis_outputs,
            "report_target_spec": report_target_spec,
            "market_flow_result": market_flow_result,
            "candidate_scoring_result": candidate_scoring_result,
            "instruction": build_evidence_builder_instruction(config),
        },
    )

    assert_agent_success(evidence_stage, "ReportEvidenceBuilderAgent")

    evidence_result = materialize_agent_result(
        evidence_stage,
        output_dir,
        "report_evidence",
        preferred_keys=[
            "report_evidence_result",
            "evidence_result",
        ],
    )

    save_json(evidence_result, output_dir / "04_report_evidence_result.json")
    print(" - Report Evidence Builder Agent completed")

    shared_outputs["artifact_context"] = refresh_artifact_context(shared_outputs["artifact_context"])
    common_payload = build_common_artifact_payload(shared_outputs)

    # --------------------------------------------------------
    # 5) Report Generative Agent
    # --------------------------------------------------------

    report_generative_agent = try_init_agent(ReportGenerativeAgent)

    final_report_stage = safe_execute(
        report_generative_agent,
        {
            **common_payload,
            "experiment_name": experiment_name,
            "experiment_config": config,
            "flags": flags,
            "report_target_spec": report_target_spec,
            "market_flow_result": market_flow_result,
            "candidate_scoring_result": candidate_scoring_result,
            "evidence_result": evidence_result,
            "analysis_outputs": analysis_outputs,
            "instruction": build_report_generation_instruction(config),
        },
    )

    assert_agent_success(final_report_stage, "ReportGenerativeAgent")

    final_report = materialize_agent_result(
        final_report_stage,
        output_dir,
        "report_generation",
        preferred_keys=[
            "report_output",
            "report_md",
            "markdown",
            "final_report",
        ],
    )

    save_json(final_report, output_dir / "05_final_report.json")
    save_text(to_markdown_report(final_report), output_dir / "05_final_report.md")
    print(" - Report Generative Agent completed")

    run_result = {
        "experiment_name": experiment_name,
        "experiment_config": config,
        "analysis_outputs": analysis_outputs,
        "report_target_spec": report_target_spec,
        "market_flow_result": market_flow_result,
        "candidate_scoring_result": candidate_scoring_result,
        "evidence_result": evidence_result,
        "final_report": final_report,
    }

    save_json(run_result, output_dir / "run_result.json")

    print(f"[2] Report pipeline finished: {experiment_name}")

    return run_result


# ============================================================
# 11. Mode-specific Instructions
# ============================================================

def build_market_flow_instruction(config: Dict[str, Any]) -> str:
    if not config["use_analysis_layer"]:
        return """
BASELINE MODE.

Generate the market flow summary only from structured market data,
price data, volume data, supply-demand data, and internal news features.

Do not use MarketAnalysisAgent, NewsInvestigationAgent, or RiskScoreAgent outputs.
Do not use RAG evidence.
Do not claim external verification.

Allowed:
- summarize observed price movement
- summarize volume changes
- summarize foreign/institutional flows
- summarize frequent news keywords or provided news features

Not allowed:
- deep causal interpretation beyond provided data
- RAG-based evidence
- external news verification
- unsupported financial interpretation
"""

    if config["use_analysis_layer"] and not config["use_rag"]:
        return """
OURS A MODE.

Generate the market flow summary using the multi-agent analysis outputs:
MarketAnalysisAgent, NewsInvestigationAgent, and RiskScoreAgent.

Use only internal dataset and agent outputs.
Do not use RAG evidence.
Do not claim that information was externally or retrieval-verified.

Allowed:
- synthesize market, news, and risk analysis
- explain likely market flow based on internal evidence
- connect stock candidates to observed market conditions

Not allowed:
- external RAG evidence
- retrieved evidence claims
- unsupported speculation
"""

    return """
OURS B MODE.

Generate the market flow summary using multi-agent analysis outputs and
RAG-supported NewsInvestigationAgent evidence.

Use retrieved_evidence when available.
Prefer claims supported by direct_company_news or direct_business_relation evidence.
Avoid unsupported speculation.
Explain uncertainty when evidence is limited.
"""


def build_evidence_builder_instruction(config: Dict[str, Any]) -> str:
    if not config["use_analysis_layer"]:
        return """
BASELINE MODE.

Build report evidence only from structured input data.
Evidence should be limited to directly observed data points.

Do not create causal evidence.
Do not include RAG-based verification.
Do not include retrieved evidence.
"""

    if config["use_analysis_layer"] and not config["use_rag"]:
        return """
OURS A MODE.

Build report evidence from structured data and multi-agent analysis outputs.
Use internal news and market evidence only.
Do not include RAG retrieved evidence.
Do not claim source-level verification.
"""

    return """
OURS B MODE.

Build report evidence from structured data, multi-agent analysis outputs,
and RAG-supported retrieved evidence.

Prefer:
1. direct_company_news
2. direct_business_relation
3. internally observed structured data

Do not use indirect_mention, market_context, or low_relevance evidence as core support.
"""


def build_report_generation_instruction(config: Dict[str, Any]) -> str:
    if not config["use_analysis_layer"]:
        return """
BASELINE MODE.

Generate the final report using the same report structure as other experiments.
However, the content should be based only on structured data and internal news features.

Keep claims descriptive and data-based.
Avoid deep causal interpretation.
Avoid RAG evidence.
Avoid unsupported financial reasoning.
"""

    if config["use_analysis_layer"] and not config["use_rag"]:
        return """
OURS A MODE.

Generate the final report using multi-agent analysis outputs.
The report should include more coherent market interpretation than baseline,
but must not claim RAG evidence or source-level retrieval verification.
"""

    return """
OURS B MODE.

Generate the final report using multi-agent analysis outputs and RAG-supported evidence.
The report should emphasize factual consistency, evidence grounding,
financial validity, and reliability.

When possible, ground important claims in retrieved_evidence.
"""


# ============================================================
# 12. Evaluation
# ============================================================

def count_evidence_items(evidence_result: Any) -> int:
    if evidence_result is None:
        return 0

    if isinstance(evidence_result, list):
        return len(evidence_result)

    if isinstance(evidence_result, dict):
        for key in ["evidence", "evidence_items", "claims", "supporting_evidence", "items"]:
            if key in evidence_result and isinstance(evidence_result[key], list):
                return len(evidence_result[key])

        return len(evidence_result.keys())

    return 1


def simple_evaluate_report(run_result: Dict[str, Any]) -> Dict[str, Any]:
    experiment_name = run_result["experiment_name"]
    config = run_result["experiment_config"]

    final_report_text = to_markdown_report(run_result["final_report"])
    evidence_count = count_evidence_items(run_result.get("evidence_result"))

    analysis_outputs = run_result.get("analysis_outputs") or {}
    rag_metrics = analysis_outputs.get("rag_metrics") if isinstance(analysis_outputs, dict) else None

    return {
        "experiment_name": experiment_name,
        "label": config["label"],
        "use_analysis_layer": config["use_analysis_layer"],
        "use_rag": config["use_rag"],
        "market_flow_mode": config["market_flow_mode"],
        "report_length_chars": len(final_report_text),
        "evidence_count_proxy": evidence_count,
        "rag_metrics": rag_metrics,
        "note": "Proxy metrics only. Final validation should use rubric-based human or LLM evaluation.",
    }


def build_comparison_markdown(evaluations: Dict[str, Any]) -> str:
    lines = []

    lines.append("# Contribution 2 Experiment Comparison")
    lines.append("")
    lines.append("| Experiment | Analysis Layer | RAG | Market Flow Mode | Report Length | Evidence Count | RAG Evidence |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")

    for _, ev in evaluations.items():
        rag_metrics = ev.get("rag_metrics") or {}
        rag_evidence_count = rag_metrics.get("retrieved_evidence_count", 0)

        lines.append(
            f"| {ev['label']} | "
            f"{'O' if ev['use_analysis_layer'] else 'X'} | "
            f"{'O' if ev['use_rag'] else 'X'} | "
            f"{ev['market_flow_mode']} | "
            f"{ev['report_length_chars']} | "
            f"{ev['evidence_count_proxy']} | "
            f"{rag_evidence_count} |"
        )

    lines.append("")
    lines.append("## Experimental Conditions")
    lines.append("")
    lines.append("- Baseline: data-only report generation without analysis layer or RAG.")
    lines.append("- Ours A: multi-agent analysis layer without RAG.")
    lines.append("- Ours B: multi-agent analysis layer with local news-corpus RAG evidence verification.")
    lines.append("")
    lines.append("## Recommended Final Evaluation Metrics")
    lines.append("")
    lines.append("1. Factual Consistency")
    lines.append("2. Evidence Grounding")
    lines.append("3. Coherence")
    lines.append("4. Financial Validity")
    lines.append("5. Reliability")

    return "\n".join(lines)


# ============================================================
# 13. Workspace Isolation
# ============================================================

def clean_experiment_workspace(artifact_dir: Path) -> None:
    """
    기존 최종 pipeline에서 생성된 market_analysis/news_invest/report outputs가
    Baseline/Ours A/Ours B 실험에 섞이지 않도록 제거한다.

    ingest, feature_table, prep_reco, model_match_v1, prep_apply는 유지.
    """

    artifact_dir = Path(artifact_dir)

    for stage_dir_name in STALE_EXPERIMENT_STAGE_DIRS:
        target = artifact_dir / stage_dir_name

        if target.exists():
            shutil.rmtree(target)

    contribution_dir = artifact_dir / "contribution2_experiment"

    if contribution_dir.exists():
        shutil.rmtree(contribution_dir)


def copy_artifact_workspace(src_artifact_dir: Path, dst_artifact_dir: Path) -> None:
    src_artifact_dir = Path(src_artifact_dir)
    dst_artifact_dir = Path(dst_artifact_dir)

    if dst_artifact_dir.exists():
        shutil.rmtree(dst_artifact_dir)

    def ignore_func(dir_path, names):
        ignored = set()

        for name in names:
            if name in [
                "contribution2_experiment",
                "__pycache__",
                ".git",
                ".ipynb_checkpoints",
            ]:
                ignored.add(name)

        return ignored

    shutil.copytree(src_artifact_dir, dst_artifact_dir, ignore=ignore_func)
    clean_experiment_workspace(dst_artifact_dir)


def clone_shared_outputs_for_experiment(
    shared_outputs: Dict[str, Any],
    experiment_artifact_dir: Path,
) -> Dict[str, Any]:
    cloned = copy.deepcopy(shared_outputs)

    artifact_context = resolve_ingestion_artifacts(
        ingested_output={"artifact_run_dir": str(experiment_artifact_dir)},
        artifacts_root=cloned.get("artifacts_root", "artifacts"),
        existing_artifact_run_dir=str(experiment_artifact_dir),
    )

    cloned["artifact_context"] = artifact_context
    cloned["artifacts_root"] = str(experiment_artifact_dir.parent)

    return cloned


# ============================================================
# 14. Main Runner
# ============================================================

def run_all_experiments(
    input_paths: Dict[str, str],
    artifacts_root: str = "artifacts",
    existing_artifact_run_dir: Optional[str] = None,
    output_root: str = "experiment_runs",
) -> Path:

    run_id = now_run_id()

    bootstrap_dir = Path(output_root) / f"_bootstrap_{run_id}"
    bootstrap_shared_dir = bootstrap_dir / "shared"
    ensure_dir(bootstrap_shared_dir)

    print("=" * 80)
    print("Contribution 2 Experiment Started")
    print(f"Experiment Run ID: {run_id}")
    print("=" * 80)

    shared_outputs = run_shared_front_pipeline(
        input_paths=input_paths,
        shared_dir=bootstrap_shared_dir,
        artifacts_root=artifacts_root,
        existing_artifact_run_dir=existing_artifact_run_dir,
    )

    artifact_run_dir = Path(shared_outputs["artifact_context"]["artifact_run_dir"])

    run_dir = artifact_run_dir / "contribution2_experiment" / run_id
    ensure_dir(run_dir)

    final_shared_dir = run_dir / "shared"

    if final_shared_dir.exists():
        shutil.rmtree(final_shared_dir)

    shutil.copytree(bootstrap_shared_dir, final_shared_dir)

    try:
        shutil.rmtree(bootstrap_dir)
    except Exception:
        pass

    experiment_manifest = {
        "experiment_run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "artifact_run_dir": str(artifact_run_dir),
        "input_paths": input_paths,
        "existing_artifact_run_dir": existing_artifact_run_dir,
        "experiment_configs": EXPERIMENT_CONFIGS,
        "notes": {
            "workspace_isolation": (
                "Each experiment runs in its own copied artifact workspace. "
                "Existing analysis/report stage directories are removed before execution "
                "to prevent stale RAG outputs from leaking into Baseline or Ours A."
            )
        },
    }

    save_json(experiment_manifest, run_dir / "experiment_manifest.json")

    print(f"\nExperiment output dir: {run_dir}")

    evaluations: Dict[str, Any] = {}

    for experiment_name, config in EXPERIMENT_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f"Running experiment: {config['label']}")
        print("=" * 80)

        exp_dir = run_dir / experiment_name
        ensure_dir(exp_dir)

        save_json(config, exp_dir / "experiment_config.json")

        exp_work_artifact_dir = run_dir / "_work_artifacts" / experiment_name

        copy_artifact_workspace(
            src_artifact_dir=artifact_run_dir,
            dst_artifact_dir=exp_work_artifact_dir,
        )

        experiment_shared_outputs = clone_shared_outputs_for_experiment(
            shared_outputs=shared_outputs,
            experiment_artifact_dir=exp_work_artifact_dir,
        )

        save_json(
            {
                "experiment_name": experiment_name,
                "work_artifact_dir": str(exp_work_artifact_dir),
                "cleaned_stage_dirs": STALE_EXPERIMENT_STAGE_DIRS,
            },
            exp_dir / "work_artifact_context.json",
        )

        analysis_outputs = run_analysis_layer(
            experiment_name=experiment_name,
            shared_outputs=experiment_shared_outputs,
            config=config,
            output_dir=exp_dir,
        )

        run_result = run_report_pipeline(
            experiment_name=experiment_name,
            config=config,
            shared_outputs=experiment_shared_outputs,
            analysis_outputs=analysis_outputs,
            output_dir=exp_dir,
        )

        evaluation = simple_evaluate_report(run_result)
        evaluations[experiment_name] = evaluation

        save_json(evaluation, exp_dir / "evaluation_proxy.json")

    evaluation_dir = run_dir / "evaluation"
    ensure_dir(evaluation_dir)

    save_json(evaluations, evaluation_dir / "evaluation_proxy_summary.json")

    comparison_md = build_comparison_markdown(evaluations)
    save_text(comparison_md, evaluation_dir / "comparison_table.md")

    print("\n" + "=" * 80)
    print("Contribution 2 Experiment Finished")
    print(f"Results saved to: {run_dir}")
    print("=" * 80)

    return run_dir


# ============================================================
# 15. CLI Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="artifacts",
        help="Data Ingestion Agent가 run_id 폴더를 저장하는 root 경로",
    )

    parser.add_argument(
        "--artifact-run-dir",
        type=str,
        default=None,
        help="이미 생성된 artifacts/run_id=... 폴더를 사용할 경우 지정. 지정하면 Data Ingestion Agent를 skip함.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="experiment_runs",
        help="bootstrap 임시 저장 위치",
    )

    args = parser.parse_args()

    INPUT_PATHS = {}

    run_all_experiments(
        input_paths=INPUT_PATHS,
        artifacts_root=args.artifacts_root,
        existing_artifact_run_dir=args.artifact_run_dir,
        output_root=args.output_root,
    )