from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.agent_base import BaseAgent
from core.result import StageResult
from core.context import RunContext
from core.artifacts import ArtifactPaths


@dataclass
class PrepApplyArtifacts:
    run_dir: Path
    output_dir: Path
    manifest_path: Path
    output_files: List[str]


class PreprocessingImplementorAgent(BaseAgent):
    """
    preprocessing_plan.json + data_contract.json + model_matching_result.json
    + feature_table 하위 csv들을 읽어서
    모델 입력용 preprocessed csv들을 생성하는 agent
    """

    stage = "prep_apply"

    def __init__(
        self,
        encoding: str = "utf-8",
        sample_rows: int = 200_000,
        random_state: int = 42,
    ):
        self.encoding = encoding
        self.sample_rows = sample_rows
        self.random_state = random_state

    # =========================
    # Public entrypoints
    # =========================
    def execute(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        return self.run(ctx, ap)

    def run(self, ctx: RunContext, ap: ArtifactPaths) -> StageResult:
        run_dir = Path(ctx.artifact_root)
        print(f"[START] {self.stage} started | run_dir={run_dir}")

        try:
            artifacts = self._execute(ctx, ap)

            print(f"[SUCCESS] {self.stage} completed | outputs={len(artifacts.output_files)}")

            return StageResult.success(
                stage=self.stage,
                outputs=artifacts.output_files + [str(artifacts.manifest_path)],
                metrics={
                    "run_dir": str(artifacts.run_dir),
                    "output_dir": str(artifacts.output_dir),
                    "manifest_path": str(artifacts.manifest_path),
                    "output_files": artifacts.output_files,
                    "n_outputs": len(artifacts.output_files),
                },
            )

        except Exception as e:
            print(f"[ERROR] {self.stage} failed | run_dir={run_dir} | error={e}")
            raise

    # =========================
    # Core execution
    # =========================
    def _execute(self, ctx: RunContext, ap: ArtifactPaths) -> PrepApplyArtifacts:
        run_dir = Path(ctx.artifact_root)
        prep_reco_dir = Path(ap.prep_reco_dir())
        model_match_dir = Path(ap.stage_dir("model_match_v1"))
        feature_dir = Path(ap.feature_table_dir())
        output_dir = Path(ap.stage_dir(self.stage))
        output_dir.mkdir(parents=True, exist_ok=True)

        contract_path = prep_reco_dir / "data_contract.json"
        plan_path = prep_reco_dir / "preprocessing_plan.json"
        match_path = model_match_dir / "model_matching_result.json"

        for required in [contract_path, plan_path, match_path]:
            if not required.exists():
                raise FileNotFoundError(f"Required input file not found: {required}")

        contract = self._read_json(contract_path)
        plan = self._read_json(plan_path)
        match = self._read_json(match_path)
        pol = self._get_match_policies(match)

        model_id = pol["model_id"]
        column_map = pol["column_map"]
        missing_policy = pol["missing_policy"]
        constraints = pol["constraints"]
        date_policy = pol["date_policy"]

        csv_inventory = self._list_csvs_recursive(feature_dir)
        deny_pats = self._leakage_deny_patterns(plan)

        canonical_ticker_name = (
            (plan.get("join_key_requirements_global") or {})
            .get("canonical_ticker_name", "종목코드")
        )
        canon_steps = (
            (plan.get("join_key_requirements_global") or {})
            .get("canonicalization", [])
        )

        tables_plan = plan.get("tables") or {}
        output_files: List[str] = []

        self._log(f"[INFO] stage={self.stage}")
        self._log(f"[INFO] model_id={model_id}")
        self._log(f"[INFO] run_dir={run_dir}")
        self._log(f"[INFO] prep_reco_dir={prep_reco_dir}")
        self._log(f"[INFO] model_match_dir={model_match_dir}")
        self._log(f"[INFO] feature_dir={feature_dir}")
        self._log(f"[INFO] output_dir={output_dir}")

        for table_key, table_cfg in tables_plan.items():
            out_file = self._process_single_table(
                table_key=table_key,
                table_cfg=table_cfg,
                contract=contract,
                plan=plan,
                csv_inventory=csv_inventory,
                deny_pats=deny_pats,
                model_id=model_id,
                column_map=column_map,
                missing_policy=missing_policy,
                constraints=constraints,
                date_policy=date_policy,
                canonical_ticker_name=canonical_ticker_name,
                canon_steps=canon_steps,
                output_dir=output_dir,
            )
            if out_file is not None:
                output_files.append(str(out_file))

        manifest = {
            "stage": self.stage,
            "model_id": model_id,
            "run_dir": str(run_dir),
            "prep_reco_dir": str(prep_reco_dir),
            "model_match_dir": str(model_match_dir),
            "feature_dir": str(feature_dir),
            "output_dir": str(output_dir),
            "output_files": output_files,
            "n_outputs": len(output_files),
        }

        manifest_path = output_dir / "prep_apply_result.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self._log("[DONE]")

        return PrepApplyArtifacts(
            run_dir=run_dir,
            output_dir=output_dir,
            manifest_path=manifest_path,
            output_files=output_files,
        )

    def _process_single_table(
        self,
        table_key: str,
        table_cfg: Dict[str, Any],
        contract: Dict[str, Any],
        plan: Dict[str, Any],
        csv_inventory: Dict[str, str],
        deny_pats: List[re.Pattern],
        model_id: str,
        column_map: Dict[str, Any],
        missing_policy: Dict[str, Any],
        constraints: Dict[str, Any],
        date_policy: Dict[str, Any],
        canonical_ticker_name: str,
        canon_steps: List[str],
        output_dir: Path,
    ) -> Optional[Path]:
        resolved = self._resolve_file_for_table(table_key, csv_inventory)
        if resolved is None:
            self._log(f"[SKIP] {table_key} (no input file)")
            return None

        if resolved.startswith("__MULTI__:"):
            files = resolved.replace("__MULTI__:", "").split("|")
            df = self._load_finance_multi(files, csv_inventory, plan)
            input_name = f"MULTI({len(files)})"
        else:
            df = self._read_csv_flexible(csv_inventory[resolved])
            input_name = resolved

        if df.empty:
            self._log(f"[WARN] {table_key} empty ({input_name})")
            return None

        t_contract = self._get_contract_table(contract, table_key)

        roles_plan = table_cfg.get("roles") or {}
        roles_contract = self._contract_role_cols(t_contract)

        is_news_raw = "news.news_raw_merged" in (table_key or "").lower()
        time_col = roles_plan.get("time_col") or (roles_contract.get("time_cols", [None])[0])
        ticker_col = roles_plan.get("ticker_col") or (roles_contract.get("ticker_cols", [None])[0])
        id_cols = roles_plan.get("id_cols") or roles_contract.get("id_cols", [])

        protected = {c for c in id_cols if c in df.columns}
        if time_col and time_col in df.columns:
            protected.add(time_col)
        if ticker_col and ticker_col in df.columns:
            protected.add(ticker_col)

        drop_by_contract = [
            c for c in self._contract_model_use_false_cols(t_contract)
            if c in df.columns and c not in protected
        ]
        if drop_by_contract:
            df.drop(columns=drop_by_contract, inplace=True, errors="ignore")

        if ticker_col and ticker_col in df.columns and ("zero_pad_to_6_if_all_digits" in canon_steps):
            df[ticker_col] = self._canonicalize_ticker(df[ticker_col])
        elif canonical_ticker_name in df.columns:
            df[canonical_ticker_name] = self._canonicalize_ticker(df[canonical_ticker_name])
            protected.add(canonical_ticker_name)

        if time_col and time_col in df.columns:
            match_actions = ((column_map.get(time_col) or {}).get("actions") or [])
            match_requires_dt = ("to_datetime" in match_actions) or ("date_floor" in match_actions)

            year_like = self._contract_time_is_year_like(t_contract, time_col)
            plan_finance_year = (
                "finance.financial_features" in (table_key or "").lower()
                and time_col.lower() == "year"
            )

            if match_requires_dt:
                df[time_col] = self._parse_and_normalize_date(
                    df[time_col],
                    assume_utc=True,
                    floor_to_day=(not is_news_raw),
                )
                df = df.dropna(subset=[time_col])
            else:
                if (not year_like) and (not plan_finance_year):
                    df[time_col] = self._parse_and_normalize_date(
                        df[time_col],
                        assume_utc=True,
                        floor_to_day=(not is_news_raw),
                    )
                    df = df.dropna(subset=[time_col])
                else:
                    df = df.dropna(subset=[time_col])

        plan_leak = [
            c for c in df.columns
            if self._is_leakage_col_by_pattern(c, deny_pats) and c not in protected
        ]
        if plan_leak:
            df.drop(columns=plan_leak, inplace=True, errors="ignore")

        self._enforce_constraints(df, constraints, protected)
        encode_cols = self._apply_column_map(df, column_map, protected)

        miss_req = ((table_cfg.get("requirements") or {}).get("missingness") or {})
        explicit_high = [
            c for c in (miss_req.get("high_missing_cols") or [])
            if c in df.columns and c not in protected
        ]
        if explicit_high:
            df.drop(columns=explicit_high, inplace=True, errors="ignore")

        drop_row_th = float(missing_policy.get("drop_row_threshold", 0.7))
        drop_col_th = float(missing_policy.get("drop_col_threshold", 0.9))
        self._drop_high_missing_rows(df, threshold=drop_row_th)
        self._drop_high_missing_cols(df, threshold=drop_col_th, protected=protected)

        numeric_cols, cat_cols, _ = self._split_cols(df)
        skip_impute = {
            c for c in (missing_policy.get("id_columns_never_impute") or []) if c in df.columns
        } | (protected & set(df.columns))

        per_type = missing_policy.get("per_type") or {}
        num_policy = str(per_type.get("numeric", ""))
        cat_policy = str(per_type.get("categorical", ""))

        if "median_impute" in num_policy:
            self._median_impute(df, numeric_cols, skip_cols=skip_impute)
        if ("missing_token" in cat_policy) or ("most_frequent_or_missing_token" in cat_policy):
            self._missing_token_impute(df, cat_cols, skip_cols=skip_impute, token="__MISSING__")

        dedup_keys_match = [k for k in (date_policy.get("dedup_keys") or []) if k in df.columns]
        dedup_keys_plan = [
            k for k in ((((table_cfg.get("requirements") or {}).get("join_keys") or {}).get("unique_key_suggestion")) or [])
            if k in df.columns
        ]

        if len(dedup_keys_match) >= 2:
            dedup_keys = dedup_keys_match
        elif len(dedup_keys_plan) >= 2:
            dedup_keys = dedup_keys_plan
        else:
            dedup_keys = [c for c in [ticker_col, time_col] if c and c in df.columns]

        if not is_news_raw:
            self._dedup_keep_last(df, dedup_keys)

        encode_cols = [
            c for c in encode_cols
            if c in df.columns and c not in protected and c != time_col
        ]
        df_out = self._one_hot_encode_strict(df, encode_cols)

        safe_key = table_key.replace(".", "__")
        out_path = output_dir / f"preprocessed__{model_id}__{safe_key}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

        if time_col and time_col in df_out.columns:
            self._log(
                f"[OK] {table_key} -> {out_path.name} | "
                f"rows={len(df_out)} cols={df_out.shape[1]} | "
                f"{time_col}.dtype={df_out[time_col].dtype}"
            )
        else:
            self._log(f"[OK] {table_key} -> {out_path.name} | rows={len(df_out)} cols={df_out.shape[1]}")

        return out_path

    # =========================
    # Utility methods
    # =========================
    def _log(self, msg: str) -> None:
        print(msg)

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_dictionary_file(self, fn: str) -> bool:
        f = fn.lower()
        return ("dictionary" in f) or f.endswith("_dictionary.csv") or f.endswith("_feature_dictionary.csv")

    def _list_csvs_recursive(self, root_dir: Path) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for fp in root_dir.rglob("*.csv"):
            if self._is_dictionary_file(fp.name):
                continue
            out[fp.name] = str(fp)
        return out

    def _read_csv_flexible(self, path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        encs = [self.encoding, "utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
        last_err = None
        for enc in encs:
            try:
                return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed reading CSV: {path} ({last_err})")

    def _canonicalize_ticker(self, series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        mask = s.str.fullmatch(r"\d+")
        s.loc[mask] = s.loc[mask].str.zfill(6)
        return s

    def _parse_and_normalize_date(
        self,
        s: pd.Series,
        assume_utc: bool = True,
        floor_to_day: bool = True,
    ) -> pd.Series:
        dt = pd.to_datetime(s, errors="coerce", utc=assume_utc)
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")

        if floor_to_day:
            try:
                dt = dt.dt.floor("D")
            except Exception:
                pass
        return dt

    def _leakage_deny_patterns(self, plan: dict) -> List[re.Pattern]:
        pats = (plan.get("leakage_policy", {}) or {}).get("deny_name_patterns", []) or []
        return [re.compile(p) for p in pats]

    def _is_leakage_col_by_pattern(self, col: str, deny_pats: List[re.Pattern]) -> bool:
        return any(p.search(col) for p in deny_pats)

    def _split_cols(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        dt = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        cat = [c for c in df.columns if c not in numeric and c not in dt]
        return numeric, cat, dt

    def _one_hot_encode_strict(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df
        return pd.get_dummies(df, columns=cols, dummy_na=True, drop_first=False)

    def _median_impute(self, df: pd.DataFrame, numeric_cols: List[str], skip_cols: set) -> None:
        for c in numeric_cols:
            if c in skip_cols:
                continue
            m = df[c].median(skipna=True)
            if pd.isna(m):
                continue
            df[c] = df[c].fillna(m)

    def _missing_token_impute(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        skip_cols: set,
        token: str = "__MISSING__",
    ) -> None:
        for c in cat_cols:
            if c in skip_cols:
                continue
            df[c] = df[c].astype(object).fillna(token)

    def _drop_high_missing_cols(self, df: pd.DataFrame, threshold: float, protected: set) -> None:
        miss = df.isna().mean()
        to_drop = [c for c, r in miss.items() if (r >= threshold and c not in protected)]
        if to_drop:
            df.drop(columns=to_drop, inplace=True, errors="ignore")

    def _drop_high_missing_rows(self, df: pd.DataFrame, threshold: float) -> None:
        if df.empty:
            return
        row_miss = df.isna().mean(axis=1)
        df.drop(index=df.index[row_miss >= threshold], inplace=True)

    def _dedup_keep_last(self, df: pd.DataFrame, keys: List[str]) -> None:
        keys = [k for k in keys if k in df.columns]
        if not keys:
            return
        df.sort_values(keys, inplace=True, kind="mergesort")
        df.drop_duplicates(subset=keys, keep="last", inplace=True)

    def _extract_ticker_from_filename(self, fn: str) -> Optional[str]:
        m = re.search(r"(^|\D)(\d{6})(\D|$)", fn)
        if not m:
            return None
        return m.group(2)

    # =========================
    # Contract helpers
    # =========================
    def _norm_table_key(self, k: str) -> str:
        return (k or "").strip().lower().replace("__", ".").replace(" ", "")

    def _get_contract_table(self, contract: dict, table_key: str) -> Dict[str, Any]:
        tables = contract.get("tables") or {}
        if not isinstance(tables, dict):
            return {}

        tk = self._norm_table_key(table_key)

        if table_key in tables:
            return tables[table_key] or {}

        for k, v in tables.items():
            if self._norm_table_key(k) == tk:
                return v or {}

        suffix = tk.split(".")[-1]
        for k, v in tables.items():
            if self._norm_table_key(k).endswith(suffix):
                return v or {}

        return {}

    def _get_contract_columns(self, table_contract: Dict[str, Any]) -> Dict[str, Any]:
        if not table_contract or not isinstance(table_contract, dict):
            return {}

        cols = table_contract.get("columns")
        if isinstance(cols, dict):
            return cols
        if isinstance(cols, list):
            out = {}
            for it in cols:
                if isinstance(it, dict) and it.get("name"):
                    out[str(it["name"])] = it
            return out

        schema = table_contract.get("schema")
        if isinstance(schema, list):
            out = {}
            for it in schema:
                if isinstance(it, dict) and it.get("name"):
                    out[str(it["name"])] = it
            return out

        if isinstance(schema, dict):
            cols2 = schema.get("columns")
            if isinstance(cols2, dict):
                return cols2
            if isinstance(cols2, list):
                out = {}
                for it in cols2:
                    if isinstance(it, dict) and it.get("name"):
                        out[str(it["name"])] = it
                return out

        return {}

    def _contract_model_use_false_cols(self, table_contract: Dict[str, Any]) -> List[str]:
        cols = self._get_contract_columns(table_contract)
        out = []

        for c, spec in cols.items():
            if not isinstance(spec, dict):
                continue

            mu = spec.get("model_use", None)
            dm = spec.get("dictionary_meta") or {}
            if isinstance(dm, dict):
                if mu is None:
                    mu = dm.get("model_use", None)
                notes = str(dm.get("notes", "")).lower()
            else:
                notes = ""

            if isinstance(mu, str):
                mu_norm = mu.strip().lower()
                if mu_norm == "false":
                    out.append(c)
                    continue
                if mu_norm == "true":
                    continue

            if mu is False:
                out.append(c)
                continue

            if "model_use=false" in notes:
                out.append(c)

        return out

    def _contract_role_cols(self, table_contract: Dict[str, Any]) -> Dict[str, List[str]]:
        if not table_contract or not isinstance(table_contract, dict):
            return {}
        roles = table_contract.get("roles") or {}
        if not isinstance(roles, dict):
            return {}

        out = {}
        if isinstance(roles.get("id_cols"), list):
            out["id_cols"] = roles["id_cols"]

        tc = roles.get("time_col")
        if isinstance(tc, str) and tc:
            out["time_cols"] = [tc]

        tk = roles.get("ticker_col")
        if isinstance(tk, str) and tk:
            out["ticker_cols"] = [tk]

        return out

    def _contract_time_is_year_like(self, table_contract: Dict[str, Any], time_col: str) -> bool:
        if not time_col:
            return False

        cols = self._get_contract_columns(table_contract)
        spec = cols.get(time_col, {})
        if not isinstance(spec, dict):
            spec = {}

        dtype = str(spec.get("dtype", "")).lower()
        dm = spec.get("dictionary_meta") or {}
        if not isinstance(dm, dict):
            dm = {}

        notes = str(dm.get("notes", "")).lower()
        desc_ko = str(dm.get("description_ko", "")).lower()

        if time_col.lower() == "year" and ("int" in dtype):
            return True
        if "semantic_type=year" in notes or "semantic_type:year" in notes:
            return True
        if "year" in notes and "semantic_type" in notes:
            return True
        if "연도" in desc_ko:
            return True

        return False

    # =========================
    # Resolver / policy helpers
    # =========================
    def _resolve_file_for_table(self, table_key: str, csv_inventory: Dict[str, str]) -> Optional[str]:
        k = table_key.lower()
        candidates = []

        if "price.ohlcv_last365" in k:
            candidates = [fn for fn in csv_inventory if "ohlcv" in fn.lower()]
        elif "price.foreign_snapshot_today" in k:
            candidates = [fn for fn in csv_inventory if "foreign" in fn.lower()]
        elif "news.news_raw_merged" in k:
            candidates = [
                fn for fn in csv_inventory
                if ("news_raw" in fn.lower() or "news_merged" in fn.lower() or "naver_news_merged" in fn.lower())
                and "dictionary" not in fn.lower()
            ]
        elif "news.news_features_by_stock" in k:
            candidates = [fn for fn in csv_inventory if "news_features" in fn.lower()]
        elif "user.user_snapshot" in k:
            candidates = [fn for fn in csv_inventory if "user_features" in fn.lower() or "user_snapshot" in fn.lower()]
        elif "finance.financial_features" in k:
            candidates = [fn for fn in csv_inventory if ("finance_features" in fn.lower() or "financial_features" in fn.lower())]

        if not candidates:
            frag = table_key.split(".")[-1].lower()
            candidates = [fn for fn in csv_inventory if frag in fn.lower()]

        if not candidates:
            return None

        if "finance.financial_features" in k:
            return "__MULTI__:" + "|".join(sorted(candidates))

        candidates = sorted(candidates, key=lambda x: (len(x), x))
        return candidates[0]

    def _load_finance_multi(self, files: List[str], csv_inventory: Dict[str, str], plan: dict) -> pd.DataFrame:
        dfs = []
        plan_text = json.dumps(plan, ensure_ascii=False)
        enable_filename_ticker = "finance_filename_ticker_note" in plan_text

        for fn in files:
            d = self._read_csv_flexible(csv_inventory[fn])

            if enable_filename_ticker and "종목코드" not in d.columns:
                t = self._extract_ticker_from_filename(fn)
                if t is not None:
                    d["종목코드"] = t

            dfs.append(d)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _get_match_policies(self, match: dict) -> Dict[str, Any]:
        model_id = ((match.get("selected_model") or {}).get("model_id")) or "unknown_model"
        return {
            "model_id": model_id,
            "column_map": match.get("column_preprocessing_map") or {},
            "missing_policy": match.get("missing_value_policy") or {},
            "constraints": match.get("constraints") or {},
            "date_policy": match.get("date_policy") or {},
        }

    def _enforce_constraints(self, df: pd.DataFrame, constraints: Dict[str, Any], protected: set) -> None:
        if constraints.get("drop_company_name_always", False):
            for cand in ["종목명", "company_name", "name"]:
                if cand in df.columns and cand not in protected:
                    df.drop(columns=[cand], inplace=True, errors="ignore")

    def _apply_column_map(
        self,
        df: pd.DataFrame,
        column_map: Dict[str, Any],
        protected: set,
    ) -> List[str]:
        leakage_drop_cols = []
        encode_cols = []

        for col, spec in (column_map or {}).items():
            if col not in df.columns:
                continue
            actions = spec.get("actions") or []
            keep = spec.get("keep", True)

            if "to_datetime" in actions or "date_floor" in actions:
                df[col] = self._parse_and_normalize_date(df[col], assume_utc=True)

            if (not keep) or ("drop_column" in actions):
                if col not in protected:
                    df.drop(columns=[col], inplace=True, errors="ignore")
                continue

            if "drop_or_mask_leakage" in actions and col not in protected:
                leakage_drop_cols.append(col)

            if "encoding" in actions and col not in protected:
                encode_cols.append(col)

        if leakage_drop_cols:
            df.drop(
                columns=[c for c in leakage_drop_cols if c in df.columns],
                inplace=True,
                errors="ignore",
            )

        return sorted(list(set(encode_cols)))