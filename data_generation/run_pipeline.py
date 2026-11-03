"""Orchestrate dataset construction from selected message records using the supplied techniques.

The released 528-chain benchmark does NOT need this pipeline. New source generation
will not reproduce a historical random draw or identical stochastic model outputs.
"""
from __future__ import annotations
import argparse
import asyncio
import importlib.util
from pathlib import Path
from rag_utils.io import read_json, atomic_json, fingerprint, require_new_manifest, output_lock, file_sha256
from rag_utils.tables import read_records
from rag_utils.gemini import api_key, Model


def build(input_path, workdir, *, narrative_model="gemini-2.5-flash", qa_model="gemini-2.5-pro",
          validation_model="gemini-2.0-flash", multihop_model="gemini-2.5-flash", seed=42,
          pks_model=None, calibration_dir=None, hooks=None):
    """Sequential stage boundaries. hooks is exclusively for tests, not a fake CLI mode."""
    from .narrative_generation.generate_narratives import run as narrate
    from .entity_remapping.remap_entities import process_excel_file as remap
    from .qa_generation.generate_qa import run as qa
    from .schema_adapters.prepare_distractor_input import prepare
    from .distractor_selection.select_random_distractors import select_for_row, check_candidate
    from .dataset_validation import validate_gold
    from .multihop_construction import multi_create
    from .multihop_construction.filter_multihop import filter_entries
    hooks = hooks or {}; wd = Path(workdir); wd.mkdir(parents=True, exist_ok=True)
    sources = read_records(input_path)
    if pks_model and not calibration_dir: raise ValueError("PKS requires the reference calibration directory before model calls")
    if len(sources) < 11: raise ValueError("At least 11 source documents are required, and more may be needed after screening")
    code_paths = [Path(__file__), *Path(__file__).parent.rglob("*.py"), *Path(__file__).parents[1].joinpath("rag_utils").glob("*.py")]
    code_identity = fingerprint({str(p.relative_to(Path(__file__).parents[1])): file_sha256(p) for p in code_paths if "tests" not in p.parts})
    identity = {"source": fingerprint(sources), "code_sha256": code_identity, "seed": seed, "models": [narrative_model, qa_model, validation_model, multihop_model],
                "pks_model": pks_model, "calibration_dir": str(calibration_dir), "pipeline_version": "corrected-construction-v1"}
    with output_lock(wd / "pipeline"):
        require_new_manifest(wd / "pipeline_manifest.json", identity)
        p_narr, p_remap, p_qa, p_docs = (wd / n for n in ("narratives.json", "transformed.json", "qa.json", "documents_qa.json"))
        hooks.get("narrate", narrate)(sources, p_narr, model=narrative_model)
        hooks.get("remap", remap)(p_narr, p_remap, seed=seed)
        hooks.get("qa", qa)(p_remap, p_qa, model=qa_model)
        prepare(p_qa, p_docs)
        rows = read_records(p_docs); docs = list(dict.fromkeys(r["Document"] for r in rows))
        if len(docs) < 11: raise ValueError("Need at least 11 distinct transformed documents")
        if len({r["question"] for r in rows}) != len(rows): raise ValueError("Duplicate questions; no question-key overwrite allowed")
        key = None if hooks else api_key()
        selected = {}; audits = []
        for row in rows:
            q, item, audit = select_for_row(row, docs, 10, seed, "llm", key, validation_model, check_fn=hooks.get("check_candidate", check_candidate))
            audits.append(audit)
            if item: selected[q] = item
        atomic_json(wd / "distractor_audit.json", audits)
        atomic_json(wd / "base_with_distractors.json", selected)
        if any(a["api_or_parse_failures"] for a in audits): raise RuntimeError("Distractor checking had API errors; rerun after fixing them")
        if not selected: raise RuntimeError("No questions have ten safe distractors; expand the source corpus")
        if not hooks:
            validate_gold.API_KEYS = [key]; validate_gold.MODEL_NAME = validation_model
        validate_fn = hooks.get("validate_gold", validate_gold.validate_gold_with_answer_list)
        async def validate_all():
            passed = {}; failures = []
            for q, item in selected.items():
                result = await validate_fn(q, item["golden_document_text"], item["golden_answer"])
                if result.get("ok"):
                    passed[q] = dict(item, gold_validation_matched_answer=result.get("matched_answer"), gold_validation_justification=result.get("justification"))
                else: failures.append({"question": q, "validation": result})
            return passed, failures
        gold, gold_errors = asyncio.run(validate_all())
        atomic_json(wd / "gold_validated.json", gold); atomic_json(wd / "gold_validation_errors.json", gold_errors)
        if any(e["validation"].get("reason") in {"api_error", "json_parse_error", "invalid_schema", "bad_schema_verdict"} for e in gold_errors):
            raise RuntimeError("Gold validation failed operationally; do not continue")
        if not gold: raise RuntimeError("No valid base QA pairs")
        model = hooks.get("multihop_model_object") or Model(multihop_model, key=key)
        chain_fn = hooks.get("process_chain", multi_create.process_chain)
        async def chains_all():
            result = []
            for q, item in gold.items(): result.append(await chain_fn(q, item, docs, model))
            return result
        chains = asyncio.run(chains_all()); atomic_json(wd / "multihop_unfiltered.json", chains)
        valid, rejected = filter_entries(chains, family="random", strict=True)
        atomic_json(wd / "multihop_validated.json", valid); atomic_json(wd / "multihop_rejected.json", rejected)
        if not valid: raise RuntimeError("No complete valid chains; inspect multihop_rejected.json and expand sources")
        status = {"source_samples": len(sources), "safe_base_contexts": len(selected), "gold_validated": len(gold),
                  "complete_chains_before_pks": len(valid), "pks_applied": False, "final_dataset": None}
        if pks_model:
            if not calibration_dir: raise ValueError("PKS requires the reference calibration directory")
            location = Path(__file__).resolve().parents[1] / "parametric_filtering/pks_filtering/00_para14.py"
            spec = importlib.util.spec_from_file_location("pks_filter", location); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module.run(wd / "multihop_validated.json", pks_model, Path(calibration_dir), wd / "pks", "auto", "error")
            kept = read_json(wd / "pks/000_abl_filtered_unknowns.json")
            if not kept: raise RuntimeError("PKS retained no chains")
            atomic_json(wd / "benchmark.json", kept)
            status.update(pks_applied=True, retained_chains=len(kept), final_dataset="benchmark.json")
        else:
            status["next_stage"] = "PKS calibration and filtering are required before calling this a final filtered benchmark"
        atomic_json(wd / "pipeline_summary.json", status)
        return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Selected messages JSON (Serial, text, optional source label)")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--narrative-model", default="gemini-2.5-flash")
    p.add_argument("--qa-model", default="gemini-2.5-pro")
    p.add_argument("--validation-model", default="gemini-2.0-flash")
    p.add_argument("--multihop-model", default="gemini-2.5-flash")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pks-model"); p.add_argument("--calibration-dir", type=Path)
    a = p.parse_args()
    build(a.input, a.workdir, narrative_model=a.narrative_model, qa_model=a.qa_model,
          validation_model=a.validation_model, multihop_model=a.multihop_model, seed=a.seed,
          pks_model=a.pks_model, calibration_dir=a.calibration_dir)

if __name__ == "__main__": main()
