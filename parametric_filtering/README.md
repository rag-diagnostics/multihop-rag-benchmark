# Parametric Knowledge Filtering

This directory implements the no-context Parametric Knowledge Score (PKS) calibration and instance filtering procedure described in Section 3.4 of **A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators**.

## Method

Given a question \(q\), its short answer \(a\), and summed token-level negative log-likelihoods \(L\) from the reference language model, PKS is

\[
\mathrm{PKS}(q,a)=L_q+L_a-L_{aq}.
\]

The same question-answer prompt format is used for each likelihood calculation, with the question and/or answer left blank as appropriate. No retrieved document context is supplied.

Scores are calibrated against three QA distributions:

1. **Known and plausible** (`common`)
2. **Unknown and plausible** (`plausible`)
3. **Unknown and implausible** (`nonsense`)

For each class, compute the mean \(\mu_c\) and population standard deviation \(\sigma_c\) of its PKS scores. Assign an instance to the class minimising \(\lvert(\mathrm{PKS}-\mu_c)/\sigma_c\rvert\). Remove instances assigned to `common` and retain those assigned to `plausible` or `nonsense`. Questions sharing an `original_question` are scored once and assigned the same selection decision across their multi-hop rows.

## Files

| File | Role |
|---|---|
| `baseline_calibration/para11.py` | Score the three calibration QA sets and write `pmi_classic` distributions |
| `pks_filtering/00_para14.py` | Score benchmark questions, apply the three-class decision, and produce retained/removed JSON |
| `validation/para19.py` | Independently re-score Qwen instances and compare classification decisions |
| `validation/para13.py` | Compare PKS calibration distributions across reference models |
| `pks_core.py` | Shared NLL, PKS, calibration and classification operations |
| `tests/` | Offline unit tests |

## Requirements

Python 3.10 or newer, PyTorch, Transformers, and Accelerate. Install a PyTorch build appropriate for the machine, then:

```bash
pip install -r parametric_filtering/requirements.txt
```

Model inference requires the reference model checkpoint. The main PKS reference is **Qwen2.5-7B-Instruct**. Supply the Hugging Face ID (`Qwen/Qwen2.5-7B-Instruct`) or the path to the locally stored checkpoint with `--model`.

## Required inputs

The three calibration files are:

```text
  7common_knowledge.json
  7plausible_but_false.json
  7nonsense_knowledge.json
```

These represent the three calibration categories in Section 3.4. Supply the research calibration samples to reproduce the paper's calibration statistics. The calibration QA data are **not distributed in this code-only package**.

The benchmark input must be a JSON array containing `original_question` and `answer` fields. Other fields—including hop definitions, gold evidence and distractors—are preserved exactly in the filtered JSON outputs.

## Run the PKS pipeline

Commands below are executed from the repository root. Store generated scores, model checkpoints and benchmark data outside the public repository or in an ignored working directory.

### 1. Compute calibration scores

```bash
python parametric_filtering/baseline_calibration/para11.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir calibration_data \
  --output-dir parametric_filtering/outputs/calibration
```

The script writes three baseline files under `outputs/calibration`, named:

- `7common_knowledge_scores_v2_no_ctx.json`
- `7plausible_but_false_scores_v2_no_ctx.json`
- `7nonsense_scores_v2_no_ctx.json`

Each file contains a `pmi_classic` score array used by the filter. A `calibration_summary.json` provides means, standard deviations and sample counts.

### 2. Filter benchmark questions

```bash
python parametric_filtering/pks_filtering/00_para14.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data_generation/work/000_new_multihop_dataset_final.json \
  --calibration-dir parametric_filtering/outputs/calibration \
  --output-dir parametric_filtering/outputs/filtering
```

The output directory contains:

- `000_abl_filtered_unknowns.json` — retained benchmark rows
- `000_abl_removed_knowns.json` — removed benchmark rows
- `000_abl_question_categories.csv` — one category per unique question
- `pks_question_decisions.json` — PKS scores, Z-distances and decisions
- `pks_filter_summary.json` — question and row counts

The original filtering behaviour keeps a question when inference fails (`--on-error keep`). Such cases are explicitly recorded in `pks_question_decisions.json`. To require every question to receive a valid PKS score, use `--on-error error`.

### 3. Independently verify the Qwen decisions

```bash
python parametric_filtering/validation/para19.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data_generation/work/000_new_multihop_dataset_final.json \
  --calibration-dir parametric_filtering/outputs/calibration \
  --compare-categories parametric_filtering/outputs/filtering/000_abl_question_categories.csv \
  --expected-total-questions 600 \
  --expected-removed-questions 72 \
  --output-dir parametric_filtering/outputs/verification
```

The expected-count arguments assert the paper's reported 600 unique questions and 72 removed questions (528 retained). Omit these arguments for another dataset or run. The validator re-runs model scoring and checks the category assignments rather than simply comparing filenames.

### 4. Reference-model calibration comparison (optional)

Provide a JSON object mapping readable model names to checkpoint paths or Hugging Face model IDs. For example:

```json
{
  "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
  "Granite-3.1-8B-Instruct": "/models/granite-3.1-8b-instruct",
  "Apertus-8B-2509": "/models/Apertus-8B-2509",
  "AFM-4.5B": "/models/AFM-4.5B",
  "Llama-3.2-3B": "/models/Llama-3.2-3B"
}
```

Save this mapping as `models.json` and run:

```bash
python parametric_filtering/validation/para13.py \
  --models-json models.json \
  --data-dir calibration_data \
  --primary-name Qwen/Qwen2.5-7B-Instruct \
  --primary-cache-dir parametric_filtering/outputs/calibration \
  --output-dir parametric_filtering/outputs/reference_models
```

Each reference model uses its own score distribution; Qwen's scores can be reused from Step 1. This optional script compares calibration distributions, not full instance-level filter agreement.

