# Gold-answer validation

```bash
export GEMINI_API_KEY="<set-this-locally-not-in-GitHub>"
python data_generation/dataset_validation/validate_gold.py \
  --input data_generation/work/rag_benchmark_clean_new.json \
  --output data_generation/work/rag_benchmark_gold_validated_new.json \
  --errors data_generation/work/gold_validation_errors.json
```
