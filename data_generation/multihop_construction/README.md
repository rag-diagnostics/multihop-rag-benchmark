# Multi-hop construction and quality filtering


## Run (uses Gemini API; does not require uploading generated data)

```bash
export GEMINI_API_KEY="<set-this-locally-not-in-GitHub>"
python data_generation/multihop_construction/multi_create.py \
  --input data_generation/work/rag_benchmark_gold_validated_new.json \
  --global-docs /path/to/updated_modified_final_dataset_6_02.xlsx \
  --output data_generation/work/000_new_multihop_dataset_final.json \
  --errors data_generation/work/000_new_multihop_errors.json
```



## Remove failed hop chains

Use `filter_multihop.py`

```bash
python -c 'from data_generation.multihop_construction.filter_multihop import filter_json_remove_entries_with_hops_error; filter_json_remove_entries_with_hops_error("data_generation/work/000_new_multihop_dataset_final.json", "data_generation/work/filtered_multihop.json")'
```

