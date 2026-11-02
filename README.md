# Multi-Hop RAG Benchmark

**A controllable, re-instantiable benchmark for diagnosing reasoning failures in the generator stage of retrieval-augmented generation (RAG).**

The benchmark measures how language models use evidence distributed across multiple documents while controlling reasoning depth, document placement and distractor composition. In the principal evaluation setting, retrieval is held fixed so that errors can be analysed at the generation stage.

## Benchmark design

- **Source corpus:** 600 Diplomacy-derived narratives, constructed from 300 randomly selected LLM-versus-LLM game samples and 300 samples from the [QANTA Diplomacy corpus](https://sites.google.com/view/qanta/projects/diplomacy) (Peskov et al., 2020).
- **Question answering:** short, document-grounded factual answers, with supporting evidence and intermediate bridge entities.
- **Compositional reasoning:** controlled questions requiring one through four evidence documents.
- **Context size:** 11 documents per example, with 1–4 gold documents and 10–7 distractors.
- **Evidence placement:** A1–A3 for single-hop start/middle/end; B1–B3 for multi-hop canonical, reversed and randomised arrangements.
- **Re-instantiation:** entity and time transformations support generating variations of source narratives.

The companion research manuscript also describes parametric-answerability screening and generator evaluation across multiple language models. These stages are maintained separately from dataset construction.

## Repository structure and scope

The [`data_generation/`](data_generation/README.md) directory contains the dataset-construction modules: source preparation, narrative conversion, entity remapping, QA generation, random distractor selection, validation, multi-hop composition and context formation. Each module has its own usage guide.

Parametric-answerability screening (paper §3.4), generator evaluation (paper §4), and analysis (paper §5) are separate research stages, not part of the dataset-construction code. Generated datasets, model outputs, local credentials and large simulation logs are not committed as source code.

## Getting started

Start with the [dataset-construction guide](data_generation/README.md). Each module has its own README specifying inputs, dependencies, configuration and outputs. GPU-backed game simulation uses the upstream [GoodStartLabs AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) engine; Gemini-powered construction stages use credentials supplied through environment variables.



## Source attribution

The research builds on the QANTA Diplomacy dataset of Peskov et al. (2020) and the GoodStartLabs AI_Diplomacy game environment. Their respective source terms and software licences apply. The [`source_collection/webdiplomacy/`](data_generation/source_collection/webdiplomacy/README.md) utilities are supplementary and are **not** the source of the 300 QANTA examples in the benchmark.

## Research team

Syed Mahbubul Huq · Christopher Child · Tillman Weyde · Pranava Madhyastha

City St George's, University of London · The Alan Turing Institute

## Citation

For the project citation and author information, see [`CITATION.cff`](CITATION.cff). Cite the source datasets and game environment when using those components.
