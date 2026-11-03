# API configuration

The current construction modules use `rag_utils/gemini.py` with GEMINI_API_KEY or GOOGLE_API_KEY read from the environment and sent in an HTTP header. Use the [construction guide](../docs/BUILD_DATASET.md) and [change record](../docs/FIXES.md). This supersedes earlier Google SDK-specific setup directions. Keep source data and credentials outside Git.
