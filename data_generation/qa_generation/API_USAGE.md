# API usage

Supply GEMINI_API_KEY or GOOGLE_API_KEY through the environment. The release uses the shared requests-based Gemini REST adapter, not hard-coded tokens or global SDK key rotation. Select --model explicitly. The original prompt is in prompt.py. Failed/blocked/incomplete outputs are recorded as errors and retried on an identical rerun, not saved as research answers. See ../../docs/BUILD_DATASET.md for commands and ../README.md for the stage map.
