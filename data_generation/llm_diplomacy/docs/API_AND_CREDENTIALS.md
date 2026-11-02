# APIs, local endpoints, Hugging Face, and secrets

This is a separate guide because **local model serving** and **paid hosted APIs** are not interchangeable.

## Which services were actually used in these game-generation scripts?

| Service | Evidence in `llm_dip.zip` | Required for the archived games? |
| --- | --- | --- |
| **vLLM local OpenAI-compatible HTTP API** | The SLURM scripts start local servers on ports `8000` and `8001`; model config lines use `openai:MODEL@http://HOST:PORT/v1`. | **Yes.** This is the interface used to call Llama and Qwen. |
| **Hugging Face Hub** | Server scripts load model weights from `meta-llama/...` and `Qwen/...`, and attempt token authentication with `HF_TOKEN`. | **For model access/downloads**, subject to model permissions. No per-token inference API is necessary when served locally. |
| **OpenAI paid API** | The local client reuses the OpenAI SDK and its request schema. No evidence of paid OpenAI calls in the selected simulation scripts. | **No** for those local game runs. |
| **Gemini API** | Upstream includes Gemini clients and optional phase-summary/formatting facilities; the ECIR paper describes a separate later Gemini 2.5 Pro conversion stage. | **Not for these archived game runs by default.** That later narrative-conversion script has not been found. |
| **OpenRouter, Anthropic, DeepSeek, Together** | Listed in upstream client support and `.env.example` / `config.py`. | **Optional upstream capabilities**, not evidence that they were called in this experiment. |

## Local vLLM: OpenAI-compatible does not mean OpenAI-hosted

The model spec uses the upstream client's prefix syntax:

```text
openai:meta-llama/Llama-3.1-8B-Instruct@http://127.0.0.1:8000/v1
openai:Qwen/Qwen2.5-7B-Instruct@http://127.0.0.1:8001/v1
```

`openai:` means the **OpenAI-compatible client implementation**. `@http://127.0.0.1:...` routes requests to your own model server, not to `api.openai.com`. The model server must advertise a matching model ID at `/v1/models`.

The upstream `OpenAIClient` constructor specifically requires either a supplied inline key or `OPENAI_API_KEY` from the environment, even for a localhost endpoint. When running an unauthenticated local server:

```bash
export OPENAI_API_KEY=local-vllm-placeholder
```

This is a dummy string, not a secret. Do **not** use your actual paid OpenAI key for local vLLM servers. The historical model configurations included `#not_needed`, a placeholder inline key; this reconstructed package avoids inline keys entirely.

## Hugging Face authentication

- Confirm the relevant model licence/access terms before downloading or running each model.
- For gated Hugging Face models, set `HF_TOKEN` in your secure shell/cluster secret environment if required. Do **not** add it to source files or batch logs.
- Use your Hugging Face-supported login/environment mechanism; do not paste tokens into documentation or command-line arguments.
- Local inference on your own GPU does not require a commercial Hugging Face inference API subscription just because weights came from Hugging Face.

## Hosted API alternatives (not established for the historical game batch)

The upstream project also supports provider-specific environment variables:

```dotenv
# EXAMPLE ONLY: fill only if you deliberately use a hosted provider
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
OPENROUTER_API_KEY=
TOGETHER_API_KEY=
HF_TOKEN=
```

Use these only for the matching provider and client. Hosted requests can incur charges and may send prompt/game content to external services. Consult each provider's current endpoint and data-use terms. Do not confuse this option with the two local vLLM endpoints used in the archived experiment.

### Optional features that can trigger extra provider calls

- `--generate_phase_summaries true`: by default **false**. The archived upstream `narrative.py` can call a configured narrative model (its archived default is an OpenRouter-served Gemini variant) to create short phase summaries. This was **not** enabled in the selected game-launch scripts.
- `--use_unformatted_prompts true`: by default **false**. The archived documentation describes an additional Gemini formatting call. It was **not** enabled in those game-launch scripts.
- The later **Gemini 2.5 Pro game-to-narrative transformation** mentioned in the ECIR paper is **not identified** in the two ZIP folders and must be documented separately.

## Security checklist

- Never commit `.env`, API keys, Hugging Face tokens, SLURM secrets or provider credentials.
- Avoid putting tokens after `#` in model spec strings; these can end up in command histories, process lists, logs, and exported configurations.
- Do not log full confidential prompts or responses to a public repository.
- Do not bind vLLM to `0.0.0.0` on a shared cluster without access controls; the original job did this but a portable public example should use localhost.
- Do not treat cluster DNS names or personal paths from the archive as runnable configuration.
