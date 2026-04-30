# Databricks notebook source
# MAGIC %md
# MAGIC # SSCD — Self-Hosted LLM on Databricks Model Serving
# MAGIC
# MAGIC End-to-end playbook: HuggingFace model → Unity Catalog → Databricks Model Serving → OpenAI-compatible API.
# MAGIC
# MAGIC | Requirement | Solution |
# MAGIC |---|---|
# MAGIC | HuggingFace LLM in UC | Model weights stored in a UC Volume as a governed object |
# MAGIC | vLLM inference engine | vLLM wrapped in MLflow pyfunc, deployed inside the serving container |
# MAGIC | Model Serving endpoint | GPU-backed managed endpoint — autoscaling, scale-to-zero |
# MAGIC | OpenAI API spec | OpenAI messages/tools schema over the invocations endpoint |
# MAGIC | Chat completions | messages format: `[{"role": "user", "content": "..."}]` |
# MAGIC | Function / tool calling | tools + tool_choice passed alongside messages |
# MAGIC
# MAGIC **Compute:** Run this notebook on any CPU cluster or Serverless. The Model Serving endpoint provisions its own GPU — no GPU cluster needed here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies

# COMMAND ----------

# MAGIC %pip install -q mlflow databricks-sdk openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration
# MAGIC
# MAGIC Set `UC_VOLUME_MODEL_PATH` to your model weights location in the UC Volume.
# MAGIC Everything else deploys automatically.

# COMMAND ----------

import json, mlflow, mlflow.pyfunc, pandas as pd
from databricks.sdk import WorkspaceClient

# Path to HuggingFace model weights already stored in UC Volume
UC_VOLUME_MODEL_PATH = "/Volumes/serverless_markospetko_catalog/vllm/model_weights/facebook_opt-125m"

UC_CATALOG          = "serverless_markospetko_catalog"
UC_SCHEMA           = "vllm"
UC_REGISTERED_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.sscd_llm"
ENDPOINT_NAME       = "sscd-debug-v1"

# Set to True to create/update the serving endpoint (cells 11-13).
# False (default) skips deployment and queries the existing live endpoint.
DEPLOY_ENDPOINT = False

w = WorkspaceClient()
print(f"Workspace : {w.config.host}")
print(f"Model path: {UC_VOLUME_MODEL_PATH}")
print(f"Endpoint  : {ENDPOINT_NAME}")
print(f"Deploy    : {DEPLOY_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Define the vLLM Serving Model
# MAGIC
# MAGIC This pyfunc wraps vLLM and is loaded inside the Model Serving container at deploy time.
# MAGIC It handles both chat completions and tool/function calling transparently.

# COMMAND ----------

# Simple chat template for base models that lack one (e.g., OPT-125m).
# vLLM's llm.chat() requires a Jinja2 chat_template on the tokenizer.
BASE_MODEL_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "Assistant:"
)

class VLLMPyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import os, traceback
        os.environ.setdefault("MASTER_ADDR",        "localhost")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        try:
            # Monkey-patch: vllm 0.8.5 accesses tokenizer.all_special_tokens_extended,
            # which was removed in transformers>=4.47.0. Re-add it so vllm can initialize.
            import transformers
            if not hasattr(transformers.PreTrainedTokenizerBase, "all_special_tokens_extended"):
                @property
                def _all_special_tokens_extended(self):
                    return list(set(self.all_special_tokens))
                transformers.PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

            from vllm import LLM
            self.llm = LLM(
                model=context.artifacts["model_dir"],
                dtype="half",
                enforce_eager=True,
                gpu_memory_utilization=0.90,
                max_num_seqs=32,
                swap_space=0,
            )

            # If the model's tokenizer has no chat template, inject a basic one
            # so that llm.chat() doesn't error on transformers>=4.44.
            tokenizer = self.llm.get_tokenizer()
            if not getattr(tokenizer, "chat_template", None):
                tokenizer.chat_template = BASE_MODEL_CHAT_TEMPLATE

            self._load_error = None
        except Exception:
            self._load_error = traceback.format_exc()
            self.llm = None

    def predict(self, context, model_input):
        import json as _json, time, uuid
        import pandas as pd

        if self._load_error:
            return pd.DataFrame({"response": [_json.dumps({"error": self._load_error})]})

        from vllm import SamplingParams

        row      = model_input.iloc[0]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = _json.loads(messages)

        tools = None
        if "tools" in model_input.columns and row["tools"] is not None:
            tools = row["tools"]
            if isinstance(tools, str):
                tools = _json.loads(tools)

        sp = SamplingParams(temperature=0.1, max_tokens=512)
        outputs = self.llm.chat(messages=messages, tools=tools, sampling_params=sp)
        output  = outputs[0].outputs[0]

        if hasattr(output, "tool_calls") and output.tool_calls:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id":       tc.tool_call_id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                } for tc in output.tool_calls],
            }
        else:
            msg = {"role": "assistant", "content": output.text.strip()}

        response = _json.dumps({
            "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        })
        return pd.DataFrame({"response": [response]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Register Model to Unity Catalog

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="sscd_vllm"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=VLLMPyfuncModel(),
        signature=ModelSignature(
            inputs=Schema([ColSpec("string", "messages")]),
            outputs=Schema([ColSpec("string", "response")]),
        ),
        artifacts={"model_dir": UC_VOLUME_MODEL_PATH},
        # Let vllm pull its own transformers version.
        # The monkey-patch in load_context re-adds the removed
        # all_special_tokens_extended attribute so vllm 0.8.5 works
        # with transformers>=4.47.0.
        pip_requirements=[
            "vllm==0.8.5.post1",
            "mlflow>=2.14.1",
        ],
    )
    registered = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name=UC_REGISTERED_MODEL,
    )

print(f"Registered: {registered.name} v{registered.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Deploy to Model Serving Endpoint
# MAGIC
# MAGIC Databricks provisions a GPU automatically. The endpoint:
# MAGIC - Scales to zero when idle (no cost)
# MAGIC - Autoscales under load
# MAGIC - Exposes an OpenAI-compatible REST API
# MAGIC
# MAGIC > First deployment takes ~20-30 minutes. Progress visible in the Serving UI.

# COMMAND ----------

import requests

token      = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url   = f"{w.config.host}/api/2.0/serving-endpoints"
invoke_url = f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations"

if not DEPLOY_ENDPOINT:
    print(f"DEPLOY_ENDPOINT is False — skipping create/update. Will query existing endpoint '{ENDPOINT_NAME}'.")
else:
    latest_version = max(
        w.model_versions.list(full_name=UC_REGISTERED_MODEL),
        key=lambda v: int(v.version),
    ).version
    print(f"Deploying model version: {latest_version}")

    config = {
        "served_entities": [{
            "entity_name":          UC_REGISTERED_MODEL,
            "entity_version":        str(latest_version),
            "workload_size":         "Small",
            "workload_type":         "GPU_SMALL",
            "scale_to_zero_enabled": True,
        }]
    }

    # Check if endpoint exists — create or update accordingly
    exists = requests.get(f"{base_url}/{ENDPOINT_NAME}", headers=headers).status_code == 200

    if exists:
        resp = requests.put(f"{base_url}/{ENDPOINT_NAME}/config", headers=headers, json=config)
        print(f"Updating endpoint '{ENDPOINT_NAME}' to v{latest_version}...")
    else:
        resp = requests.post(base_url, headers=headers, json={"name": ENDPOINT_NAME, "config": config})
        print(f"Creating endpoint '{ENDPOINT_NAME}'...")

    if resp.status_code in (200, 201):
        print("Request accepted — wait for 'Ready' status before querying (~20-30 min for new, ~10 min for update)")
    else:
        raise Exception(f"Failed: {resp.status_code} — {resp.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to be Ready
# MAGIC
# MAGIC Polls until the endpoint status is `READY` before querying. Typically 20-30 min for first deployment.

# COMMAND ----------

import time

if not DEPLOY_ENDPOINT:
    print(f"DEPLOY_ENDPOINT is False — skipping wait. Assuming endpoint '{ENDPOINT_NAME}' is already live.")
else:
    print(f"Waiting for endpoint '{ENDPOINT_NAME}' to be ready...")
    while True:
        resp = requests.get(
            f"{w.config.host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}",
            headers=headers,
        )
        state = resp.json().get("state", {}).get("ready", "")
        config_state = resp.json().get("state", {}).get("config_update", "")
        print(f"  Status: {state} | Config: {config_state}")
        if state == "READY":
            print(f"Endpoint '{ENDPOINT_NAME}' is ready")
            break
        elif state == "FAILED":
            raise Exception(f"Endpoint failed: {resp.json()}")
        time.sleep(60)

# COMMAND ----------

# DBTITLE 1,OpenAI API Compatibility Note
# MAGIC %md
# MAGIC ## Step 6 — OpenAI-Compatible Client Wrapper
# MAGIC
# MAGIC The class below wraps the Databricks Model Serving **pyfunc `dataframe_records`** endpoint in an interface that mirrors the OpenAI Python SDK.
# MAGIC
# MAGIC | What changes | What stays the same |
# MAGIC |---|---|
# MAGIC | `openai.OpenAI(...)` → `DatabricksVLLMClient(endpoint_url, token)` | `messages`, `tools`, `tool_choice` — identical format |
# MAGIC | No `base_url`, no `/v1` route needed | Response dict: `result["choices"][0]["message"]` |
# MAGIC
# MAGIC **No additional infrastructure required.** No AI Gateway, no Provisioned Throughput, no redeployment. The wrapper talks directly to the existing pyfunc endpoint created in Step 5.
# MAGIC
# MAGIC Vivek's team copies this class into their codebase, swaps `openai.OpenAI` for `DatabricksVLLMClient`, and the only things that change are the endpoint URL and the auth token.

# COMMAND ----------

# DBTITLE 1,DatabricksVLLMClient class definition
import requests, json


class DatabricksVLLMClient:
    """
    OpenAI-style wrapper around a Databricks Model Serving pyfunc endpoint.

    Translates chat_completions_create() calls into the dataframe_records
    invocation format and parses the response back into an OpenAI
    chat.completion dict.

    Usage:
        client = DatabricksVLLMClient(endpoint_url=invoke_url, token=token)
        result = client.chat_completions_create(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(result["choices"][0]["message"]["content"])
    """

    def __init__(self, endpoint_url: str, token: str):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def chat_completions_create(self, messages, tools=None, **kwargs):
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts [{"role": ..., "content": ...}]
            tools:    Optional list of tool/function definitions (OpenAI format)
            **kwargs: Reserved for future parameters

        Returns:
            Parsed dict in OpenAI chat.completion format:
            {
                "id": "chatcmpl-...",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {...}, "finish_reason": "stop"}]
            }
        """
        record = {"messages": json.dumps(messages)}
        if tools is not None:
            record["tools"] = json.dumps(tools)

        resp = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json={"dataframe_records": [record]},
        )
        resp.raise_for_status()

        # Pyfunc returns {"predictions": [{"response": "<json-string>"}]}
        predictions = resp.json().get("predictions", [])
        if not predictions:
            raise ValueError(f"Empty predictions from endpoint: {resp.json()}")

        raw = predictions[0]
        # Handle both dict-with-response-key and bare-string formats
        response_str = raw.get("response", raw) if isinstance(raw, dict) else raw
        return json.loads(response_str) if isinstance(response_str, str) else response_str


print("DatabricksVLLMClient ready.")

# COMMAND ----------

# DBTITLE 1,Step 6 — Chat Completions (OpenAI SDK)
# MAGIC %md
# MAGIC ## Step 7 — Chat Completions
# MAGIC
# MAGIC Instantiate the wrapper and call `chat_completions_create()` with the same `messages` list you would pass to `openai.ChatCompletion.create()`.

# COMMAND ----------

# DBTITLE 1,Chat completion via OpenAI SDK
client = DatabricksVLLMClient(endpoint_url=invoke_url, token=token)

result = client.chat_completions_create(
    messages=[
        {"role": "system", "content": "You are a helpful financial data assistant."},
        {"role": "user",   "content": "Summarize the key risk factors in a trade confirmation document."},
    ]
)

print("Response ID:", result["id"])
print()
msg = result["choices"][0]["message"]
print("Reply:", msg["content"])

# COMMAND ----------

# DBTITLE 1,Step 7 — Function / Tool Calling (OpenAI SDK)
# MAGIC %md
# MAGIC ## Step 8 — Function / Tool Calling
# MAGIC
# MAGIC Same `client`, same endpoint — pass `tools` to `chat_completions_create()`.
# MAGIC The model extracts structured fields and returns them as a typed function call.
# MAGIC
# MAGIC > Requires an instruction-tuned model with tool calling support
# MAGIC > (e.g., Llama 3 Instruct, Qwen2.5 Instruct). The current PoC uses `facebook/opt-125m` (base model) — tool calls will only produce structured output after swapping to an instruct model.

# COMMAND ----------

# DBTITLE 1,Tool calling via OpenAI SDK
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_document_fields",
            "description": "Extract structured fields from a financial document",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id":   {"type": "string"},
                    "date":          {"type": "string"},
                    "amount":        {"type": "number"},
                    "counterparty":  {"type": "string"},
                    "document_type": {"type": "string", "enum": ["invoice", "trade_confirm", "statement"]},
                },
                "required": ["document_id", "date", "amount", "counterparty", "document_type"],
            },
        },
    }
]

result = client.chat_completions_create(
    messages=[{
        "role": "user",
        "content": "Extract fields: Trade Confirm #TC-9921, Date: 2024-11-20, Amount: $4,500,000, Counterparty: State Street Bank",
    }],
    tools=tools,
)

msg = result["choices"][0]["message"]

if "tool_calls" in msg and msg["tool_calls"]:
    for tc in msg["tool_calls"]:
        print(f"Function : {tc['function']['name']}")
        print(f"Extracted:")
        print(json.dumps(json.loads(tc["function"]["arguments"]), indent=2))
else:
    print("Reply:", msg.get("content", ""))