# Databricks notebook source
# MAGIC %md
# MAGIC # SSCD — vLLM Serve Inside Model Serving (Subprocess Proxy)
# MAGIC
# MAGIC Architecture: UC Volume model → pyfunc container starts `vllm serve` as subprocess → Databricks Model Serving endpoint → OpenAI-compatible client
# MAGIC
# MAGIC | Requirement | Solution |
# MAGIC |---|---|
# MAGIC | HuggingFace LLM in UC | Model weights in UC Volume, referenced at serve time |
# MAGIC | vLLM inference engine | `vllm serve` runs inside the Model Serving container as a subprocess |
# MAGIC | Model Serving endpoint | GPU-backed managed endpoint — autoscaling, scale-to-zero |
# MAGIC | OpenAI API spec | pyfunc proxies `/v1/chat/completions` directly from vllm — no translation layer |
# MAGIC | Chat completions | Full OpenAI `chat.completion` response format |
# MAGIC | Function / tool calling | `--enable-auto-tool-choice --tool-call-parser hermes` passed to vllm serve |
# MAGIC
# MAGIC **Compute:** Run this notebook on any CPU cluster or Serverless — no GPU needed here.
# MAGIC The Model Serving endpoint provisions its own GPU automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies

# COMMAND ----------

# MAGIC %pip install -q "vllm>=0.8.5" mlflow databricks-sdk openai huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration

# COMMAND ----------

import json, requests, mlflow, mlflow.pyfunc, pandas as pd
from databricks.sdk import WorkspaceClient

UC_VOLUME_MODEL_PATH = "/Volumes/serverless_markospetko_catalog/vllm/model_weights/Qwen2.5-3B-Instruct"

UC_CATALOG          = "serverless_markospetko_catalog"
UC_SCHEMA           = "vllm"
UC_REGISTERED_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.sscd_llm_serve"
ENDPOINT_NAME       = "sscd-vllm-serve-v1"

w          = WorkspaceClient()
token      = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
invoke_url = f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations"

print(f"Workspace : {w.config.host}")
print(f"Model path: {UC_VOLUME_MODEL_PATH}")
print(f"Endpoint  : {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Define the Pyfunc Model
# MAGIC
# MAGIC `load_context` starts `vllm serve` as a subprocess inside the serving container and polls until healthy.
# MAGIC `predict` forwards requests to `http://localhost:8000/v1/chat/completions` and returns the raw OpenAI response.
# MAGIC No translation of the response — it comes back in native OpenAI format.

# COMMAND ----------

class VLLMServePyfuncModel(mlflow.pyfunc.PythonModel):
    _PORT       = 8000
    _MODEL_NAME = "Qwen2.5-3B-Instruct"

    def load_context(self, context):
        import subprocess, time
        import requests as _req

        model_dir = context.artifacts["model_dir"]

        self._proc = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model",                  model_dir,
                "--served-model-name",      self._MODEL_NAME,
                "--host",                   "0.0.0.0",
                "--port",                   str(self._PORT),
                "--enable-auto-tool-choice",
                "--tool-call-parser",       "hermes",
                "--dtype",                  "half",
                "--gpu-memory-utilization", "0.90",
                "--max-num-seqs",           "32",
                "--trust-remote-code",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for _ in range(60):
            try:
                r = _req.get(f"http://localhost:{self._PORT}/health", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(10)

        self._proc.kill()
        raise RuntimeError("vLLM server failed to start within 10 minutes.")

    def predict(self, context, model_input):
        import json as _json
        import requests as _req

        row      = model_input.iloc[0]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = _json.loads(messages)

        tools = None
        if "tools" in model_input.columns and row["tools"] is not None:
            tools = row["tools"]
            if isinstance(tools, str):
                tools = _json.loads(tools)

        payload = {
            "model":       self._MODEL_NAME,
            "messages":    messages,
            "temperature": 0.1,
            "max_tokens":  512,
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"

        resp = _req.post(
            f"http://localhost:{self._PORT}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return pd.DataFrame({"response": [resp.text]})

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.poll() is None:
            self._proc.kill()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Register Model to Unity Catalog

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="sscd_vllm_serve"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=VLLMServePyfuncModel(),
        signature=ModelSignature(
            inputs=Schema([
                ColSpec("string", "messages"),
                ColSpec("string", "tools", required=False),
            ]),
            outputs=Schema([ColSpec("string", "response")]),
        ),
        artifacts={"model_dir": UC_VOLUME_MODEL_PATH},
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
# MAGIC Databricks provisions the GPU automatically. First deployment takes ~20–30 minutes.
# MAGIC Note: `vllm serve` starts inside the container on cold-start, adding ~3–5 min to first-request latency.

# COMMAND ----------

base_url = f"{w.config.host}/api/2.0/serving-endpoints"

latest_version = max(
    w.model_versions.list(full_name=UC_REGISTERED_MODEL),
    key=lambda v: int(v.version),
).version
print(f"Deploying model version: {latest_version}")

config = {
    "served_entities": [{
        "entity_name":           UC_REGISTERED_MODEL,
        "entity_version":        str(latest_version),
        "workload_size":         "Small",
        "workload_type":         "GPU_SMALL",
        "scale_to_zero_enabled": True,
    }]
}

exists = requests.get(f"{base_url}/{ENDPOINT_NAME}", headers=headers).status_code == 200

if exists:
    resp = requests.put(f"{base_url}/{ENDPOINT_NAME}/config", headers=headers, json=config)
    print(f"Updating endpoint '{ENDPOINT_NAME}' to v{latest_version}...")
else:
    resp = requests.post(base_url, headers=headers, json={"name": ENDPOINT_NAME, "config": config})
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")

if resp.status_code not in (200, 201):
    raise Exception(f"Failed: {resp.status_code} — {resp.text}")

print("Request accepted — endpoint provisioning in progress.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Wait for Endpoint Ready

# COMMAND ----------

import time

print(f"Waiting for '{ENDPOINT_NAME}'...")
while True:
    state        = requests.get(f"{base_url}/{ENDPOINT_NAME}", headers=headers).json().get("state", {})
    ready        = state.get("ready", "")
    config_state = state.get("config_update", "NOT_UPDATING")
    print(f"  Status: {ready} | Config: {config_state}")

    if ready == "READY" and config_state != "IN_PROGRESS":
        print(f"\nEndpoint '{ENDPOINT_NAME}' is ready.")
        break
    elif ready == "FAILED" or config_state == "UPDATE_FAILED":
        raise Exception(f"Endpoint failed: {state}")
    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Client
# MAGIC
# MAGIC Translates OpenAI-style calls into the Databricks `dataframe_records` invocation format.
# MAGIC The response coming back from the pyfunc is already a native OpenAI `chat.completion` JSON string —
# MAGIC this wrapper just unwraps the Databricks prediction envelope.

# COMMAND ----------

class DatabricksVLLMClient:

    def __init__(self, endpoint_url: str, token: str):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    def chat_completions_create(self, messages, tools=None, **kwargs):
        record = {"messages": json.dumps(messages)}
        if tools is not None:
            record["tools"] = json.dumps(tools)

        resp = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json={"dataframe_records": [record]},
        )
        resp.raise_for_status()

        predictions  = resp.json().get("predictions", [])
        raw          = predictions[0]
        response_str = raw.get("response", raw) if isinstance(raw, dict) else raw
        return json.loads(response_str) if isinstance(response_str, str) else response_str


client = DatabricksVLLMClient(endpoint_url=invoke_url, token=token)
print("Client ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Chat Completions

# COMMAND ----------

result = client.chat_completions_create(
    messages=[
        {"role": "system", "content": "You are a helpful financial data assistant."},
        {"role": "user",   "content": "Summarize the key risk factors in a trade confirmation document."},
    ]
)

print("Response ID:", result["id"])
print()
print("Reply:", result["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — Tool / Function Calling
# MAGIC
# MAGIC Tool calling works natively — `vllm serve` handles `--enable-auto-tool-choice` and `--tool-call-parser hermes`.
# MAGIC The pyfunc proxies the response directly, so `tool_calls` comes back in the standard OpenAI format.

# COMMAND ----------

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
