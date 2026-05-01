# Databricks notebook source
# MAGIC %md
# MAGIC # SSCD — vLLM Serve via MLflow ChatModel (Native OpenAI API)
# MAGIC
# MAGIC Architecture: UC Volume model → MLflow ChatModel → Databricks Model Serving (GPU) → native OpenAI SDK
# MAGIC
# MAGIC | Requirement | Solution |
# MAGIC |---|---|
# MAGIC | HuggingFace LLM in UC | Model weights in UC Volume, registered as UC model |
# MAGIC | vLLM inference engine | `vllm serve` subprocess inside the Model Serving container |
# MAGIC | Model Serving endpoint | GPU-backed managed endpoint — autoscaling, scale-to-zero |
# MAGIC | Native OpenAI API spec | `mlflow.pyfunc.ChatModel` exposes `/v1/chat/completions` natively |
# MAGIC | Chat completions | Standard `openai.OpenAI` client — zero wrapper |
# MAGIC | Function / tool calling | `--enable-auto-tool-choice --tool-call-parser hermes` on vllm serve |
# MAGIC
# MAGIC **Compute:** Run Steps 1–5 on any cluster (CPU or Serverless). No GPU needed here.
# MAGIC Databricks provisions the GPU automatically when the endpoint starts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies
# MAGIC
# MAGIC Only `mlflow`, `databricks-sdk`, and `openai` are needed on this cluster.
# MAGIC `vllm` is installed inside the Model Serving container via `pip_requirements`.

# COMMAND ----------

# MAGIC %pip install -q mlflow databricks-sdk openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration

# COMMAND ----------

import json, requests, mlflow, mlflow.pyfunc
from databricks.sdk import WorkspaceClient

UC_VOLUME_MODEL_PATH = "/Volumes/serverless_markospetko_catalog/vllm/model_weights/Qwen2.5-3B-Instruct"

UC_CATALOG          = "serverless_markospetko_catalog"
UC_SCHEMA           = "vllm"
UC_REGISTERED_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.sscd_llm_chat"
ENDPOINT_NAME       = "sscd-vllm-chat-v1"

w          = WorkspaceClient()
token      = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
invoke_url = f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/v1"

print(f"Workspace : {w.config.host}")
print(f"Model path: {UC_VOLUME_MODEL_PATH}")
print(f"Endpoint  : {ENDPOINT_NAME}")
print(f"Client URL: {invoke_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Define the ChatModel
# MAGIC
# MAGIC `mlflow.pyfunc.ChatModel` is the key class here. When deployed to Databricks Model Serving,
# MAGIC it automatically exposes a native `/v1/chat/completions` endpoint — no wrapper needed on the client.
# MAGIC
# MAGIC `load_context` starts `vllm serve` as a subprocess inside the serving container.
# MAGIC `predict` converts the incoming OpenAI-format request, forwards it to vllm, and returns the response.

# COMMAND ----------

from mlflow.types.llm import ChatResponse, ChatChoice, ChatMessage

class VLLMChatModel(mlflow.pyfunc.ChatModel):
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

    def predict(self, context, messages, params):
        import requests as _req

        msgs = []
        for m in messages:
            d = {"role": m.role, "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls
            if getattr(m, "tool_call_id", None):
                d["tool_call_id"] = m.tool_call_id
            msgs.append(d)

        payload = {
            "model":       self._MODEL_NAME,
            "messages":    msgs,
            "temperature": getattr(params, "temperature", None) or 0.1,
            "max_tokens":  getattr(params, "max_tokens",  None) or 512,
        }
        if getattr(params, "tools", None):
            payload["tools"]       = params.tools
            payload["tool_choice"] = getattr(params, "tool_choice", "auto") or "auto"

        resp = _req.post(
            f"http://localhost:{self._PORT}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_choice = data["choices"][0]
        raw_msg    = raw_choice["message"]

        return ChatResponse(
            choices=[ChatChoice(
                index=0,
                message=ChatMessage(
                    role=raw_msg["role"],
                    content=raw_msg.get("content"),
                    tool_calls=raw_msg.get("tool_calls"),
                ),
                finish_reason=raw_choice.get("finish_reason", "stop"),
            )],
            usage=data.get("usage", {}),
            model=data.get("model", self._MODEL_NAME),
        )

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.poll() is None:
            self._proc.kill()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Register Model to Unity Catalog
# MAGIC
# MAGIC No signature needed — `ChatModel` defines the OpenAI-compatible schema automatically.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="sscd_vllm_chat"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=VLLMChatModel(),
        artifacts={"model_dir": UC_VOLUME_MODEL_PATH},
        pip_requirements=[
            "vllm==0.6.3.post1",
            "transformers==4.46.0",
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
# MAGIC First deployment takes ~20–30 minutes. The endpoint provisions its own GPU.
# MAGIC `vllm serve` starts inside the container on first request (adds ~5 min to cold-start).

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

print("Accepted — endpoint provisioning in progress. Check Serving UI for status.")

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
# MAGIC ## Step 7 — Chat Completions
# MAGIC
# MAGIC Native OpenAI SDK — no wrapper, no translation layer.
# MAGIC Identical to calling OpenAI directly, just swap `base_url` and `api_key`.

# COMMAND ----------

from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url=invoke_url,
)

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful financial data assistant."},
        {"role": "user",   "content": "Summarize the key risk factors in a trade confirmation document."},
    ],
    max_tokens=512,
)

print("Response ID:", response.id)
print()
print("Reply:", response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — Tool / Function Calling
# MAGIC
# MAGIC Same client, same endpoint — pass `tools` exactly as you would with OpenAI.
# MAGIC `tool_calls` comes back in the standard OpenAI format.

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

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[{
        "role": "user",
        "content": "Extract fields: Trade Confirm #TC-9921, Date: 2024-11-20, Amount: $4,500,000, Counterparty: State Street Bank",
    }],
    tools=tools,
    tool_choice="auto",
    max_tokens=512,
)

msg = response.choices[0].message

if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Function : {tc.function.name}")
        print(f"Extracted:")
        print(json.dumps(json.loads(tc.function.arguments), indent=2))
else:
    print("Reply:", msg.content)
