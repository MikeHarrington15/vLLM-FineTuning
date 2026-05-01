# Databricks notebook source
# MAGIC %md
# MAGIC # SSCD — vLLM ChatModel (Python API, No Subprocess)
# MAGIC
# MAGIC Architecture: UC Volume model → vLLM Python API inside MLflow ChatModel → Databricks Model Serving (GPU) → native OpenAI SDK
# MAGIC
# MAGIC | Requirement | Solution |
# MAGIC |---|---|
# MAGIC | HuggingFace LLM in UC | Model weights in UC Volume, registered as UC model |
# MAGIC | vLLM inference engine | `vllm.LLM` Python class — no subprocess, no ports |
# MAGIC | Model Serving endpoint | GPU-backed managed endpoint — autoscaling, scale-to-zero |
# MAGIC | Native OpenAI API spec | `mlflow.pyfunc.ChatModel` exposes `/v1/chat/completions` natively |
# MAGIC | Chat completions | Standard `openai.OpenAI` client |
# MAGIC | Function / tool calling | `tokenizer.apply_chat_template(tools=...)` + Qwen2.5 tool call parsing |
# MAGIC
# MAGIC **Compute:** Run Steps 1–5 on any CPU cluster or Serverless. No GPU needed here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies

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
UC_REGISTERED_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.sscd_llm_chat_v2"
ENDPOINT_NAME       = "sscd-vllm-chat-v2"

w          = WorkspaceClient()
token      = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers    = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
invoke_url = f"{w.config.host}/serving-endpoints"

print(f"Workspace : {w.config.host}")
print(f"Model path: {UC_VOLUME_MODEL_PATH}")
print(f"Endpoint  : {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Define the ChatModel
# MAGIC
# MAGIC Uses the vLLM Python API (`LLM` class) directly — no subprocess, no port management, no health checks.
# MAGIC `load_context` creates the LLM instance (skipped on CPU clusters during MLflow validation).
# MAGIC `predict` formats requests via the tokenizer chat template and parses Qwen2.5 tool call output.

# COMMAND ----------

class VLLMChatModel(mlflow.pyfunc.ChatModel):
    _MODEL_NAME = "Qwen2.5-3B-Instruct"

    def load_context(self, context):
        import os, subprocess
        model_dir = context.artifacts["model_dir"]

        def _has_gpu():
            if os.path.exists('/dev/nvidiactl'):
                return True
            try:
                return subprocess.run(
                    ['nvidia-smi'], capture_output=True, timeout=5
                ).returncode == 0
            except Exception:
                pass
            return False

        if not _has_gpu():
            self._llm = None
            return

        from vllm import LLM
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self._llm = LLM(
            model=model_dir,
            dtype="half",
            gpu_memory_utilization=0.90,
            max_num_seqs=32,
            trust_remote_code=True,
        )

    def predict(self, context, messages, params):
        import re, time
        from mlflow.types.llm import ChatCompletionResponse
        from vllm import SamplingParams

        def _dummy():
            return ChatCompletionResponse.from_dict({
                "id": "chatcmpl-validation",
                "object": "chat.completion",
                "created": 0,
                "model": self._MODEL_NAME,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        if self._llm is None:
            return _dummy()

        msgs    = [{"role": m.role, "content": m.content or ""} for m in messages]
        tools   = getattr(params, "tools",       None)
        temp    = getattr(params, "temperature", None) or 0.1
        max_tok = getattr(params, "max_tokens",  None) or 512

        prompt = self._tokenizer.apply_chat_template(
            msgs,
            tools=tools if tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )

        outputs = self._llm.generate(
            prompt,
            SamplingParams(temperature=temp, max_tokens=max_tok),
        )

        result = outputs[0]
        text   = result.outputs[0].text.strip()

        tool_call_matches = re.findall(
            r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL
        )

        if tool_call_matches and tools:
            tool_calls = []
            for i, match in enumerate(tool_call_matches):
                try:
                    data = json.loads(match)
                    tool_calls.append({
                        "id":       f"call_{i}_{int(time.time())}",
                        "type":     "function",
                        "function": {
                            "name":      data["name"],
                            "arguments": json.dumps(
                                data.get("arguments", data.get("parameters", {}))
                            ),
                        },
                    })
                except (json.JSONDecodeError, KeyError):
                    pass
            message       = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        else:
            message       = {"role": "assistant", "content": text}
            finish_reason = "stop"

        return ChatCompletionResponse.from_dict({
            "id":      f"chatcmpl-{int(time.time())}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   self._MODEL_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage":   {
                "prompt_tokens":     len(result.prompt_token_ids),
                "completion_tokens": len(result.outputs[0].token_ids),
                "total_tokens":      len(result.prompt_token_ids) + len(result.outputs[0].token_ids),
            },
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Register Model to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="sscd_vllm_chat_v2"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=VLLMChatModel(),
        artifacts={"model_dir": UC_VOLUME_MODEL_PATH},
        pip_requirements=[
            "vllm==0.8.5.post1",
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

print("Accepted — endpoint provisioning in progress.")

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
