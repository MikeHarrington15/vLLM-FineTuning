# Databricks notebook source
# MAGIC %md
# MAGIC # SSCD — Native OpenAI Tool Calling via vLLM + External Model
# MAGIC
# MAGIC Architecture: HuggingFace model → UC Volume → vLLM serve (this cluster) → Databricks External Model endpoint → OpenAI SDK
# MAGIC
# MAGIC | Requirement | Solution |
# MAGIC |---|---|
# MAGIC | HuggingFace LLM in UC | Model weights stored in a UC Volume |
# MAGIC | vLLM inference engine | `vllm serve` on the driver node — full control over all vLLM parameters |
# MAGIC | Databricks Model Serving endpoint | External Model endpoint — appears in Serving UI, callable with a Databricks token |
# MAGIC | OpenAI API spec | Native — `vllm serve` exposes `/v1/chat/completions` directly |
# MAGIC | Chat completions | Native OpenAI SDK — no wrapper needed |
# MAGIC | Function / tool calling | `--enable-auto-tool-choice --tool-call-parser hermes` |
# MAGIC
# MAGIC **Compute:** Run this notebook on a GPU ML cluster (A10 or better, 24GB+ VRAM). `vllm serve` runs on the driver node.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies

# COMMAND ----------

# MAGIC %pip install -q "vllm>=0.8.5" openai databricks-sdk huggingface_hub
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration
# MAGIC
# MAGIC Set `UC_VOLUME_MODEL_PATH` to your model weights in the UC Volume.

# COMMAND ----------

import json, requests, time
from databricks.sdk import WorkspaceClient

UC_VOLUME_MODEL_PATH = "/Volumes/serverless_markospetko_catalog/vllm/model_weights/Qwen2.5-3B-Instruct"
ENDPOINT_NAME        = "sscd-vllm-external-v1"
VLLM_PORT            = 8000

w          = WorkspaceClient()
ctx        = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token      = ctx.apiToken().get()
cluster_id = ctx.clusterId().get()
org_id     = ctx.tags().apply("orgId")
ws_url     = spark.conf.get("spark.databricks.workspaceUrl")

driver_proxy_base = f"https://{ws_url}/driver-proxy-api/o/{org_id}/{cluster_id}/{VLLM_PORT}"
invoke_url        = f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/v1"

print(f"Cluster        : {cluster_id}")
print(f"Driver proxy   : {driver_proxy_base}")
print(f"Endpoint URL   : {invoke_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Start vLLM Serve
# MAGIC
# MAGIC Starts `vllm serve` on the driver node with tool calling enabled.
# MAGIC Polls the health endpoint until the server is ready before proceeding.
# MAGIC Typically takes 3–5 minutes for Qwen2.5-3B-Instruct.

# COMMAND ----------

import subprocess

proc = subprocess.Popen(
    [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  UC_VOLUME_MODEL_PATH,
        "--host",                   "0.0.0.0",
        "--port",                   str(VLLM_PORT),
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

print("Starting vLLM server...")
for _ in range(60):
    try:
        r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
        if r.status_code == 200:
            print("vLLM server ready.")
            break
    except Exception:
        pass
    time.sleep(10)
else:
    proc.kill()
    raise RuntimeError("vLLM server failed to start within 10 minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Register as Databricks External Model Endpoint
# MAGIC
# MAGIC Creates a Databricks Model Serving endpoint that proxies to the vLLM server on this cluster.
# MAGIC The endpoint appears in the Serving UI and is callable with a standard Databricks token.

# COMMAND ----------

headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url = f"{w.config.host}/api/2.0/serving-endpoints"

endpoint_body = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_entities": [{
            "external_model": {
                "name":                "Qwen2.5-3B-Instruct",
                "provider":            "custom",
                "task":                "llm/v1/chat",
                "custom_provider_url": driver_proxy_base,
                "bearer_token_auth":   {"token": token},
            }
        }]
    }
}

exists = requests.get(f"{base_url}/{ENDPOINT_NAME}", headers=headers).status_code == 200

if exists:
    resp = requests.put(
        f"{base_url}/{ENDPOINT_NAME}/config",
        headers=headers,
        json=endpoint_body["config"],
    )
    print(f"Updated endpoint '{ENDPOINT_NAME}'")
else:
    resp = requests.post(base_url, headers=headers, json=endpoint_body)
    print(f"Created endpoint '{ENDPOINT_NAME}'")

if resp.status_code not in (200, 201):
    raise Exception(f"Failed: {resp.status_code} — {resp.text}")

print(f"Endpoint live at: {invoke_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Chat Completions
# MAGIC
# MAGIC Call the External Model endpoint with the standard OpenAI Python SDK.
# MAGIC No wrapper, no adapter — identical to calling OpenAI directly.

# COMMAND ----------

from openai import OpenAI

client = OpenAI(
    api_key=token,
    base_url=invoke_url,
)

result = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful financial data assistant."},
        {"role": "user",   "content": "Summarize the key risk factors in a trade confirmation document."},
    ],
    max_tokens=512,
)

print("Response ID:", result.id)
print()
print("Reply:", result.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Function / Tool Calling
# MAGIC
# MAGIC Same client, same endpoint.
# MAGIC Tool calling works natively — no text parsing, no wrapper.

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

result = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": "Extract fields: Trade Confirm #TC-9921, Date: 2024-11-20, Amount: $4,500,000, Counterparty: State Street Bank",
    }],
    tools=tools,
    tool_choice="auto",
)

msg = result.choices[0].message

if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Function : {tc.function.name}")
        print(f"Extracted:")
        print(json.dumps(json.loads(tc.function.arguments), indent=2))
else:
    print("Reply:", msg.content)
