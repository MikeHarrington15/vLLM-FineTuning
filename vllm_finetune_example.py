# Databricks notebook source
# MAGIC %md
# MAGIC # vLLM Self-Hosting + Fine-Tuning on Databricks
# MAGIC
# MAGIC End-to-end workflow for self-hosting a HuggingFace model on Databricks with vLLM:
# MAGIC
# MAGIC 1. Download model from HuggingFace and store weights in Unity Catalog
# MAGIC 2. Serve with vLLM on classic GPU compute
# MAGIC 3. Fine-tune with LoRA/PEFT
# MAGIC 4. Re-serve the fine-tuned model with vLLM
# MAGIC 5. Register to Unity Catalog as a governed model
# MAGIC 6. Observe with MLflow tracing
# MAGIC
# MAGIC **Requirements**: Databricks Runtime 15.4 LTS ML, classic GPU cluster (e.g., `g4dn.xlarge` or `g5.xlarge`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install "vllm==0.6.3.post1" "transformers==4.46.0" "peft>=0.13.0" "trl>=0.9.0" "datasets>=2.14.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC
# MAGIC Update `HF_MODEL_NAME` with your model and the UC paths with your catalog/schema.

# COMMAND ----------

import os
import time
import json
import shutil
import gc
from pathlib import Path
from typing import Optional, List

import mlflow
import mlflow.pyfunc
import pandas as pd
import torch
from databricks.sdk import WorkspaceClient

# --- Model Configuration ---
# Replace with your HuggingFace model (e.g., your OCR model repo ID)
HF_MODEL_NAME = "facebook/opt-125m"

# --- Unity Catalog Configuration ---
# Update these to your catalog/schema
UC_CATALOG = "mh_sandbox"
UC_SCHEMA = "vllm_poc"
UC_VOLUME = "model_weights"
UC_REGISTERED_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.opt125m_finetuned"

# --- Paths ---
LOCAL_CACHE = "/local_disk0/tmp/hf_model_cache"
UC_VOLUME_PATH = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}"

# --- Inference defaults ---
DEFAULT_MAX_TOKENS = 128

os.makedirs(LOCAL_CACHE, exist_ok=True)
client = WorkspaceClient()
print(f"Workspace: {client.config.host}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create UC Volume
# MAGIC
# MAGIC Unity Catalog Volumes provide governed storage for model weights.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {UC_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}")
print("UC catalog/schema/volume ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download Model from HuggingFace and Store in UC
# MAGIC
# MAGIC Model weights are stored as a governed object in Unity Catalog via Volumes.
# MAGIC This ensures access control, lineage, and auditability.

# COMMAND ----------

from huggingface_hub import snapshot_download

def download_hf_model(model_name: str, cache_dir: str) -> str:
    """Download HF model to local cache. Returns local path."""
    cache_path = Path(cache_dir) / model_name.replace("/", "_")
    if cache_path.exists() and (cache_path / "config.json").exists():
        print(f"Already cached: {cache_path}")
        return str(cache_path)
    print(f"Downloading {model_name}...")
    cache_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_name, local_dir=str(cache_path))
    print(f"Downloaded to: {cache_path}")
    return str(cache_path)

local_model_path = download_hf_model(HF_MODEL_NAME, LOCAL_CACHE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Copy weights to UC Volume

# COMMAND ----------

uc_model_dir = f"{UC_VOLUME_PATH}/{HF_MODEL_NAME.replace('/', '_')}"
os.makedirs(uc_model_dir, exist_ok=True)

print(f"Copying model weights to UC Volume: {uc_model_dir}")
local_path = Path(local_model_path)
file_count = 0
for f in local_path.rglob("*"):
    if f.is_file():
        rel = f.relative_to(local_path)
        dest = Path(uc_model_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(f), str(dest))
        file_count += 1
print(f"Copied {file_count} files to UC Volume: {uc_model_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Serve with vLLM on Classic GPU
# MAGIC
# MAGIC [vLLM](https://docs.vllm.ai/) provides high-throughput inference with PagedAttention,
# MAGIC continuous batching, and an OpenAI-compatible API. It runs on classic GPU compute
# MAGIC where you have full control over the serving infrastructure.

# COMMAND ----------

from vllm import LLM, SamplingParams

print("Initializing vLLM engine...")
llm = LLM(
    model=local_model_path,
    dtype="half",
    enforce_eager=True,
    gpu_memory_utilization=0.70,
    max_num_seqs=32,
    swap_space=0,
)
print("vLLM engine ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity test

# COMMAND ----------

sp = SamplingParams(temperature=0.1, max_tokens=128)
outputs = llm.generate(["User: What is machine learning?\nAssistant:"], sampling_params=sp)
print(outputs[0].outputs[0].text.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Fine-Tune with LoRA (PEFT)
# MAGIC
# MAGIC vLLM is an *inference* engine. Fine-tuning uses HuggingFace Transformers + [PEFT](https://huggingface.co/docs/peft).
# MAGIC After fine-tuning, we merge the LoRA adapter back into the base model and re-serve with vLLM.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Free vLLM GPU memory before fine-tuning

# COMMAND ----------

del llm
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory freed. Available: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load base model and apply LoRA adapter

# COMMAND ----------

from transformers import OPTForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = OPTForCausalLM.from_pretrained(HF_MODEL_NAME)
model.half().cuda()

# LoRA configuration — adjust target_modules for your model architecture
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a. Prepare Training Data
# MAGIC
# MAGIC Replace this with your own dataset. The examples below demonstrate
# MAGIC document field extraction — swap in your domain-specific training pairs.

# COMMAND ----------

from datasets import Dataset

train_data = [
    {"text": "Extract the key fields from this invoice: Invoice #1234, Date: 2024-01-15, Total: $5,432.10, Vendor: Acme Corp\n\nInvoice Number: 1234\nDate: 2024-01-15\nTotal: $5,432.10\nVendor: Acme Corp"},
    {"text": "Extract the key fields from this invoice: PO-9876, Date: 2024-03-22, Amount Due: $12,750.00, From: Global Systems Inc\n\nPO Number: 9876\nDate: 2024-03-22\nAmount Due: $12,750.00\nVendor: Global Systems Inc"},
    {"text": "Parse this receipt: Store: Target, Date: 03/15/2024, Items: Groceries $45.23, Household $22.10, Tax $5.39, Total $72.72\n\nStore: Target\nDate: 03/15/2024\nItems:\n- Groceries: $45.23\n- Household: $22.10\nTax: $5.39\nTotal: $72.72"},
    {"text": "Extract fields: Contract ID C-2024-0456, Effective 2024-06-01, Expiration 2025-05-31, Party A: Acme Corp, Party B: Vendor LLC, Value: $2.1M\n\nContract ID: C-2024-0456\nEffective Date: 2024-06-01\nExpiration Date: 2025-05-31\nParty A: Acme Corp\nParty B: Vendor LLC\nContract Value: $2.1M"},
    {"text": "Parse this trade confirmation: Trade ID T-78901, Security: AAPL, Quantity: 1000, Price: $178.50, Side: Buy, Settlement: T+2\n\nTrade ID: T-78901\nSecurity: AAPL\nQuantity: 1,000\nPrice: $178.50\nSide: Buy\nSettlement: T+2"},
]

dataset = Dataset.from_list(train_data)
print(f"Training samples: {len(dataset)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b. Run Fine-Tuning

# COMMAND ----------

from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="/local_disk0/tmp/lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    warmup_ratio=0.1,
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
)

print("Starting LoRA fine-tuning...")
trainer.train()
print("Fine-tuning complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c. Merge LoRA Adapter and Save
# MAGIC
# MAGIC Merge the LoRA weights back into the base model for clean vLLM serving.

# COMMAND ----------

adapter_path = "/local_disk0/tmp/lora_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved: {adapter_path}")

del model, trainer
gc.collect()
torch.cuda.empty_cache()

print("Reloading base model for adapter merge...")
base_model = OPTForCausalLM.from_pretrained(HF_MODEL_NAME)
base_model.half().cuda()

merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()

merged_path = "/local_disk0/tmp/merged_model"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved: {merged_path}")

del base_model, merged_model
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Serve Fine-Tuned Model with vLLM
# MAGIC
# MAGIC Reload the merged model into vLLM and verify inference.

# COMMAND ----------

from vllm import LLM, SamplingParams

print("Loading fine-tuned model into vLLM...")
llm_ft = LLM(
    model=merged_path,
    dtype="half",
    enforce_eager=True,
    gpu_memory_utilization=0.70,
    max_num_seqs=32,
    swap_space=0,
)

test_prompt = "Extract fields: Invoice #5577, Date: 2024-08-10, Total: $8,200.00, Vendor: DataCo Analytics\n\n"
sp = SamplingParams(temperature=0.1, max_tokens=128)
out = llm_ft.generate([test_prompt], sampling_params=sp)
print("Fine-tuned model response:")
print(out[0].outputs[0].text.strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register Fine-Tuned Model to Unity Catalog
# MAGIC
# MAGIC Register the model as a governed object in Unity Catalog via MLflow.
# MAGIC This provides versioning, lineage, access controls, and a path to Model Serving deployment.

# COMMAND ----------

del llm_ft
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

class VLLMPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for vLLM inference at serving time."""

    def load_context(self, context):
        from vllm import LLM
        model_path = context.artifacts["model_dir"]
        self.llm = LLM(model=model_path, dtype="half", enforce_eager=True,
                        gpu_memory_utilization=0.70, max_num_seqs=32, swap_space=0)

    def predict(self, context, model_input):
        from vllm import SamplingParams
        prompts = model_input["prompt"].tolist()
        sp = SamplingParams(temperature=0.1, max_tokens=128)
        outputs = self.llm.generate(prompts, sampling_params=sp)
        return pd.DataFrame({
            "prompt": prompts,
            "response": [o.outputs[0].text.strip() for o in outputs],
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log and register model

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "prompt")]),
    outputs=Schema([ColSpec("string", "prompt"), ColSpec("string", "response")]),
)

deploy_model = VLLMPyfuncModel()

with mlflow.start_run(run_name="vllm_finetuned_lora") as run:
    mlflow.log_params({
        "base_model": HF_MODEL_NAME,
        "inference_engine": "vllm",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "learning_rate": 2e-4,
    })

    mlflow.log_artifacts(merged_path, artifact_path="model_dir")

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=deploy_model,
        signature=signature,
        artifacts={"model_dir": merged_path},
        pip_requirements=[
            "vllm==0.6.3.post1",
            "transformers==4.46.0",
            "torch",
            "mlflow>=2.14.1",
        ],
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=UC_REGISTERED_MODEL)

print(f"Registered: {registered.name} v{registered.version}")
print(f"Model is now a governed object in Unity Catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Observe with MLflow Tracing
# MAGIC
# MAGIC Wrap vLLM inference calls with MLflow tracing for production observability.
# MAGIC Each call is logged with input, output, and latency in the MLflow Experiment UI.

# COMMAND ----------

from vllm import LLM, SamplingParams

llm_ft = LLM(
    model=merged_path,
    dtype="half",
    enforce_eager=True,
    gpu_memory_utilization=0.70,
    max_num_seqs=32,
    swap_space=0,
)

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/vllm_poc_tracing")

@mlflow.trace(name="vllm_inference")
def traced_inference(prompt: str) -> str:
    sp = SamplingParams(temperature=0.1, max_tokens=128)
    out = llm_ft.generate([prompt], sampling_params=sp)
    return out[0].outputs[0].text.strip()

test_prompts = [
    "Extract fields: Invoice #9999, Date: 2024-12-01, Total: $3,100.00, Vendor: FinTech Solutions\n\n",
    "Parse: Trade ID T-44001, Security: MSFT, Qty: 500, Price: $415.20, Side: Sell\n\n",
]

for p in test_prompts:
    result = traced_inference(p)
    print(f"Response: {result}\n---")

print("Check the MLflow Experiment UI to see traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Serverless vs Classic GPU — When to Use What
# MAGIC
# MAGIC | | Serverless (Foundation Model APIs) | Classic GPU (vLLM) |
# MAGIC |---|---|---|
# MAGIC | **Custom HuggingFace models** | Not supported | Full control over model and infra |
# MAGIC | **Databricks-managed models** (Llama, DBRX, Mixtral) | Fully managed, pay-per-token | Not needed |
# MAGIC | **Fine-tuning** | Via Mosaic fine-tuning only | Any framework (PEFT, full fine-tune) |
# MAGIC | **vLLM configuration** | Not user-configurable | Full control |
# MAGIC | **Scale to zero** | Automatic | Configurable on Model Serving |
# MAGIC | **Best for** | Standard LLM chat, embeddings | Custom/specialized models (OCR, vision, domain-specific) |
# MAGIC
# MAGIC **For custom HuggingFace models**: Classic GPU compute with vLLM is the recommended path.
# MAGIC Register the fine-tuned model to UC (done above), then deploy to a
# MAGIC [Model Serving endpoint](https://docs.databricks.com/en/machine-learning/model-serving/index.html) with GPU compute for production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cost Estimation — Classic GPU Compute (AWS)
# MAGIC
# MAGIC All-Purpose Compute pricing (Databricks Premium plan, AWS, DBU + EC2 combined):
# MAGIC
# MAGIC | Instance | GPU | VRAM | DBU/hr | Total $/hr | Good for |
# MAGIC |---|---|---|---|---|---|
# MAGIC | `g4dn.xlarge` | 1x T4 | 16 GB | 0.71 | $0.92 | Small models, dev/POC |
# MAGIC | `g5.xlarge` | 1x A10G | 24 GB | 1.36 | $1.75 | Models up to 7B, dev/test |
# MAGIC | `g5.2xlarge` | 1x A10G | 24 GB | 1.64 | $2.11 | 7B models, light production |
# MAGIC | `g5.12xlarge` | 4x A10G | 96 GB | 7.69 | $9.90 | 13-30B models, tensor parallel |
# MAGIC
# MAGIC Jobs Compute is cheaper ($0.15/DBU vs $0.55/DBU for All-Purpose).
# MAGIC
# MAGIC **Recommendations by model size:**
# MAGIC - **1-3B params** (e.g., OCR models): `g4dn.xlarge` ($0.92/hr) or `g5.xlarge` ($1.75/hr)
# MAGIC - **7B params**: `g5.2xlarge` ($2.11/hr) with scale-to-zero
# MAGIC - **13B+ params**: `g5.12xlarge` ($9.90/hr) with tensor parallelism
# MAGIC
# MAGIC *Sources: [Databricks Pricing Calculator](https://www.databricks.com/product/pricing/product-pricing/instance-types) (Premium, AWS, GPU Instances). EC2 on-demand rates from AWS. Negotiated enterprise pricing may differ.*