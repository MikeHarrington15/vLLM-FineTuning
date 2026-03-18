# vLLM Self-Hosting + Fine-Tuning on Databricks

End-to-end POC for self-hosting a HuggingFace model on Databricks with vLLM, including fine-tuning with LoRA and MLflow observability.

## What This Covers

1. **Download model from HuggingFace** and store weights in Unity Catalog (governed storage)
2. **Serve with vLLM** on classic GPU compute
3. **Fine-tune with LoRA/PEFT** using a small example dataset
4. **Re-serve the fine-tuned model** with vLLM
5. **Register to Unity Catalog** as a versioned, governed model via MLflow
6. **Observe with MLflow Tracing** — input/output logging and latency tracking
7. **Serverless vs classic GPU comparison** and cost estimates

## Requirements

- **Databricks Runtime**: 15.4 LTS ML
- **Cluster**: Classic GPU compute (e.g., `g4dn.xlarge` with 1x T4, or `g5.xlarge` with 1x A10G)
- **Unity Catalog**: A catalog and schema you have write access to

## Usage

1. Import `vllm_finetune_poc.py` into your Databricks workspace
2. Attach to a GPU cluster running Runtime 15.4 LTS ML
3. Update the configuration section with your UC catalog/schema and HuggingFace model name
4. Run All

## Adapting for Your Model

- Set `HF_MODEL_NAME` to your HuggingFace model repo ID
- Update LoRA `target_modules` to match your model's architecture
- Replace the sample training data with your own dataset
- For vision/multimodal models, use the appropriate Transformers pipeline for fine-tuning
