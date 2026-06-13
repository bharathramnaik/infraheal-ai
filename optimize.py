"""
InfraHeal AI — Agent Optimizer (LoRA Fine-Tuning)
==================================================
Reads approved experiences from the experience store and fine-tunes the
remediation agent using LoRA.  Outputs adapter weights that can be loaded
by vLLM via ``--lora-modules``.

Usage:
    python optimize.py                         # default: train on all approved
    python optimize.py --adapter-name my_v1    # named adapter version

Requires: torch, transformers, peft, bitsandbytes, accelerate, datasets
(Pre-installed on the AMD JupyterLab image.)
"""

import argparse
import json
import logging
import os
import sys
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("optimize")

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIENCE_STORE_PATH = os.path.join(HERE, "experience_store.json")
ADAPTERS_DIR = os.path.join(HERE, "adapters")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_experiences(path: str) -> List[dict]:
    if not os.path.exists(path):
        logger.warning("Experience store not found at %s", path)
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("experiences", [])


def build_training_pairs(experiences: List[dict]) -> List[dict]:
    """Convert approved experiences into (prompt, completion) pairs."""
    pairs = []
    for exp in experiences:
        if exp.get("verdict") != "approved":
            continue
        actions = exp.get("actions_attempted", [])
        if not actions:
            continue
        approved_actions = [a for a in actions if a.get("approved")]
        if not approved_actions:
            continue
        # Input: incident context
        prompt = (
            f"incident sev={exp.get('severity','?')} cat={exp.get('category','?')}\n"
            f"rca root={exp.get('root_cause','?')}\n"
            f"Plan remediation. ONLY JSON."
        )
        # Output: the approved actions as JSON
        actions_json = json.dumps(
            [{"tool_name": a["tool_name"], "risk_level": a.get("risk_level", "medium")}
             for a in approved_actions]
        )
        completion = (
            f'{{"recommended_actions":{actions_json},'
            f'"execution_order":"seq",'
            f'"rollback_plan":"Standard rollback",'
            f'"estimated_resolution_time":"5-10min",'
            f'"warnings":[],"confidence":0.85}}'
        )
        pairs.append({"prompt": prompt, "completion": completion})
    logger.info("Built %d training pairs from %d experiences", len(pairs), len(experiences))
    return pairs


def train_lora(pairs: List[dict], model_name: str = DEFAULT_MODEL, adapter_name: str = "remediation"):
    """LoRA fine-tune the model on approved remediation decisions."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    if not pairs:
        logger.warning("No training pairs — skipping training")
        return None

    logger.info("Loading base model %s (4-bit quantized)...", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format dataset with chat template
    def format_example(ex):
        text = (
            f"<|im_start|>system\nYou are a remediation planner. Plan remediation actions as JSON.<|im_end|>\n"
            f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['completion']}<|im_end|>"
        )
        return tokenizer(text, truncation=True, max_length=1024, padding="max_length")

    dataset = Dataset.from_list(pairs).map(format_example, remove_columns=["prompt", "completion"])

    adapter_path = os.path.join(ADAPTERS_DIR, adapter_name)
    os.makedirs(adapter_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    logger.info("Starting LoRA fine-tuning (%d samples)...", len(pairs))
    trainer.train()
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    logger.info("Adapter saved to %s", adapter_path)

    # Write adapter config for vLLM
    adapter_config = {
        "model": model_name,
        "adapter": adapter_name,
        "path": adapter_path,
        "trained_on": len(pairs),
        "command": f"vllm serve {model_name} --enable-lora --lora-modules {adapter_name}={adapter_path}",
    }
    config_path = os.path.join(adapter_path, "infraheal_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    logger.info("Config written to %s", config_path)
    return adapter_path


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune InfraHeal remediation agent")
    parser.add_argument("--adapter-name", default="remediation", help="Adapter name/save directory")
    args = parser.parse_args()

    experiences = load_experiences(EXPERIENCE_STORE_PATH)
    if not experiences:
        logger.error("No experiences found — cannot train. Approve some actions first.")
        sys.exit(1)

    pairs = build_training_pairs(experiences)
    if not pairs:
        logger.error("No approved training pairs found.")
        sys.exit(1)

    path = train_lora(pairs, adapter_name=args.adapter_name)
    if path:
        logger.info("Done. Load with: vllm serve %s --enable-lora --lora-modules %s=%s",
                     DEFAULT_MODEL, args.adapter_name, path)
    else:
        logger.error("Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
