"""
MyClone - Fine-tune on chat data using Unsloth + LoRA.

Kaggle quickstart:
  !pip install unsloth
  !git clone <your-repo> MyClone && cd MyClone
  !python train.py --dataset /kaggle/input/YOUR_DATASET/sft

Local:
  pip install unsloth
  python train.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 2048


def find_dataset_dir():
    local = Path("./dataset/sft")
    if (local / "sft-my.json").exists():
        return str(local)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if (d / "sft" / "sft-my.json").exists():
                return str(d / "sft")
            if (d / "sft-my.json").exists():
                return str(d)

    return str(local)


def load_sft_dataset(dataset_dir):
    data_file = os.path.join(dataset_dir, "sft-my.json")
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found")
        print("Use --dataset to specify the directory containing sft-my.json")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    conversations = []
    for item in raw:
        msgs = []
        if item.get("system"):
            msgs.append({"role": "system", "content": item["system"]})
        msgs.extend(item["messages"])
        conversations.append({"messages": msgs})

    print(f"Loaded {len(conversations)} conversations from {data_file}")
    return Dataset.from_list(conversations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Directory containing sft-my.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    dataset_dir = args.dataset or find_dataset_dir()
    is_kaggle = os.path.exists("/kaggle")
    output_dir = args.output or ("/kaggle/working/output" if is_kaggle else "./output")

    # Load data
    dataset = load_sft_dataset(dataset_dir)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Add LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Format conversations using chat template
    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, num_proc=1)

    # Train
    print(f"\nTraining: {args.epochs} epochs, bs={args.batch_size}x{args.grad_accum}, "
          f"lr={args.lr}, rank={args.lora_rank}")
    print(f"Output:   {output_dir}\n")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=100,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataset_num_proc=1,
        ),
    )

    trainer.train()

    # Save final adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nDone! LoRA adapter saved to: {output_dir}")
    if is_kaggle:
        print("Download from the notebook's Output tab.")


if __name__ == "__main__":
    main()
