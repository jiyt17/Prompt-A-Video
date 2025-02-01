import torch
import json
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments

from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader

# data
dataset = json.load(open('/path/to/dpo.json', 'r'))
train_dataset = Dataset.from_list(dataset)

# model
model_name = '/path/to/base_model'

max_seq_length = 2048 # Supports automatic RoPE Scaling, so choose any number.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False, # Use 4bit quantization to reduce memory usage. Can be False.
)
print(type(model)) # LlamaForCausalLM
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)
print(type(model)) # PeftModelForCausalLM


dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs/booster_dpo_lora",
    ),
    beta = 0.1,
    train_dataset = train_dataset,
    # eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

dpo_trainer.train()