import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    AdaLoraConfig,
    TaskType,
    PeftType
)

# 选择模型和数据集
model_name = "gpt2"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(model_name)

# 1. LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
lora_model = get_peft_model(model, lora_config)

# 2. Prefix Tuning 配置
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20
)
prefix_model = get_peft_model(model, prefix_config)

# 3. P-Tuning 配置
ptuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20
)
ptuning_model = get_peft_model(model, ptuning_config)

# 4. Prompt Tuning（与P-Tuning类似，区别在于初始化方式和应用场景）
prompt_tuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT"
)
prompt_tuning_model = get_peft_model(model, prompt_tuning_config)

# 5. AdaLoRA 配置
adalora_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_r=4,
    init_r=12,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10
)
adalora_model = get_peft_model(model, adalora_config)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=20,
    save_total_limit=1,
    fp16=True,
    report_to=[]
)

# 选择一种方法进行训练（以LoRA为例，其他方法替换lora_model即可）
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 你可以通过替换trainer中的model参数，分别训练和比较不同的微调方法
# LoRA: lora_model
# Prefix Tuning: prefix_model
# P-Tuning: ptuning_model
# Prompt Tuning: prompt_tuning_model
# AdaLoRA: adalora_model

# 训练完成后可保存模型
lora_model.save_pretrained("./lora_model")