import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, load_metric
from peft import LoraConfig, get_peft_model, QLoraConfig, apply_qlora

# Load the CoLA dataset
dataset = load_dataset("glue", "cola")
metric = load_metric("glue", "cola")

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the data
def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split the data
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["validation"]

# Define the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Apply LoRA
lora_config = LoraConfig(
    target_modules=["classifier"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# Apply QLoRA
model = apply_qlora(model, bits=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: metric.compute(predictions=p.predictions.argmax(-1), references=p.label_ids),
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Evaluation result: {eval_result}")
