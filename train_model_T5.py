import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import json

# Load pre-trained model and tokenizer
model_name = 't5-small'  # Or use 't5-base' or 't5-large' for more capacity
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

train_file_path = "train.json"
val_file_path = "val.json"

# Chuyển dữ liệu thành định dạng phù hợp cho Hugging Face Dataset
data_processed = []
# Đọc file JSON
with open(train_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        for wrong_sentence in data["wrong_sentences"]:
            data_processed.append({
                "input": wrong_sentence,
                "output": data["sentence"]
            })

data_validate = []
# Đọc file JSON
with open(val_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        for wrong_sentence in data["wrong_sentences"]:
            data_processed.append({
                "input": wrong_sentence,
                "output": data["sentence"]
            })

# Chuyển đổi dữ liệu thành Dataset của Hugging Face
train_data = Dataset.from_dict({
    'input': [item['input'] for item in data_processed],
    'output': [item['output'] for item in data_processed]
})

val_data = Dataset.from_dict({
    'input': [item['input'] for item in data_validate],
    'output': [item['output'] for item in data_validate]
})



## Hàm chuẩn bị dữ liệu đầu vào và đầu ra
def preprocess_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['output']]

    # Tokenize inputs và outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Áp dụng hàm tiền xử lý cho dữ liệu
train_dataset = train_data.map(preprocess_function, batched=True)
val_dataset = val_data.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('../model')
tokenizer.save_pretrained('../model')
