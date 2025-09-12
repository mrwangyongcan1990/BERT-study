from datasets import load_dataset
import evaluate

#from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import numpy as np
from transformers import TrainingArguments

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Example: use a built-in dataset; replace with load_dataset("csv", data_files=...)
raw = load_dataset("imdb")

# Create a validation set from the training set (10% of train)
split = raw["train"].train_test_split(test_size=0.1, seed=42)
raw["train"] = split["train"]
raw["validation"] = split["test"]

# Keep the original test set
raw["test"] = raw["test"]

# Split test set into validation + test
#raw = raw["test"].train_test_split(test_size=0.5, seed=42)
#raw["train"] = load_dataset("imdb", split="train")

#raw = load_dataset("imdb")            # example binary sentiment dataset
#raw = raw.shuffle(seed=42).train_test_split(test_size=0.1)
#raw["validation"] = raw["test"]

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized = raw.map(preprocess, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Metrics
import evaluate
#metric = evaluate.load("accuracy")
#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    preds = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=preds, references=labels)
# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


training_args = TrainingArguments(
    output_dir="./bert-finetune",
    #    evaluation_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,               # if your GPU/torch supports it
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
model.save_pretrained("./best-bert")
tokenizer.save_pretrained("./best-bert")
