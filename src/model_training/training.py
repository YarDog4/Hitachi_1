from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load & prep dataset
df = pd.read_csv(file_path)
df = df[['text', 'category']]
label2id = {label: i for i, label in enumerate(df['category'].unique())}
df['label'] = df['category'].map(label2id)

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split
train_test = dataset.train_test_split(test_size=0.2)

# Fine-tune
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"]
)

trainer.train()

# Predict
inputs = tokenizer(article, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
predicted_class_id = torch.argmax(outputs.logits).item()
print("Predicted category:", list(label2id.keys())[list(label2id.values()).index(predicted_class_id)])
