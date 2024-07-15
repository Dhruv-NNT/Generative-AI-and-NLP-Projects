from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, logging
from datasets import load_dataset, Dataset
from peft import LoraModel, LoraConfig
import pandas as pd
from tqdm import tqdm
import numpy as np
from trl import SFTTrainer
from transformers import TrainerCallback
from huggingface_hub import login

login(token="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")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', split='train')

# subset_size = 1000  # Define the size of the subset
# dataset_subset = dataset.shuffle(seed=42).select(range(subset_size))

df = pd.DataFrame(dataset)

# Split the dataset into training and remainder
train_dataset, remainder_dataset = train_test_split(df, test_size=0.2, random_state=42)

# Count the number of samples in each class
class_counts = train_dataset['label'].value_counts()

# Determine the number of samples in the negative class
min_samples = class_counts.min()

# Downsample the positive and neutral classes to match the number of samples in the negative class
downsampled_dataset = pd.concat([
    train_dataset[train_dataset['label'] == 0],  # Negative class
    train_dataset[train_dataset['label'] == 1].sample(min_samples, random_state=42),  # Neutral class
    train_dataset[train_dataset['label'] == 2].sample(min_samples, random_state=42)  # Positive class
])

# Shuffle the downsampled dataset
downsampled_dataset = downsampled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the downsampled dataset into training and remainder
train_dataset, remainder_dataset = train_test_split(downsampled_dataset, test_size=0.2, random_state=42)

# Split the remainder into validation and test sets
validation_dataset, test_dataset = train_test_split(remainder_dataset, test_size=0.5, random_state=42)

# Define functions for generating prompts
def generate_prompts(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

            [{data_point["sentence"]}] = {data_point["label"]}
            """.strip()

def generate_test_val_prompt(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

            [{data_point["sentence"]}] = """.strip()

# Apply functions to create prompts for training, validation, and testing data
X_train = pd.DataFrame(train_dataset.apply(generate_prompts, axis=1), columns=["sentence"])
X_validation = pd.DataFrame(validation_dataset.apply(generate_test_val_prompt, axis=1), columns=["sentence"])
X_test = pd.DataFrame(test_dataset.apply(generate_test_val_prompt, axis=1), columns=["sentence"])

train_data = Dataset.from_pandas(X_train)
val_data = Dataset.from_pandas(X_validation)
eval_data = Dataset.from_pandas(X_test)

# Extract true labels for testing and validation data
y_test = test_dataset["label"]
y_val = validation_dataset["label"]
y_train = train_dataset["label"]
### Evaluation function
def evaluate(y_true, y_pred):
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'none':1, 'negative': 0}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

print("Train Length:", len(train_dataset))
print("Validation Length:", len(validation_dataset))
print("Test Length:", len(test_dataset))

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                          padding_side="left",
                                          add_bos_token=True,
                                          add_eos_token=True,
                                         )

tokenizer.pad_token = tokenizer.eos_token

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["sentence"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens = 1, 
                        temperature = 0.0,
                       )
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]['generated_text'].split("=")[-1].lower()
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred

# y_pred = predict(X_test, model, tokenizer)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # 4
    optim="paged_adamw_32bit",
    save_steps=1,
    save_strategy='epoch', #added save strategy
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="epoch", 
    do_eval=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,
    dataset_text_field="sentence",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=1024
)

trainer.train()

trainer.model.save_pretrained("trained-model-downsampled")