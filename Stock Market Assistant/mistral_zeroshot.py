from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load the dataset
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', split='train')

df = pd.DataFrame(dataset)

# Split the dataset into training and remainder
train_dataset, remainder_dataset = train_test_split(df, test_size=0.2, random_state=42)

# Split the remainder into validation and test sets
validation_dataset, test_dataset = train_test_split(remainder_dataset, test_size=0.5, random_state=42)

base_model_id = "mistralai/Mistral-7B-v0.1"

# Define functions for generating prompts
def generate_prompts(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive", "neutral" or "negative"[/INST]

            [{data_point["sentence"]}] = {data_point["label"]}
            """.strip()

def generate_test_val_prompt(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive","neutral","negative"[/INST]

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
    # Define labels and mapping
    labels = ['positive', 'neutral', 'negative']
    mapping = {'positive': 2, 'neutral': 1, 'negative': 0, 2: 2, 1: 1, 0: 0}

    # Map true and predicted labels
    y_true_mapped = [mapping[label] for label in y_true]
    y_pred_mapped = [mapping[label] for label in y_pred]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy for each label
    for label, idx in mapping.items():
        label_indices = [i for i, y in enumerate(y_true_mapped) if y == idx]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {label_accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=[0, 1, 2], target_names=labels)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=[0, 1, 2])
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
    label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["sentence"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer,
                        max_new_tokens=1, 
                        temperature=0.0,
                       )
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]['generated_text'].split("=")[-1].strip().lower()
        if 'ne' in answer or 'neg' in answer:
            y_pred.append(label_mapping['negative'])
        elif 'pos' in answer or 'po' in answer:
            y_pred.append(label_mapping['positive'])
        else:
            y_pred.append(label_mapping['neutral'])
    return y_pred

y_pred = predict(X_test, model, tokenizer)
test_dataset["prediction"] = y_pred
print("Unique labels in y_true:", set(y_test))
# evaluate(y_test, y_pred)
print("Unique labels in y_pred:", set(y_pred))
evaluate(y_test, y_pred)
test_dataset.to_csv("predictions_zeroshot__phrasebank.csv", index=False)
# print(X_train.head())
