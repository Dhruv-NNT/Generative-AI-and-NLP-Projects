import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm

def generate_prompts(data_point):
    summary = data_point["sentence"]
    return f"""
            [Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[{summary}] = """.strip()

def predict_sentiment(X_test, model, tokenizer):
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

if __name__ == "__main__":
    # Define the model path
    model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory

    # Load the dataset
    dataset = load_dataset('financial_phrasebank', 'sentences_50agree', split='train')
    df = pd.DataFrame(dataset)

    # Split the dataset into training and remainder
    train_dataset, remainder_dataset = train_test_split(df, test_size=0.2, random_state=42)

    # Split the remainder into validation and test sets
    validation_dataset, test_dataset = train_test_split(remainder_dataset, test_size=0.5, random_state=42)

    # Apply functions to create prompts for training, validation, and testing data
    X_train = pd.DataFrame(train_dataset.apply(generate_prompts, axis=1), columns=["sentence"])
    X_validation = pd.DataFrame(validation_dataset.apply(generate_prompts, axis=1), columns=["sentence"])
    X_test = pd.DataFrame(test_dataset.apply(generate_prompts, axis=1), columns=["sentence"])

    # Load the base model
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # Configure BitsAndBytesConfig for quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load the fine-tuned model
    ft_model = PeftModel.from_pretrained(base_model, model_path)

    # Set the model to evaluation mode
    ft_model.eval()

    ft_model.config.use_cache = False
    ft_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                          trust_remote_code=True,
                                          padding_side="left",
                                          add_bos_token=True,
                                          add_eos_token=True,
                                         )

    tokenizer.pad_token = tokenizer.eos_token

    # Generate predictions for the test set
    y_test = test_dataset["label"]
    y_pred = predict_sentiment(X_test, ft_model, tokenizer)

    # Evaluate the predictions
    evaluate(y_test, y_pred)
