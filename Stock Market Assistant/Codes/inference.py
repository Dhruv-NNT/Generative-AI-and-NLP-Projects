# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from datetime import datetime
# from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
# from datasets import load_dataset
# from peft import prepare_model_for_kbit_training,LoraConfig,PeftModel,get_peft_model, PeftModel
# import torch

# def predict_news_sentiment(model_path, dataset_path, output_path):
#     base_model_id = "mistralai/Mistral-7B-v0.1"
#     bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     base_model = AutoModelForCausalLM.from_pretrained(base_model_id, 
#                                              quantization_config = bnb_config, 
#                                              device_map = 'auto', 
#                                              trust_remote_code = "True")
    
#     ft_model = PeftModel.from_pretrained(base_model, "/home/msai/ar0001uv/repos/dnlp/logs/checkpoint-100")

#     ft_model.eval()
#     with torch.no_grad():
#         print(f'Tuned Model output:\n\n{eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=300)[0], skip_special_tokens=True)}')

#     # Load the custom dataset
#     dataset = pd.read_csv(dataset_path)

#     # Define a function for generating predictions
#     def predict_sentiment(summary):
#         pipe = pipeline(
#             task="text-generation", 
#             model=model, 
#             tokenizer=tokenizer,
#             max_new_tokens=1, 
#             temperature=0.0,
#         )
#         result = pipe(summary, pad_token_id=tokenizer.eos_token_id)
#         answer = result[0]['generated_text'].split("=")[-1].strip().lower()
#         return answer

#     # Generate predictions for each summary in the dataset
#     predictions = [predict_sentiment(summary) for summary in dataset["Summary"]]

#     # Save predictions to a CSV file
#     dataset["predicted_sentiment"] = predictions
#     dataset.to_csv(output_path, index=False)

# if __name__ == "__main__":
#     # Define the model path and dataset path
#     model_path = "/home/msai/ar0001uv/repos/dnlp/logs/checkpoint-100"  # Update with the path to the fine-tuned model directory
#     dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
#     output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

#     # Generate predictions for the news summaries using the pre-trained model
#     predict_news_sentiment(model_path, dataset_path, output_path)
##########################################################################################################
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig
# from peft import PeftModel

# def predict_news_sentiment(model_path, dataset_path, output_path):
#     base_model_id = "mistralai/Mistral-7B-v0.1"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,  # Mistral, same as before
#         quantization_config=bnb_config,  # Same quantization config as before
#         device_map="auto",
#         trust_remote_code=True)

#     # Load the fine-tuned model
#     ft_model = PeftModel.from_pretrained(base_model, "/home/msai/ar0001uv/repos/dnlp/logs/checkpoint-100")

#     # Set the model to evaluation mode
#     ft_model.eval()

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#     # Define a function for generating predictions
#     def predict_sentiment(summary):
#         pipe = pipeline(
#             task="text-generation", 
#             model=ft_model, 
#             tokenizer=tokenizer,
#             max_new_tokens=1, 
#             temperature=0.0,
#         )
#         result = pipe(summary, pad_token_id=tokenizer.eos_token_id)
#         answer = result[0]['generated_text'].split("=")[-1].strip().lower()
#         return answer

#     # Load the custom dataset
#     dataset = pd.read_csv(dataset_path)

#     # Generate predictions for each summary in the dataset
#     predictions = [predict_sentiment(summary) for summary in dataset["Summary"]]

#     # Save predictions to a CSV file
#     dataset["predicted_sentiment"] = predictions
#     dataset.to_csv(output_path, index=False)

# if __name__ == "__main__":
#     # Define the model path and dataset path
#     model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory
#     dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
#     output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

#     # Generate predictions for the news summaries using the fine-tuned model
#     predict_news_sentiment(model_path, dataset_path, output_path)
###################################################################################

# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig
# from peft import PeftModel

# def predict_news_sentiment(model_path, dataset_path, output_path):
#     # Define the base model ID
#     base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     # Configure BitsAndBytesConfig for quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     # Load the base model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     # Load the fine-tuned model
#     ft_model = PeftModel.from_pretrained(base_model, model_path)

#     # Set the model to evaluation mode
#     ft_model.eval()

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#     # Define a function for generating predictions
#     def predict_sentiment(summary):
#         pipe = pipeline(
#             task="text-generation", 
#             model=ft_model, 
#             tokenizer=tokenizer,
#             max_new_tokens=1, 
#             temperature=0.0,
#         )
#         result = pipe(summary, pad_token_id=tokenizer.eos_token_id)
#         answer = result[0]['generated_text'].split("=")[-1].strip().lower()
#         return answer

#     # Load the custom dataset
#     dataset = pd.read_csv(dataset_path)

#     # Generate predictions for each summary in the dataset
#     predictions = [predict_sentiment(summary) for summary in dataset["Summary"]]

#     # Save predictions to a CSV file
#     dataset["predicted_sentiment"] = predictions
#     dataset.to_csv(output_path, index=False)

# if __name__ == "__main__":
#     # Define the model path and dataset path
#     model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory
#     dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
#     output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

#     # Generate predictions for the news summaries using the fine-tuned model
#     predict_news_sentiment(model_path, dataset_path, output_path)
##################################################################################

# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig
# from peft import PeftModel
# import torch

# def generate_prompts(data_point):
#     return f"""
#             [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
#             determine if it is positive, neutral, or negative, and return the answer as 
#             the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

#             [{data_point["Summary"]}] = """.strip()

# def predict_news_sentiment(model_path, dataset_path, output_path):
#     # Define the base model ID
#     base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     # Configure BitsAndBytesConfig for quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     # Load the base model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     # Load the fine-tuned model
#     ft_model = PeftModel.from_pretrained(base_model, model_path)

#     # Set the model to evaluation mode
#     ft_model.eval()

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#     # Load the custom dataset
#     dataset = pd.read_csv(dataset_path)

#     # Apply function to create prompts for each summary in the dataset
#     dataset["prompt"] = dataset.apply(generate_prompts, axis=1)

#     # Define a function for generating predictions
#     def predict_sentiment(prompt):
#         pipe = pipeline(
#             task="text-generation", 
#             model=ft_model, 
#             tokenizer=tokenizer,
#             max_new_tokens=1, 
#             temperature=0.0,
#         )
#         result = pipe(prompt, pad_token_id=tokenizer.eos_token_id)
#         answer = result[0]['generated_text'].split("=")[-1].strip().lower()
#         return answer

#     # Generate predictions for each prompt in the dataset
#     predictions = [predict_sentiment(prompt) for prompt in dataset["prompt"]]

#     # Save predictions to a CSV file
#     dataset["predicted_sentiment"] = predictions
#     dataset.to_csv(output_path, index=False)

# if __name__ == "__main__":
#     # Define the model path and dataset path
#     model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory
#     dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
#     output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

#     # Generate predictions for the news summaries using the fine-tuned model
#     predict_news_sentiment(model_path, dataset_path, output_path)
######################################################################################

# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from transformers import BitsAndBytesConfig
# from peft import PeftModel
# import torch
# # import torch.nn as nn
# # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, logging
# # from datasets import load_dataset, Dataset
# # from peft import LoraModel, LoraConfig
# # import pandas as pd
# # from tqdm import tqdm
# # import numpy as np
# # from trl import SFTTrainer
# # from transformers import TrainerCallback

# def generate_prompts(data_point):
#     # print("Type of data_point:", type(data_point))
#     # print("Value of data_point:", data_point)
#     summary = data_point["Summary"]
#     return f"""
#             [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
#             determine if it is positive, neutral, or negative, and return the answer as 
#             the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

#             [{summary}] = """.strip()

# def predict_sentiment(prompt, tokenizer, ft_model):
#     pipe = pipeline(
#             task="text-generation", 
#             model=ft_model, 
#             tokenizer=tokenizer,
#             max_new_tokens=1, 
#             temperature=0.0,
#         )
#     result = pipe(prompt, pad_token_id=tokenizer.eos_token_id)
#     answer = result[0]['generated_text'].split("=")[-1].strip().lower()
#     print(answer)
#     return answer

# def predict_news_sentiment(model_path, dataset_path, output_path):
#     # Define the base model ID
#     base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     # Configure BitsAndBytesConfig for quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     # Load the base model
#     base_model = AutoModelForCausalLM.from_pretrained(
#         base_model_id,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     # Load the fine-tuned model
#     ft_model = PeftModel.from_pretrained(base_model, model_path)

#     # Set the model to evaluation mode
#     ft_model.eval()

#     ft_model.config.use_cache = False
#     ft_model.config.pretraining_tp = 1

#     tokenizer = AutoTokenizer.from_pretrained(base_model_id,
#                                           trust_remote_code=True,
#                                           padding_side="left",
#                                           add_bos_token=True,
#                                           add_eos_token=True,
#                                          )

#     tokenizer.pad_token = tokenizer.eos_token

#     # Load the custom dataset
#     dataset = pd.read_csv(dataset_path)

#     # Apply function to create prompts for each summary in the dataset
#     dataset["prompt"] = dataset.apply(lambda row: generate_prompts(row), axis=1)    

#     # Generate predictions for each prompt in the dataset
#     predictions = [predict_sentiment(prompt, tokenizer, ft_model) for prompt in dataset["prompt"]]

#     # Save predictions to a CSV file
#     dataset["predicted_sentiment"] = predictions
#     dataset.to_csv(output_path, index=False)

# if __name__ == "__main__":
#     # Define the model path and dataset path
#     model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory
#     dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
#     output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

#     # Generate predictions for the news summaries using the fine-tuned model
#     predict_news_sentiment(model_path, dataset_path, output_path)

#Correct version to some extent is above
#########################################################################

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch

def generate_prompts(data_point):
    summary = data_point["Summary"]
    return f"""
            [Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[{summary}] = """.strip()

def predict_sentiment(prompt, tokenizer, ft_model):
    pipe = pipeline(
            task="text-generation", 
            model=ft_model, 
            tokenizer=tokenizer,
            max_new_tokens=10, 
            temperature=0,
        )
    result = pipe(prompt, pad_token_id=tokenizer.eos_token_id)
    answer = result[0]['generated_text'].split("=")[-1].strip().lower()
    if "positive" in answer.lower():
        answer = "positive"
    elif "negative" in answer.lower():
        answer = "negative"
    else:
        answer = "neutral"
    return answer

def predict_news_sentiment(model_path, dataset_path, output_path):
    # Define the base model ID
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

    # Load the custom dataset
    dataset = pd.read_csv(dataset_path)

    # Apply function to create prompts for each summary in the dataset
    dataset["prompt"] = dataset.apply(lambda row: generate_prompts(row), axis=1)    

    # Generate predictions for each prompt in the dataset
    predictions = [predict_sentiment(prompt, tokenizer, ft_model) for prompt in dataset["prompt"]]

    # Save predictions to a CSV file
    dataset["predicted_sentiment"] = predictions
    dataset.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Define the model path and dataset path
    model_path = "/home/msai/ar0001uv/repos/dnlp/trained-model"  # Update with the path to the fine-tuned model directory
    dataset_path = "/home/msai/ar0001uv/repos/dnlp/articles.csv"  # Update with the path to the custom dataset CSV file
    output_path = "predictions_test_set.csv"  # Path to save the predictions CSV file

    # Generate predictions for the news summaries using the fine-tuned model
    predict_news_sentiment(model_path, dataset_path, output_path)