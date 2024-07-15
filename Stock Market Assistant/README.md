# 🚀 Advanced Stock Market Analysis with Deep Learning and NLP

Welcome to the Advanced Stock Market Analysis project! This project leverages state-of-the-art Natural Language Processing (NLP) techniques and cutting-edge deep learning models to revolutionize stock market analysis by extracting actionable insights from vast amounts of unstructured data from news articles and social media.

## 🎯 Project Overview

This project aims to democratize financial expertise by automating the extraction of actionable insights from complex financial data. Key functionalities include:

- **Sentiment Analysis**: Using advanced techniques to provide reliable and actionable insights.
- **Text Summarization**: Implementing efficient summarization techniques that condense information into concise and informative summaries.
- **User-Friendly Integration**: Offering a seamless, free-to-use platform for enhanced market analysis.

## 🚀 Features

- **Improved Summarization**: Captures critical nuances while condensing information.
- **Enhanced Sentiment Analysis**: Ensures accurate sentiment detection, especially crucial in financial contexts.
- **Seamless Integration**: Free and easy-to-use platform without the need for deep financial knowledge.

## 🛠️ Technologies Used

- **Large Language Models (LLMs)**: Mistral, Gemma
- **Parameter-Efficient Fine-Tuning (PEFT)**: Techniques to fine-tune LLMs with minimal computational resources.
- **Quantized Low-Rank Adaptation (QLoRA)**: Efficient adaptation method for LLMs.
- **Sentiment Analysis**: Advanced NLP techniques
- **Text Summarization**: BERT (Bidirectional Encoder Representations) Summarizer
- **Embedding Storage and Retrieval**: FAISS (Facebook AI Similarity Search)

## 📁 Project Structure

The repository is structured as follows:

```
📦Stock Market Assistant Project
 ┣ 📂Codes
 ┃ ┣ 📜finetuned_accuracy.py         # Script for evaluating fine-tuned model accuracy
 ┃ ┣ 📜inference.py                  # Script for running inference on new data
 ┃ ┣ 📜llm_SFT.py                    # Script for supervised fine-tuning of LLM
 ┃ ┣ 📜llm_sft_downsampling.py       # Script for fine-tuning with downsampling
 ┃ ┣ 📜llm_sft_upsampling.py         # Script for fine-tuning with upsampling
 ┃ ┣ 📜mistral_zeroshot.py           # Script for zero-shot predictions with Mistral on financial phrasebank data for sentiment classification
 ┃ ┣ 📜run.sh                        # Shell script to run the application
 ┃ ┗ 📜Streamlit_app.py              # Streamlit application for user interaction
 ┣ 📂Necessary Datasets
 ┃ ┣ 📜one_month_news_summary.json   # JSON file with summarized news articles
 ┃ ┣ 📜predictions_test_set.csv      # CSV file with test set predictions
 ┃ ┗ 📜stock_data.csv                # CSV file with stock data
```

## 📜 Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Data Preparation

Prepare and preprocess the news articles by running:

```bash
python Codes/finetuned_accuracy.py
```

### Sentiment Analysis

Run sentiment analysis on the prepared data:

```bash
python Codes/inference.py
```

### Text Summarization

Generate summaries of the financial news articles:

```bash
python Codes/llm_SFT.py
```

### Zero-Shot Predictions

Run zero-shot predictions using the Mistral model:

```bash
python Codes/mistral_zeroshot.py
```

### Running the Streamlit App

To launch the Streamlit app for interactive use, run:

```bash
streamlit run Codes/Streamlit_app.py
```

### Shell Script

You can also use the provided shell script to run the application:

```bash
sh Codes/run.sh
```

## 🎥 Demo

Check out our demo video showc asing the key features and functionalities of the project.

https://github.com/user-attachments/assets/c586daaf-7449-4657-8a04-fbd112a36905



## 📊 Results

We achieved an overall accuracy of 23% on the Financial PhraseBank dataset using zero-shot techniques with the Mistral model, with significant improvements needed in handling financial language nuances.

## 📈 Performance Metrics

- **Accuracy**:
  - Positive: 7%
  - Neutral: 28.1%
  - Negative: 62.5%
- **Training Loss**: Detailed in the notebooks

## 🔍 Future Work

- Address class imbalance and model generalization challenges
- Enhance domain-specific tuning for financial contexts
- Further research in financial sentiment analysis

## 📧 Contact

For any queries or collaboration, feel free to reach out to us.

## 🎉 Acknowledgements

Special thanks to all contributors and the open-source community for their invaluable support. 

---

Feel free to customize this template further based on the specific details of your project. If there are specific points from the walkthrough video you want to include, please share those details, and I will integrate them accordingly.
