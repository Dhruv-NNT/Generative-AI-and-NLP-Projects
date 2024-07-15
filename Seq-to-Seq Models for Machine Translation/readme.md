# 🌐 Seq-to-Seq Models for Machine Translation

Welcome to the Seq-to-Seq Models for Machine Translation project. This project explores various sequence-to-sequence (Seq2Seq) models, focusing on their performance in translating text from one language to another. This work is part of the AI6127 - Deep Neural Networks for Natural Language Processing course.

## 📝 Project Overview

In this project, I investigate different Seq2Seq models for machine translation tasks. The project includes experiments with various neural network architectures, such as LSTM, BiLSTM, and Transformer models. I evaluate these models based on their translation accuracy and performance metrics.

## 🌟 Features

- **Model Architectures**: Experiments with different Seq2Seq architectures, including LSTM, BiLSTM, and Transformer.
- **Optimizers**: Evaluation of models with various optimizers (SGD, Adam, Adagrad).
- **Hyperparameters**: Tuning of hyperparameters like epochs, learning rate, and batch size.
- **Pre-trained Embeddings**: Use of pre-trained Word2Vec embeddings to enhance model performance.
- **Performance Metrics**: Detailed analysis using BLEU score and other relevant metrics.

## 📂 Project Structure

```
📦Seq-to-Seq Models for Machine Translation
 ┣ 📜Assignment_1_Simple_Sentiment_Analysis.ipynb  # Jupyter notebook with code and experiments
 ┣ 📜Assignment Report - Aradhya Dhruv.pdf         # Detailed assignment report
 ┣ 📜README.md                                     # Project overview and instructions
 ┗ 📜requirements.txt                              # Dependencies for the project
```

## 📈 Results Summary

### Optimizers

- **SGD**: Slower convergence, sensitive to learning rate.
- **Adam**: Effective but requires careful tuning of learning rate to avoid overfitting.
- **Adagrad**: Shows initial promise but can lead to overfitting after a certain point.

### Epochs

- Training beyond 10 epochs shows diminishing returns and potential overfitting.
- Pre-trained Word2Vec embeddings significantly improve accuracy but introduce erratic training behavior.

### Neural Network Architectures

- **LSTM**: Evaluated for its ability to handle long-range dependencies in sequence data.
- **BiLSTM**: Assessed for its potential to capture context from both directions in a sequence.
- **Transformer**: Analyzed for its efficiency in parallel processing and capturing complex dependencies.

### Model Performance

- **Feedforward Neural Networks (FFN)**: Show varying degrees of overfitting and limited improvement with added layers.
- **CNN**: Achieves the highest accuracy (88.81%) due to its position-invariant properties and hierarchical feature learning.
- **LSTM**: Competitive accuracy but struggles with long-range dependencies.
- **BiLSTM**: Lower accuracy and precision-recall balance issues compared to other models.

## 📦 Requirements

- Python 3.9
- Jupyter Notebook
- Libraries: PyTorch, TensorFlow, Keras, NumPy, Pandas, Matplotlib, NLTK, PyPDF2

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/seq-to-seq-machine-translation.git
cd seq-to-seq-machine-translation
```

2. **Create a virtual environment and activate it:**

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

## 🚀 Running the Experiments

1. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open the `Assignment_1_Simple_Sentiment_Analysis.ipynb` notebook and run the cells to execute the experiments.**

## 📊 Detailed Results

Refer to the `Assignment Report - Aradhya Dhruv.pdf` for detailed results, including:

- Performance metrics for different Seq2Seq models
- Analysis of training and validation loss curves
- Insights on the impact of pre-trained embeddings and hyperparameter tuning

## 🔍 Future Work

- Experiment with additional Seq2Seq architectures and attention mechanisms.
- Apply techniques like beam search to improve translation accuracy.
- Fine-tune pre-trained embeddings to better align with the training data.

## 📧 Contact

For any questions or feedback, feel free to reach out to:

- **Name**: Aradhya Dhruv
- **Email**: aradhya.dhruv@gmail.com
