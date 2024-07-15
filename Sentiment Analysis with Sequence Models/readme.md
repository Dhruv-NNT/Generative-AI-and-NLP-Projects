# 📊 Sentiment Analysis Using Sequence Models

Welcome to my Sentiment Analysis project, where I explore various neural network architectures and optimizers to perform sentiment analysis on text data. This project is part of the AI6127 - Deep Neural Networks for Natural Language Processing course.

## 📝 Project Overview

In this project, I present a comprehensive exploration of sentiment analysis models, encompassing experiments with various optimizers, epochs, and pre-trained embeddings. I evaluate different neural network architectures, including feedforward networks, convolutional neural networks (CNN), long short-term memory (LSTM), and bidirectional LSTM (BiLSTM), based on their performance using accuracy, precision, recall, and F1-score metrics.

## 🌟 Features

- Experiments with different optimizers: SGD, Adam, Adagrad
- Evaluation of models trained for various epochs
- Comparison of models using random and pre-trained embeddings
- Analysis of various neural network architectures (FFN, CNN, LSTM, BiLSTM)
- Detailed performance metrics and model comparisons

## 📂 Project Structure

```
📦Sentiment Analysis Using Sequence Models
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

- **Feedforward Neural Networks (FFN)**: Show varying degrees of overfitting and limited improvement with added layers.
- **CNN**: Achieves the highest accuracy (88.81%) due to its position-invariant properties and hierarchical feature learning.
- **LSTM**: Competitive accuracy but struggles with long-range dependencies.
- **BiLSTM**: Lower accuracy and precision-recall balance issues compared to other models.

## 📦 Requirements

- Python 3.9
- Jupyter Notebook
- Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, NLTK, PyPDF2

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/sentiment-analysis-sequence-models.git
cd sentiment-analysis-sequence-models
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

- Performance metrics for different models and optimizers
- Analysis of training and validation loss curves
- Insights on the impact of pre-trained embeddings

## 🔍 Future Work

- Experiment with hyperparameters like learning rate, batch size, and RNN architecture.
- Apply techniques like dropout or L1/L2 regularization to prevent overfitting.
- Fine-tune pre-trained embeddings to better align with the training data.

## 📧 Contact

For any questions or feedback, feel free to reach out to:

- **Name**: Aradhya Dhruv
- **Email**: aradhya.dhruv@gmail.com
