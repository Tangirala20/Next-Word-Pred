Next Word Prediction using LSTM (NLP Project)
Overview:
This project implements a deep learning model using LSTM (Long Short-Term Memory) networks to predict the next word in a given sequence of text. It is a classic Natural Language Processing (NLP) task designed to demonstrate how sequential models can learn the structure of language.

The project is built using PyTorch and processes text data from a literary corpus to train and test a language model.

Objective:
To develop a next word prediction model that can:

Learn contextual patterns from a text corpus.

Predict the most probable next word given a partial sentence.

Be evaluated for its accuracy and loss metrics.

Dataset
Source: Public domain text (e.g., Shakespeare or any literary corpus)

Format: Plain text

Preprocessing: Lowercasing, tokenization, stopword removal, sequence creation.

Model Architecture:
Embedding Layer: Transforms word indices into dense vector representations.

LSTM Layer: Learns temporal dependencies from word sequences.

Linear + Softmax Layer: Outputs probability distribution over the vocabulary.

Technologies Used:
Language: Python

Libraries:

PyTorch (for model development and training)

NLTK (for tokenization and stopword handling)

NumPy, Matplotlib (for analysis and plotting)

Setup Instructions
Clone this repository or download the notebook:

bash
Copy
Edit
git clone <repo-url>
Create and activate a Python environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required libraries:

bash
Copy
Edit
pip install torch nltk matplotlib
Download required NLTK resources:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook Next_Word_Project.ipynb
Key Steps in the Notebook
Text Cleaning and Preprocessing

Tokenization and Vocabulary Creation

Sequence Generation for Training

Model Definition using PyTorch

Training Loop and Loss Monitoring

Prediction Function Implementation

Evaluation and Testing

Example Usage
python
Copy
Edit
input_text = "he was not"
predicted_word = predict_next_word(model, input_text)
print("Next word:", predicted_word)
Results:
Training Loss: Reduced steadily across epochs.

Accuracy: Improved with increased training data and tuning.

Model Evaluation: Qualitative examples demonstrate logical predictions.

Future Enhancements
Use pretrained embeddings like GloVe or Word2Vec.

Implement beam search for better prediction.

Train on larger and more diverse datasets (e.g., Wikipedia).

Deploy as an API using FastAPI or a web app using Streamlit.

