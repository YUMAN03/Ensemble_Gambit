# Ensemble-Gambit Hangman Solver

This project implements a sophisticated Hangman solver using an ensemble of deep learning models. The final model combines the strengths of a Bi-LSTM, a CharCNN, and a Transformer to achieve a high win rate by learning the intricate patterns of the English language.

This was developed as part of a course on Natural Language Processing (NLP) and Large Language Models (LLMs).

## 🎯 Core Strategy

The problem is framed as a **Masked Language Modeling** task. The model is given a word with some letters hidden (e.g., `a p p _ e`) and is trained to predict the missing characters.

The final solver is not a single model but an **ensemble** of three specialist models whose predictions are combined through a weighted sum:

1. **🧠 Bidirectional LSTM (Bi-LSTM):** The sequential expert. It reads the word in both directions (left-to-right and right-to-left) to understand the immediate context of each character.

2. **✨ Character-level CNN (CharCNN):** The spatial pattern detector. It uses convolutional filters to identify common and meaningful character chunks (morphemes) like prefixes (`un-`, `re-`) and suffixes (`-ing`, `-able`).

3. **⚙️ Transformer:** The attention specialist. It uses a self-attention mechanism to weigh the importance of every character in relation to all other characters in the word, capturing long-range dependencies.

By combining these three distinct approaches, the solver makes more robust and intelligent guesses than any single model could alone.

## ✨ Key Features

* **Dynamic Data Augmentation:** Instead of creating a massive, static dataset of masked words, new masks are generated **on-the-fly** for every word in every epoch. This creates a near-infinite stream of unique training examples and prevents the model from simply memorizing answers.

* **Intelligent Prediction Logic:** The model doesn't just guess the most likely letter for a single blank. Instead, it calculates the probability of every unguessed letter across *all* available blanks and chooses the one with the highest overall chance of being in the word.

* **Positional Encoding:** The Transformer model uses sinusoidal positional encodings to inject information about the absolute position of each character in the word, allowing it to understand character ordering without relying on sequential processing.

* **Smart Padding Strategy:** Words of varying lengths are padded to a uniform size for batch processing. The models are designed to ignore padding tokens during training and inference, ensuring they focus only on actual word content.

* **Modern Training Techniques:** The models leverage a `ReduceLROnPlateau` learning rate scheduler and Gradient Clipping to ensure stable and effective training.

* **GPU Acceleration:** The code is configured to run on a specific GPU, with the `TARGET_GPU_ID` variable making it easy to switch between different hardware setups.

## 📈 Project Journey & Results

1. **Initial Model (Bi-LSTM):** The project started with a standalone Bi-LSTM. After significant hyperparameter tuning and implementing the "best bet" prediction logic, its performance plateaued at **56% accuracy**.

2. **Introducing the Ensemble:** To break past the plateau, the idea of an ensemble was born. A CharCNN was added to capture word-form patterns, and a Transformer was added to understand global character relationships.

3. **Final Result:** After training all three models and fine-tuning the weights for their combined prediction, the final **Ensemble-Gambit Solver** achieved a win rate of **62%**.