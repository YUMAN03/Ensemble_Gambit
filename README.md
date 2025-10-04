# Ensemble-Gambit Hangman Solver

This project implements a sophisticated Hangman solver using an ensemble of deep learning models. The final model combines the strengths of a Bi-LSTM, a CharCNN, and a Transformer to achieve a high win rate by learning the intricate patterns of the English language.

This was developed as part of a course on Natural Language Processing (NLP) and Large Language Models (LLMs).

## 🎯 Core Strategy

The problem is framed as a **Masked Language Modeling** task. In the classic Hangman game, the model starts with a word where all letters are hidden (e.g., `_ _ _ _ _`) and must guess letters one at a time. As correct guesses are made, those letters are revealed in their positions (e.g., `a _ _ _ e`), and the model uses this partial information to predict the next most likely letter.

The final solver is not a single model but an **ensemble** of three specialist models whose predictions are combined through a weighted sum:

1. **🧠 Bidirectional LSTM (Bi-LSTM):** The sequential expert. It reads the word pattern in both directions (left-to-right and right-to-left) to understand the immediate context around each revealed and hidden character.

2. **✨ Character-level CNN (CharCNN):** The spatial pattern detector. It uses convolutional filters to identify common and meaningful character chunks (morphemes) like prefixes (`un-`, `re-`) and suffixes (`-ing`, `-able`).

3. **⚙️ Transformer:** The attention specialist. It uses a self-attention mechanism to weigh the importance of every character position in relation to all other positions in the word, capturing long-range dependencies.

By combining these three distinct approaches, the solver makes more robust and intelligent guesses than any single model could alone.

## ✨ Key Features

* **Dynamic Data Augmentation:** Instead of creating a massive, static dataset of masked words, new masks are generated **on-the-fly** for every word in every epoch. This simulates the real Hangman game by creating various stages of partial word revelation, from completely blank to nearly complete.

* **Intelligent Prediction Logic:** The model doesn't just guess the most likely letter for a single blank. Instead, it calculates the probability of every unguessed letter across *all* available blanks and chooses the one with the highest overall chance of being in the word.

* **Positional Encoding:** The Transformer model uses sinusoidal positional encodings to inject information about the absolute position of each character in the word, allowing it to understand character ordering without relying on sequential processing.

* **Smart Padding Strategy:** Words of varying lengths are padded to a uniform size for batch processing. The models are designed to ignore padding tokens during training and inference, ensuring they focus only on actual word content.

* **Modern Training Techniques:** The models leverage a `ReduceLROnPlateau` learning rate scheduler and Gradient Clipping to ensure stable and effective training.

* **GPU Acceleration:** The code is configured to run on a specific GPU, with the `TARGET_GPU_ID` variable making it easy to switch between different hardware setups.

## 📈 Project Journey & Results

1. **Initial Model (Bi-LSTM):** The project started with a standalone Bi-LSTM. After significant hyperparameter tuning and implementing the "best bet" prediction logic, its performance plateaued at **56% accuracy**.

2. **Introducing the Ensemble:** To break past the plateau, the idea of an ensemble was born. A CharCNN was added to capture word-form patterns, and a Transformer was added to understand global character relationships.

3. **Final Result:** After training all three models and fine-tuning the weights for their combined prediction, the final **Ensemble-Gambit Solver** achieved a win rate of **62%**.