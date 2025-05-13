# N-Gram Language Model 📚

This project implements an **N-Gram Language Model** from scratch to generate and predict sequences of text based on learned probability distributions. The model estimates the probability of a word given its previous context using n-gram statistics, and can be used for tasks such as sentence generation and probability evaluation.

## Project Overview

- **Corpus Preparation:** 
  - Trained the model on text data sourced from the Gutenberg Library (public domain books).
  - Preprocessed text by lowercasing, tokenizing, and cleaning to standardize inputs.

- **N-Gram Model Building:**
  - Built N-gram models (where *n* can vary) to estimate the probability of word sequences.
  - Counted occurrences of N-grams and computed conditional probabilities.

- **Sentence Generation:**
  - Developed a system to generate random but statistically probable sentences based on the learned language model.

- **Sentence Probability Calculation:**
  - Calculated the probability of any given sentence occurring based on the trained N-gram model.

## Demo

👉 [**View Project Demo (nbviewer)**](https://nbviewer.org/github/neildewan7/n-gram-model/blob/main/cleaned_project.ipynb)

## Key Features

- Flexible N-gram size (`n` is configurable by the user).
- Can generate new sentences word-by-word using learned probabilities.
- Can evaluate and assign a probability score to any custom input sentence.
- Models trained on real-world book corpus for realistic text distributions.

## Technologies Used

- **Python**
- **Pandas** (optional, for corpus handling)
- **NumPy** (numerical calculations)
- **Random** (random selection weighted by probabilities)

## Project Structure

```plaintext
cleaned_project.ipynb    # Final notebook with full model training and examples
project-validation.py    # Helper functions and validation scripts
data/                    # Folder containing training corpus (e.g., Gutenberg text files)
images/                  # (Optional) Folder for generated plots or examples
README.md                # Project overview and instructions
