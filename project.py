# project.py
#sources: CHatGPT


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time



# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    
    time.sleep(0.5)

    response = requests.get(url)
    
    book = response.text.replace('\r\n', '\n')

    start_indx = book.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    
    start_indx = book.find("***", start_indx + len(start)) + 3
    
    end_indx = book.find( "*** END OF THE PROJECT GUTENBERG EBOOK")

    return book[start_indx:end_indx]

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):

    book_string = book_string.strip()

    book_string = re.sub(r'\n{2,}', '\x03\x02', book_string)
    
    book_string = '\x02' + book_string + '\x03'

    book_tokens = re.findall(r'\x02|\x03|[\w]+|[^\w\s]', book_string)
    
    return book_tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):

        unique = pd.Series(tokens).unique()

        probabilities = [1/ len(unique)] * len(unique)
        
        return pd.Series(data=probabilities, index=unique)
        
    
    def probability(self, words):

        if not all(word in self.mdl for word in words):
            return 0

        probabilities = self.mdl.loc[list(words)]
        
        return float(probabilities.prod())
        
    def sample(self, M):
        
        probabilities = self.mdl.values

        tokens = self.mdl.index

        samples = np.random.choice(tokens, size=M, p=probabilities, replace=True)
        
        sentences = ' '.join(samples)
        
        return sentences


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):

        total_len = len(tokens)

        frequency = pd.Series(tokens).value_counts()

        return frequency / total_len
    
    def probability(self, words):

        if not all(word in self.mdl for word in words): 

            return 0

        word_probabilities = self.mdl.loc[list(words)]

        return float(word_probabilities.prod())
        
    def sample(self, M):

        probs = self.mdl.values

        vocabulary = self.mdl.index
        
        selections = np.random.choice(vocabulary, size=M, p=probs, replace=True)

        sentences = ' '.join(selections)

        return sentences



# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        self.N = N
        ngrams = self.create_ngrams(tokens)
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)
        
        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N - 1, tokens)
    
    def create_ngrams(self, tokens):
        n = self.N
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
        
    def train(self, ngrams):
        # Count each full N-gram.
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Count each (N-1)-gram prefix.
        n_minus_one_counts = {}
        for ngram, count in ngram_counts.items():
            prefix = ngram[:-1]
            n_minus_one_counts[prefix] = n_minus_one_counts.get(prefix, 0) + count
        
        # Create rows with conditional probabilities: 
        # P(w_N | prefix) = count(ngram) / count(prefix)
        rows = []
        for ngram, count in ngram_counts.items():
            prefix = ngram[:-1]
            prob = count / n_minus_one_counts[prefix]
            rows.append((ngram, prefix, prob))
        df = pd.DataFrame(rows, columns=['ngram', 'n1gram', 'prob'])
        return df
    
    def probability(self, sentence):
        if self.N == 1:
            prob = 1.0
            for word in sentence:
                row = self.mdl[self.mdl['ngram'] == (word,)]
                if row.empty:
                    return 0
                prob *= row['prob'].iloc[0]
            return prob

        initial_history = sentence[:self.N - 1]
        prob = self.prev_mdl.probability(initial_history)
        
        for i in range(len(sentence) - self.N + 1):
            current_ngram = tuple(sentence[i:i+self.N])
            row = self.mdl[self.mdl['ngram'] == current_ngram]
            if not row.empty:
                prob *= row['prob'].iloc[0]
            else:
                if self.prev_mdl:
                    prob *= self.prev_mdl.probability(current_ngram[:-1])
                else:
                    return 0
        return prob

    def sample(self, M):
        tokens = ['\x02']  
        while (len(tokens) - 1) < M:
            context = tuple(tokens[-(self.N - 1):]) if len(tokens) >= (self.N - 1) else tuple(tokens)
            
            candidate_rows = self.mdl[self.mdl['n1gram'] == context]

            if candidate_rows.empty or ((len(tokens) - 1) == (M - 1)):
                tokens.append('\x03')
                break
            
            candidate_tokens = candidate_rows['ngram'].apply(lambda x: x[-1]).tolist()
            candidate_probs = candidate_rows['prob'].tolist()
            
            candidate_probs = np.array(candidate_probs, dtype=float)
            candidate_probs = candidate_probs / candidate_probs.sum()
            
            next_token = np.random.choice(candidate_tokens, p=candidate_probs)
            tokens.append(next_token)
        
        return " ".join(tokens)

