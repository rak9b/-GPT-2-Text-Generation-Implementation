import random
from collections import defaultdict


class MarkovModel:
    def __init__(self, n=2):
        """
        Initializes a Markov model of order `n`.
        """
        self.n = n
        self.transitions = defaultdict(list)

    def train(self, corpus):
        """
        Train the Markov model using the given corpus.
        :param corpus: List of tokens (words, characters, etc.)
        """
        for i in range(len(corpus) - self.n):
            key = tuple(corpus[i:i + self.n])
            next_token = corpus[i + self.n]
            self.transitions[key].append(next_token)

    def predict(self, context):
        """
        Predict the next token based on the given context.
        :param context: Tuple of `n` tokens
        :return: Next token
        """
        if context in self.transitions:
            return random.choice(self.transitions[context])
        return None
