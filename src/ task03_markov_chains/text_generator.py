class MarkovTextGenerator:
    def __init__(self, model):
        """
        Initializes the Markov Text Generator with a trained Markov model.
        """
        self.model = model

    def generate(self, seed, length=50):
        """
        Generate a sequence of text using the trained Markov model.
        :param seed: Starting sequence (list of tokens)
        :param length: Number of tokens to generate
        :return: Generated sequence
        """
        result = list(seed)
        for _ in range(length):
            context = tuple(result[-self.model.n:])
            next_token = self.model.predict(context)
            if next_token is None:
                break
            result.append(next_token)
        return result
