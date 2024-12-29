def test_markov_model():
    from task03_markov_chains.markov_model import MarkovModel

    model = MarkovModel(n=2)
    corpus = "the quick brown fox jumps over the lazy dog".split()
    model.train(corpus)

    context = ("the", "quick")
    assert model.predict(context) is not None
