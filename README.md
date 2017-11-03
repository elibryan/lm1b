# lm1b: Language Model on 1 Billion Word Benchmark

Tensorflow implementation of "[Exploring the Limits of Language Modeling](http://arxiv.org/abs/1602.02410)."
The paper's pre-trained model and original implementation are available [here](https://github.com/tensorflow/models/tree/master/research/lm_1b),
courtesy of [Oriol Vinyals](vinyals@google.com), [Xin Pan](xpan@google.com) and Google.

This project is a re-implementation of theirs. It builds an identical graph from tensorflow source, instead of
generating it from a graph definition file (per the original implementation).

This was intended as a personal learning exercise (as is surely apparent), but is offered here as a reference
toward diving deeper or extending their work.

Usage examples available in `test/model_consistency_test.py`.

## To run tests...
* Follow instructions on [tensorflow/models/research/lm_1b](https://github.com/tensorflow/models/tree/master/research/lm_1b) to
 download the pre-trained model.
* Create `config.json` in the project directory (see `config.example.json` for reference). Set paths in `config.json` to
point to the original vocab file and the original checkpoint directory.
* run `pytest test/` from the project directory
