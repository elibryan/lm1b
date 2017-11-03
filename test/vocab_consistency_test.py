
# Test vocab and word to character encoding
# Make sure word IDs match
# Make sure character IDs match

import pytest
import numpy as np
import tensorflow as tf
import lm1b.utils.vocab as vocab_util
import lm1b.model.vocab_nodes as vocab_nodes
import lm1b.model.model_nodes as model_nodes

import lm1b.utils.util as util
from lm1b.utils.util import merge
import lm1b.hparams

run_config= util.load_config("config.json")

hparams= lm1b.hparams.get_default_hparams()
hparams._set_from_map({'sequence_length': 20,
                       'max_word_length': 50,
                       'chars_padding_id': 4})


test_words= ["hello", "Hello", "pineapple", "donut", "I", "love", ".", "<S>", "</S>"]
expected_word_ids= [20887, 16288, 36053, 60263, 38, 978, 5, 1, 0]
expected_char_arrays= [
    [2, 104, 101, 108, 108, 111, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 72, 101, 108, 108, 111, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 112, 105, 110, 101, 97, 112, 112, 108, 101, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 100, 111, 110, 117, 116, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 73, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 108, 111, 118, 101, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 46, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    [2, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
]
padding_token_char_array= [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

graph= tf.Graph()
with graph.as_default():
    graph_nodes={}
    graph_nodes=merge( graph_nodes, vocab_nodes.attach_vocab_nodes(run_config['vocab_path'], hparams=hparams) )
    char_to_id_lookup_table = graph_nodes['lookup_char_to_id']
    word_to_id_lookup_table = graph_nodes['lookup_word_to_id']

    sess= tf.Session(graph=graph)
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.tables_initializer() )


def test_token_encoder():
    with graph.as_default():
        char_arrays=sess.run(vocab_util.encode_token(tf.constant(test_words), char_to_id_lookup_table=char_to_id_lookup_table, hparams=hparams))
        assert( np.array_equal( expected_char_arrays, char_arrays) )
        assert( np.array_equal( expected_word_ids, sess.run(word_to_id_lookup_table.lookup(tf.constant(test_words)))))


def test_seq_encoder():
    with graph.as_default():
        test_seqs = ["Hello pineapple .", "<S> Hello pineapple . </S>"]
        expected_seqs = []
        for test_seq in test_seqs:
            encoded_tokens = map( lambda x: expected_char_arrays[test_words.index( x )], test_seq.split() )
            padded_encoded_tokens = (encoded_tokens + ([padding_token_char_array] * hparams.sequence_length))[
                                    :hparams.sequence_length]
            expected_seqs.append( padded_encoded_tokens )

        actual_seqs= sess.run(vocab_util.encode_sequence(tf.constant(test_seqs), char_to_id_lookup_table=char_to_id_lookup_table, hparams=hparams))
        assert( np.array_equal( expected_seqs, actual_seqs))

