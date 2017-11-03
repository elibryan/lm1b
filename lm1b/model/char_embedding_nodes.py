
# Transform char encoded tokens to something like word embeddings

import tensorflow as tf
from lm1b.utils.model import sharded_linear, create_sharded_weights


def _create_cnn_cell( inputs, kernel_size, kernel_feature_size, name="cnn", hparams=None ):
    """
    conv neural network cell helper
    :param inputs: rank 4 tensor of character embeddings
    :param kernel_size:
    :param kernel_feature_size:
    :param name:
    :param hparams:
    :return:
    """
    with tf.variable_scope( name ):
        kernel = tf.get_variable( "kernel",
                                  shape=(kernel_size, hparams.char_embedding_size, 1, kernel_feature_size),
                                  dtype=tf.float32 )
        b = tf.get_variable( 'b', [1, 1, 1, kernel_feature_size] )
        conv = tf.nn.conv2d( input=inputs, filter=kernel, strides=[1, 1, 1, 1], padding="VALID" ) + b
        out = tf.nn.relu( conv )
        out = tf.reduce_max( out, reduction_indices=[1], keep_dims=True )
        out = tf.reshape( out, shape=[-1, kernel_feature_size] )
        return out


def _highway_layer( input, num_shards= 8, name="dense"):
    # todo: see if tf's built in highway layer can replace
    with tf.variable_scope(name):
        with tf.variable_scope( "gate" ):
            t_gate = tf.nn.sigmoid( sharded_linear( input, (input.shape[1] / num_shards, input.shape[1]), num_shards=num_shards ) - 2.0 )

        with tf.variable_scope( "tr" ):
            tr_gate = tf.nn.relu( sharded_linear( input, (input.shape[1] / num_shards, input.shape[1]), num_shards=num_shards ) )

        return tf.multiply( t_gate, tr_gate ) + tf.multiply( (1 - t_gate), input )


def attach_char_embedding_nodes( char_inputs, num_shards, hparams=None ):
    """
    Char CNN, encode character representations to ~ word embeddings.
      Based on: https://arxiv.org/abs/1508.06615
      (see also: https://github.com/mkroutikov/tf-lstm-char-cnn)

    :param char_inputs: [?,max_word_length] tensor of token character arrays
    :param hparams:
    :return: [hparams.batch_size, hparams.sequence_length, hparams.word_embedding_size] tensor
    """
    char_embeddings_lookup = tf.get_variable( "W", shape=(hparams.char_vocab_size, hparams.char_embedding_size),
                                              dtype=tf.float32,
                                              initializer=tf.random_uniform_initializer )

    char_embeddings = tf.nn.embedding_lookup( char_embeddings_lookup, char_inputs )
    char_embeddings_reshaped = tf.reshape( char_embeddings,
                                           shape=[-1, hparams.max_word_length, hparams.char_embedding_size, 1] )

    # Parameters found in original lm1b model
    kernels = [1, 2, 3, 4, 5, 6, 7, 7]
    kernel_features = [32, 32, 64, 128, 256, 512, 1024, 2048]

    cnn_layers = []
    for i, (cur_kernel_size, cur_kernel_feature_size) in enumerate( zip( kernels, kernel_features ) ):
        cnn_cell = _create_cnn_cell( char_embeddings_reshaped,
                                     kernel_size=cur_kernel_size,
                                     kernel_feature_size=cur_kernel_feature_size,
                                     name="cnn_" + str( i ),
                                     hparams=hparams)
        cnn_layers.append( cnn_cell )

    cnn_outputs = tf.concat( cnn_layers, 1 )
    cnn_outputs = tf.reshape( cnn_outputs, (hparams.batch_size * hparams.sequence_length, -1) )

    highway_0 = _highway_layer( cnn_outputs, num_shards=num_shards, name="dense_0" )
    highway_1 = _highway_layer( highway_0, num_shards=num_shards, name="dense_1" )

    with tf.variable_scope( "proj" ):
        proj = sharded_linear( highway_1, ((highway_1.shape[1].value / num_shards), hparams.word_embedding_size), num_shards=num_shards )

    word_embeddings = tf.reshape( proj, (hparams.batch_size, hparams.sequence_length, hparams.word_embedding_size) )

    return word_embeddings

