# todo: add dropout to trainer
# todo: add GPU support to trainer
# todo: reset lstm hidden state for inference
# todo: cleanup batch_sizing inconsistencies

import tensorflow as tf
import re
import os
import lm1b.model.char_embedding_nodes as char_embedding_nodes
from lm1b.utils.util import merge
from lm1b.utils.model import sharded_linear, create_sharded_weights

NUM_SHARDS=8

def _attach_cached_lstm_nodes( input, hparams=None ):
    """
    LSTM with cached / preserved hidden state
    see: https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    see: https://stackoverflow.com/questions/37969065/tensorflow-best-way-to-save-state-in-rnns

    :param input: tensor of word embeddings
    :param hparams:
    :return: lstm output and state
    """
    # LSTM with cached / preserved hidden state
    # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    cell = tf.contrib.rnn.LSTMCell( num_units=NUM_SHARDS * hparams.word_embedding_size,
                                    num_proj=hparams.word_embedding_size,
                                    num_unit_shards=NUM_SHARDS, num_proj_shards=NUM_SHARDS,
                                    forget_bias=1.0, use_peepholes=True )

    state_c = tf.get_variable( name="state_c",
                                  shape=(hparams.batch_size * hparams.sequence_length, 8192),
                                  initializer=tf.zeros_initializer,
                                  trainable=False )
    state_h = tf.get_variable( name="state_h",
                                  shape=(hparams.batch_size * hparams.sequence_length, 1024),
                                  initializer=tf.zeros_initializer,
                                  trainable=False )

    out_0, state_0 = cell( input, tf.nn.rnn_cell.LSTMStateTuple( state_c, state_h ) )

    ass_c = tf.assign( state_c, state_0[0] )
    ass_h = tf.assign( state_h, state_0[1] )

    with tf.control_dependencies( [ass_c, ass_h] ):
        out_0 = tf.identity( out_0 )

    return out_0, state_0


def _attach_projection_nodes( input, hparams=None ):
    """
    Project LSTM outputs to sparse vectors / word predictions
    :param input: lstm outputs
    :param hparams:
    :return: tensor shaped [?,vocab_size]
    """
    softmax_w = create_sharded_weights( (hparams.vocab_size / NUM_SHARDS, hparams.word_embedding_size),
                                         num_shards=NUM_SHARDS,
                                         concat_dim=1 )
    softmax_w = tf.reshape( softmax_w, shape=(-1, hparams.word_embedding_size) )
    softmax_b = tf.get_variable( 'b', shape=(hparams.vocab_size) )

    logits = tf.nn.bias_add( tf.matmul( input, softmax_w, transpose_b=True ), softmax_b, data_format="NHWC" )
    return logits


def _attach_log_perplexity_nodes( logits, targets, target_weights, hparams=None ):
    """

    :param logits:
    :param targets:
    :param target_weights:
    :param hparams:
    :return:
    """
    target_list = tf.reshape( targets, [-1] )
    target_weights_list = tf.to_float( tf.reshape( target_weights, [-1] ) )

    # hrmm
    word_count = tf.add( tf.reduce_sum( target_weights_list ), 0.0000999999974738 )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=target_list )
    cross_entropy = tf.multiply( cross_entropy, tf.to_float( target_weights ) )

    return {"log_perplexity": tf.reduce_sum( cross_entropy ) / word_count,
            "cross_entropy": cross_entropy}



CHAR_EMBEDDING_SCOPE= "char_embedding"
LSTM_SCOPE_PREFIX="lstm/lstm_"
SOFTMAX_SCOPE= "softmax"

def attach_inference_nodes( input_seqs, hparams=None ):
    """
    Predict next word for each sequence / timestep in input_seqs
    :param input_seqs: tensor of character encoded words
    :param hparams:
    :return: dict of inference nodes
    """
    with tf.variable_scope( CHAR_EMBEDDING_SCOPE ):
        word_embeddings = char_embedding_nodes.attach_char_embedding_nodes( input_seqs, num_shards=NUM_SHARDS, hparams=hparams )
        word_embeddings = tf.reshape( word_embeddings, (-1, hparams.word_embedding_size) )

    cell_out = word_embeddings
    for layer_num in range( 0, 2 ):
        with tf.variable_scope( LSTM_SCOPE_PREFIX + str( layer_num ) ):
            cell_out, cell_state = _attach_cached_lstm_nodes( cell_out, hparams=hparams )

    lstm_outputs = tf.reshape( cell_out, shape=(-1, hparams.word_embedding_size) )

    with tf.variable_scope( SOFTMAX_SCOPE ):
        logits = _attach_projection_nodes( lstm_outputs, hparams=hparams )

    return {
        "word_embeddings": word_embeddings,
        "lstm_outputs": lstm_outputs,
        "lstm_state": cell_state,
        "logits": logits
    }

def attach_predicted_word_nodes( logits, id_to_word_lookup_table, k=5, hparams=None ):
    """
    Helper to pull out the most likely words
    :param logits:
    :param id_to_word_lookup_table:
    :param k:
    :param hparams:
    :return:
    """
    top_k= tf.nn.top_k( logits, k )
    top_word_ids= top_k.indices
    word_predictions= tf.reshape( id_to_word_lookup_table.lookup( tf.to_int64( tf.reshape( top_word_ids, [-1] ) ) ), [-1, k] )
    return { "predicted_words": word_predictions,
             "top_k": top_k }



def attach_training_nodes( loss, hparams=None ):
    """
    Attach nodes for training. Work in progress...
    :param loss:
    :param hparams:
    :return:
    """
    trainable_vars = tf.trainable_variables()
    tf.get_collection( tf.GraphKeys.MODEL_VARIABLES, scope="" )
    tf.global_variables()
    all_gradients = tf.gradients( loss, trainable_vars )
    lstm_gradients = filter( lambda x: -1 < x.op.name.find( "lstm" ), all_gradients )
    non_lstm_gradients = set( all_gradients ).difference( lstm_gradients )

    lstm_gradients, global_norm = tf.clip_by_global_norm( lstm_gradients, hparams.lstm_clip_grad_norm )
    all_gradients = non_lstm_gradients.union( lstm_gradients )

    optimizer = tf.train.AdagradOptimizer( hparams.learning_rate )
    global_step = tf.Variable( 0, name='global_step', trainable=False )
    train_op = optimizer.apply_gradients( zip( all_gradients, trainable_vars ), global_step=global_step )
    return {"train_op": train_op, "global_step": global_step}


def restore_original_lm1b(sess, run_config):
    """
    Var mapping shenanigans to restore the pre-trained model to the current graph
    :param sess:
    :param run_config:
    :return:
    """
    def create_lm1b_restoration_var_map( char_embedding_vars, lstm_vars, softmax_vars):
        var_map = {}

        # Map char embedding vars
        var_map= merge(var_map, dict( map( lambda x: (x.op.name, x), char_embedding_vars ) ) )

        # Map lstm embedding vars
        var_map_regexes = {r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/projection/kernel/part_(\d).*": r"\1/W_P_\2",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/kernel/part_(\d).*": r"\1/W_\2",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/bias.*": r"\1/B",
                           r"^(" + LSTM_SCOPE_PREFIX + "\d)/lstm_cell/w_([fio])_diag.*":
                               lambda match: match.group( 1 ) + "/W_" + match.group(
                                   2 ).upper() + "_diag",
                           }
        for r_match, r_replace in var_map_regexes.items():
            matching_variables = filter( lambda x: re.match( r_match, x.name ), lstm_vars )
            for v in matching_variables:
                var_map[re.sub( r_match, r_replace, v.name )] = v

        # Map softmax embedding vars
        var_map= merge(var_map, dict( map( lambda x: (x.op.name, x), softmax_vars ) ) )

        return var_map

    var_map = create_lm1b_restoration_var_map(
        char_embedding_vars=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope=CHAR_EMBEDDING_SCOPE ),
        lstm_vars=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope=LSTM_SCOPE_PREFIX ),
        softmax_vars=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope=SOFTMAX_SCOPE )
    )

    saver = tf.train.Saver( var_list=var_map )
    saver.restore( sess, os.path.join( run_config['model_dir_path_original'], "ckpt-*" ) )
