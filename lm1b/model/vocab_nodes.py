import tensorflow as tf
from tensorflow.python.ops import lookup_ops


def attach_vocab_nodes( vocab_path, hparams ):
    """
    Attach vocab nodes for looking up word or char IDs
    :param vocab_path:
    :param hparams:
    :return:
    """
    lookup_id_to_word=lookup_ops.index_to_string_table_from_file(vocab_path,default_value=hparams.tokens_unknown)
    lookup_word_to_id= lookup_ops.index_table_from_file(vocab_path, default_value=-1)
    all_chars = map( lambda i: chr( i ), range( 0, 255 ) )
    lookup_char_to_id = lookup_ops.index_table_from_tensor( tf.constant( all_chars ), default_value=hparams.chars_unknown_id )
    lookup_id_to_char = lookup_ops.index_to_string_table_from_tensor( tf.constant( all_chars ), default_value=chr( hparams.chars_unknown_id ) )
    return {"lookup_id_to_word": lookup_id_to_word,
            "lookup_word_to_id": lookup_word_to_id,
            "lookup_char_to_id": lookup_char_to_id,
            "lookup_id_to_char": lookup_id_to_char}

