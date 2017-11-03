import tensorflow as tf


def create_sharded_weights( shape, num_shards, name="W", concat_dim=0 ):
    """
    todo: see if tf's built in variable sharding can replace
    :param shape:
    :param num_shards:
    :param name:
    :param concat_dim:
    :return:
    """

    weights = []
    for i in range( 0, num_shards ):
        cur_w = tf.get_variable( name + "_" + str( i ), shape=shape,
                                 initializer=tf.random_normal_initializer,
                                 dtype=tf.float32 )
        weights.append( cur_w )
    w = tf.concat( weights, concat_dim )
    return w


def sharded_linear( input, shape, num_shards ):
    """
    todo: see if tf's built in variable sharding can replace
    :param input:
    :param shape:
    :param num_shards:
    :return:
    """
    w = create_sharded_weights( shape, num_shards=num_shards )
    b = tf.get_variable( "b", shape=(1, shape[1],), dtype=tf.float32 )
    return tf.matmul( input, w ) + b
