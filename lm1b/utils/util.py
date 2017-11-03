# Random helpers

import json


def slurp( file_path ):
    with open( file_path, 'r' ) as fin:
        source_lines = map( str.strip, fin.readlines() )
        for s in source_lines:
            yield s


def load_config( file_path ):
    return json.loads( "\n".join( slurp( file_path ) ) )


def merge( *dictionaries ):
    """
    Helper toward immutable dictionary 'updates' (ala assoc or merge-ing a map in clojure)?
    :param dictionaries: one or more dictionaries to merge, left to right
    :return: single dict
    """
    output = {}
    for d in dictionaries:
        output.update( d )
    return output
