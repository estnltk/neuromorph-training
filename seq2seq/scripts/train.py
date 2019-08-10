"""
Usage: train_lang.py [--config=CONFIG-MODULE-PATH]
"""
import os
import sys
from docopt import docopt
import seq2seq
from general_utils import load_config_from_file

args = docopt(__doc__)
print(args)

assert 'OUT_DIR' in os.environ
assert 'DATA_DIR' in os.environ
assert 'EMBEDDINGS_DIR' in os.environ

if args['--config'] is not None:
    config = load_config_from_file(args['--config'])
else:
    from seq2seq import config

if os.path.exists(os.environ['OUT_DIR']):
    print("Output directory '%s' already exists. Exiting!" % os.environ['OUT_DIR'])
    sys.exit()

import experiment_helper
experiment_helper.run_experiment(seq2seq, config, title="Seq2Seq")
