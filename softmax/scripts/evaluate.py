"""
Usage: evaluate.py [--config=CONFIG-MODULE-PATH]
                   [--output-dir=OUTPUT-DIR]
                   (--dev | --test | --file=TEST_FILE)

Options:
  --dev                      evaluate on development set
  --test                     evaluate on test set
  --output-dir=OUTPUT-DIR    model directory
  --file=TEST_FILE           evaluation file


Creates output files:
    $OUT_DIR/$LANGUAGE-KEY/evaluation.acc:
        Containing on row with values: lang_key, acc_morph_all, acc_morph_oov, acc_morph_voc, acc_pos_all, acc_pos_oov, acc_pos_voc

    $OUT_DIR/$LANGUAGE-KEY/predictions.csv:
        Contains all prediction for evaluated dataset.

"""
import os

from docopt import docopt

from softmax import ConfigHolder, Model
from general_utils import load_config_from_file
import evaluation

if __name__ == "__main__":
    args = docopt(__doc__)

    assert "DATA_DIR" in os.environ

    if args['--output-dir'] is None:
        assert "OUT_DIR" in os.environ
    else:
        os.environ["OUT_DIR"] = args['--output-dir']

    if args['--config'] is not None:
        config = load_config_from_file(args['--config'])
    else:
        from softmax import config

    print("Using configuration", config.__file__)


    if args['--file'] is not None:
        test_file = args['--file']
        eval_type = 'file'
    elif args['--test'] is True:
        test_file = config.filename_test
        eval_type = 'test'
    elif args['--dev'] is True:
        test_file = config.filename_dev
        eval_type = 'dev'
    else:
        raise ValueError('Specify --dev or --test.')

    config_holder = ConfigHolder(config)    
    model = Model(config_holder)
 
    df = evaluation.predict(model, config_holder, test_file)
    acc_dict = evaluation.calculate_accuracy(df)
    acc_verbose = evaluation.accuracy_to_string_verbose(acc_dict)
    evaluation.save_results(df, acc_dict, acc_verbose, "lang", eval_type, config.out_dir)
