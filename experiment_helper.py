import os
from collections import defaultdict
from itertools import product

import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

plt.style.use('ggplot')
import tensorflow as tf
import numpy as np
import pandas as pd

from general_utils import print_config, load_config_from_file
import evaluation
import common_config

def evaluate(model, config_holder, test_file, lang_key, eval_type, out_dir):
    df = evaluation.predict(model, config_holder, test_file)
    acc_dict = evaluation.calculate_accuracy(df)
    acc_verbose = evaluation.accuracy_to_string_verbose(acc_dict)
    evaluation.save_results(df, acc_dict, acc_verbose, lang_key, eval_type, out_dir)


def _collect_results(memory, iter_result):
    memory.append(iter_result["dev_acc"])


def _update_output_dir(config, out_dir):
    config.out_dir = out_dir
    config.out_data_dir = os.path.join(out_dir, "data")
    config.dir_output = os.path.join(out_dir, "results")
    config.dir_model = os.path.join(config.dir_output, "model.weights")
    config.path_log = os.path.join(config.dir_output, "log.txt")


def train_model(model_module, config, params=None, values=None, do_evaluate=False):
    if params is not None:
        for param, value in zip(params, values):
            assert param in config.__dict__
            setattr(config, param, value)
    print_config(config)

    data_builder = model_module.DataBuilder(config)
    data_builder.run()

    config_holder = model_module.ConfigHolder(config)
    model = model_module.Model(config_holder)
    model.build()

    train = model_module.CoNLLDataset(config_holder.filename_train,
                                      config_holder.processing_word_train,
                                      config_holder.processing_tag,
                                      config_holder.processing_analysis,
                                      config_holder.max_iter,
                                      use_buckets=config.bucket_train_data,
                                      batch_size=config.batch_size,
                                      shuffle=config.shuffle_train_data,
                                      sort=config.sort_train_data,
                                      use_analysis_dropout=config.use_analysis_dropout,
                                      analysis_dropout_method=config.analysis_dropout_method
                                      )
    test = model_module.CoNLLDataset(config_holder.filename_dev,
                                     config_holder.processing_word_infer,
                                     config_holder.processing_tag,
                                     config_holder.processing_analysis,
                                     sort=True)
    train_eval = model_module.CoNLLDataset(config_holder.filename_train,
                                           config_holder.processing_word_infer,
                                           config_holder.processing_tag,
                                           config_holder.processing_analysis,
                                           sort=True,
                                           max_iter=config_holder.train_sentences_to_eval)
    model.train(train, test, train_eval)
    model.close_session()
    tf.reset_default_graph()

    # read accuracies for all iterations
    df = pd.read_csv(config_holder.training_log,
                     names=['epoch', 'acc_train', 'acc_test', 'train_loss', 'nbatches',
                            'epoch_time', 'train_time', 'eval_time'])
    acc_train_list = df['acc_train']
    acc_test_list = df['acc_test']
    train_loss_list = df['train_loss']

    # evaluate
    if do_evaluate is True:
        print("Evaluating...")
        evaluate(model, config_holder, config_holder.filename_dev, 'LANG_NA', 'dev', config_holder.out_dir)
        model.close_session()
        tf.reset_default_graph()

    return acc_train_list, acc_test_list, train_loss_list


def plot_experiment(acc_train_list, acc_test_list, train_loss_list, image_file_acc, image_file_loss, title):
    # plot test accuracy
    plt.plot(range(len(acc_test_list)), acc_test_list, '-', c="red",
             label="TEST : (best=%.4f, iter %d)" % (max(acc_test_list), np.argmax(acc_test_list)))
    # plot best test accuracy as vertical lines
    plt.axvline(x=np.argmax(acc_test_list), linestyle='--', c="red", label='best test acc')

    # plot train accuracy
    plt.plot(range(len(acc_train_list)), acc_train_list, '-', c="blue",
             label="TRAIN: (best=%.4f, iter %d)" % (max(acc_train_list), np.argmax(acc_train_list)))
    # plot best test accuracy as vertical lines
    plt.axvline(x=np.argmax(acc_train_list), linestyle='--', c="blue", label='best train acc')

    # plot train/test accuracy
    plt.title(title)
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylabel("accuracy")
    plt.legend(loc=4)
    plt.grid(color='white')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(image_file_acc, dpi=100)
    plt.clf()
    plt.cla()
    plt.close()

    # plot loss
    plt.plot(range(len(train_loss_list)), train_loss_list, '-',
             label="best=%.4f, iter %d" % (np.min(train_loss_list), np.argmin(train_loss_list)))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("train loss")
    plt.legend(loc=3)
    plt.grid(color='white')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(image_file_loss, dpi=100)
    plt.clf()
    plt.cla()
    plt.close()


def run_experiments(model_module, config, param_dict, title=None):
    value_acc_train_list, value_acc_test_list, value_train_loss_list = [], [], []
    base_out_dir = config.out_dir

    params = param_dict.keys()
    value_combinations = list(product(*param_dict.values()))

    for values in value_combinations:
        print("Running experiment with params:", params, "values:", values)
        new_out_dir = os.path.join(base_out_dir, "model_" + ",".join("%s=%s" % (p, v) for p, v in zip(params, values)))
        os.makedirs(new_out_dir)
        os.environ['OUT_DIR'] = new_out_dir
        # new_config = load_config_from_file(config.__file__)

        import importlib
        importlib.reload(common_config)
        new_config = importlib.reload(config)

        assert new_config.out_dir == new_out_dir
        acc_train_list, acc_test_list, train_loss_list = train_model(model_module, new_config, params, values)
        value_acc_train_list.append(acc_train_list)
        value_acc_test_list.append(acc_test_list)
        value_train_loss_list.append(train_loss_list)

        params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
        plot_experiment(acc_train_list, acc_test_list, train_loss_list,
                        image_file_acc=os.path.join(config.out_dir, 'experiment_accuracy_%s.png' % params_key),
                        image_file_loss=os.path.join(config.out_dir, 'experiment_train_loss_%s.png' % params_key),
                        title="Experiment accuracy %s" % params_key)

        for values, acc_list in zip(value_combinations, value_acc_test_list):
            params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
            print("Best test acc for %s: %f (%d epochs totall)" % (params_key, max(acc_list), len(acc_list)))

        for values, acc_list in zip(value_combinations, value_acc_test_list):
            params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
            print(params_key, list(acc_list))

        # plot test accuracies
        colors = ['r', 'b', 'y', 'g', 'c', 'm', 'k']
        for values, acc_list, color in zip(value_combinations, value_acc_test_list, colors):
            params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
            plt.plot(range(len(acc_list)), acc_list, '-', color=color,
                     label="TEST : %s (best=%.4f, iter %d)" % (params_key, max(acc_list), np.argmax(acc_list)))

        # plot best accuracies as vertical lines
        for values, acc_list, color in zip(value_combinations, value_acc_test_list, colors):
            params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
            plt.axvline(x=np.argmax(acc_list), linestyle='--', color=color,
                        label='%s: best test acc' % params_key)

        plt.title("%s %s" % (title if title else '', '/'.join(params)))
        plt.xlabel("Epoch")
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.ylabel("accuracy")
        plt.legend(loc=4)
        plt.grid(color='white')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(config.out_dir, 'development_accuracy.png'), dpi=100)
        plt.clf()
        plt.cla()
        plt.close()

        # plot loss curves
        for values, loss_list, color in zip(value_combinations, value_train_loss_list, colors):
            params_key = ",".join("%s=%s" % (p, v) for p, v in zip(params, values))
            best_loss, best_iter = min(loss_list), np.argmin(loss_list)
            plt.plot(range(len(loss_list)), loss_list, '-', color=color,
                     label="%s (best=%.4f, iter %d)" % (params_key, best_loss, best_iter))
        plt.title("%s %s" % (title if title else 'Loss for', ",".join(params)))
        plt.xlabel("Epoch")
        plt.ylabel("train loss")
        plt.legend(loc=3)
        plt.grid(color='white')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(config.out_dir, 'train_loss.png'), dpi=100)
        plt.clf()
        plt.cla()
        plt.close()


def run_experiment(model_module, config, title=None):
    acc_train_list, acc_test_list, train_loss_list = train_model(model_module, config)

    print("Best test acc %f (epoch %d)" % (max(acc_test_list), np.argmax(acc_test_list)))
    print("Iteration test acc:", list(acc_test_list))

    plot_experiment(acc_train_list, acc_test_list, train_loss_list,
                    image_file_acc=os.path.join(config.out_dir, 'accuracy.png'),
                    image_file_loss=os.path.join(config.out_dir, "train_loss.png"),
                    title=title if title else "Experiment accuracy")
