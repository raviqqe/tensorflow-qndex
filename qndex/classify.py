import argtyp
import extenteten as ex
import tensorflow as tf
import qnd


def add_num_classes_flag():
    qnd.add_required_flag("num_classes", type=int)


def add_num_labels_flag():
    qnd.add_flag("num_labels", type=int)


def def_classify():
    add_num_classes_flag()
    add_num_labels_flag()
    qnd.add_flag("hidden_layer_sizes", type=argtyp.int_list, default=[100])
    qnd.add_flag("dropout_keep_prob", type=float)

    def classify(feature, label=None, *,
                 mode, key=None, regularization_scale=1e-8):
        if qnd.FLAGS.num_classes <= 1:
            raise ValueError("Number of classes must be greater than 1.")

        num_labels = qnd.FLAGS.num_labels or ex.num_labels(label)

        return ex.classify(
            ex.mlp(
                feature,
                layer_sizes=[
                    *qnd.FLAGS.hidden_layer_sizes,
                    ex.num_logits(num_labels, qnd.FLAGS.num_classes)],
                dropout_keep_prob=(
                    qnd.FLAGS.dropout_keep_prob
                    if mode == tf.contrib.learn.ModeKeys.TRAIN else
                    None)),
            label,
            num_classes=qnd.FLAGS.num_classes,
            num_labels=num_labels,
            key=key,
            regularization_scale=regularization_scale)

    return classify
