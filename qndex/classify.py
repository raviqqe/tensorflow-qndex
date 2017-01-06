import argtyp
import extenteten as ex
import qnd


def def_classify():
    qnd.add_required_flag("num_classes", type=int)
    qnd.add_flag("hidden_layer_sizes", type=argtyp.int_list, default=[100])
    qnd.add_flag("dropout_keep_prob", type=float)

    def classify(feature, label):
        return ex.classify(
            ex.mlp(
                feature,
                layer_sizes=[
                    *qnd.FLAGS.hidden_layer_sizes,
                    ex.num_logits(ex.num_labels(label), qnd.FLAGS.num_classes)],
                dropout_keep_prob=qnd.FLAGS.dropout_keep_prob),
            label,
            binary=(qnd.FLAGS.num_classes == 2))

    return classify
