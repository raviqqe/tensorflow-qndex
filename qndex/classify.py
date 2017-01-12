import argtyp
import extenteten as ex
import qnd


def def_classify():
    qnd.add_required_flag("num_classes", type=int)
    qnd.add_flag("num_labels", type=int, help="Required only for inference")
    qnd.add_flag("hidden_layer_sizes", type=argtyp.int_list, default=[100])
    qnd.add_flag("dropout_keep_prob", type=float)

    def classify(feature, label=None, *, mode):
        if qnd.FLAGS.num_classes <= 1:
            raise ValueError("Number of classes must be greater than 1.")

        num_labels = qnd.FLAGS.num_labels or ex.num_labels(label)

        return ex.classify(
            ex.mlp(
                feature,
                layer_sizes=[
                    *qnd.FLAGS.hidden_layer_sizes,
                    ex.num_logits(num_labels, qnd.FLAGS.num_classes)],
                dropout_keep_prob=qnd.FLAGS.dropout_keep_prob),
            label,
            num_classes=qnd.FLAGS.num_classes,
            num_labels=num_labels)

    return classify