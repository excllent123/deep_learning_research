import tensorflow as tf


def get_summary_op(model, loss):
    for layer in model.layers:
        for weight in layer.weights:
            tf.summary.histogram(weight.name, weight)
        #if hasattr(layer, 'output'):
        #    tf.summary.histogram('{}_out'.format(layer.name), layer.output)
    tf.summary.scalar('loss', loss)

    summary_op = tf.summary.merge_all()
    return summary_op




