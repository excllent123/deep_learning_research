import tensorflow as tf


def get_summary_op(model, loss):
    for layer in model.layers:
        for weight in layer.weights:
            tf.histogram_summary(weight.name, weight)
        if hasattr(layer, 'output'):
            tf.histogram_summary('{}_out'.format(layer.name), layer.output)
    tf.scalar_summary('loss', loss)

    summary_op = tf.merge_all_summaries()
    return summary_op


    # tensorboard --logdir=/full_path_to_your_logs

# in Session
# writer = tf.trian.SummaryWriter(logs_path=log_dir, graph=tf.get_default_graph())
# _, summary = sess.run([train_op, summary_op], feed_dict .... )
# writer.add_summary(summary, epoch*batch_count+ i ..)
# or writer.add_summary(summary_str, epoch)


