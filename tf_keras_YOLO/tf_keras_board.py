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


    # tensorboard --logdir=/full_path_to_your_logs

# in Session
# writer = tf.trian.SummaryWriter(logs_path=log_dir, graph=tf.get_default_graph())
# _, summary = sess.run([train_op, summary_op], feed_dict = feed_dict(Flase).... )
# writer.add_summary(summary, epoch*batch_count+ i ..)
# or writer.add_summary(summary_str, epoch)


