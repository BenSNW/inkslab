# -*- coding: utf-8 -*-

import tensorflow as tf


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def get_monitored_training_session(server, task_index, model_dir):
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                 device_filters=["/job:ps", "/job:worker/task:%d" % task_index])
    hooks = [tf.train.StopAtStepHook(last_step=1000000)]
    return tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(task_index == 0),
                                             checkpoint_dir=model_dir,
                                             config=sess_config,
                                             hooks=hooks)


def get_saver():
    return tf.train.Saver()


def distributed_run(args, func):
    cluster = tf.train.ClusterSpec({"ps": args.parameter_servers.split(","),
                                    "worker": args.workers.split(",")})
    server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)

    if args.job_name == 'ps':
        server.join()
    else:
        device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args.task_index,
                                                       cluster=cluster)
        with tf.device(device_setter):
            func(args, server)


def int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tfrecord(data, path):
    sentence_list, label_list = data
    writer = tf.python_io.TFRecordWriter(path)
    for sentence, label in zip(sentence_list, label_list):
        if len(sentence) != 20:
            print('!=20', sentence)
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'sentence': int_feature(sentence),
                'label': int_feature(label)
            })
        )
        writer.write(example.SerializeToString())
    writer.close()
