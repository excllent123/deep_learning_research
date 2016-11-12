import tensorflow as tf

filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("../data/vatic_id2/*.jpg"))

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue)

image = tf.image.decode_jpeg(image_file)


# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    
