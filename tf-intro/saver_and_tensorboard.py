import tensorflow as tf

# ---------------------------------Setup------------------------------------- #

v1 = tf.Variable([1, 2, 3, 4], name='v1')
v2 = tf.Variable([0, 0, 0, 0], name='v2')

# ---------------------------Some Operation---------------------------------- #

assign = tf.assign(v1, v2, name='assign_v2_to_v1')

# ---------------------------------Saver------------------------------------- #

# Allows saving and loading of model data (variables)
saver = tf.train.Saver(name='saver')
# Alternatively save only certain variables with
saver_selective = tf.train.Saver({'v1': v1}, name='saver_selective')

# ---------------------------------Summaries--------------------------------- #

# This adds a summary of v1 to the tf.GraphKeys.SUMMARIES collection
tf.summary.tensor_summary('v1_summary', v1)
# Merge all summaries in tf.GraphKeys.SUMMARIES
merged_summaries = tf.summary.merge_all()

with tf.Session() as sess:
  # -------------------------------File Writer------------------------------- #

  # Allows saving event data (loss, variable states, etc.) and the graph to be
  # rendered in tensorboard
  event_writer = tf.summary.FileWriter('./tmp/', sess.graph)

  def create_summary(writer, step, merged_summaries):
    run_metadata = tf.RunMetadata()
    writer.add_run_metadata(run_metadata, 'last_run_tag_{}'.format(step))
    writer.add_summary(merged_summaries, step)

  # ---------------------------------Demo------------------------------------ #

  step = 0

  print('INITIALIZE')
  sess.run(tf.global_variables_initializer())
  print('v1 = {}'.format(sess.run(v1)))
  step += 1
  summaries = sess.run(merged_summaries)
  create_summary(event_writer, step, summaries)

  print('ASSIGN')
  sess.run(assign)
  print('v1 = {}'.format(sess.run(v1)))
  step += 1
  summaries = sess.run(merged_summaries)
  create_summary(event_writer, step, summaries)

  save_path = saver.save(sess, './tmp/model.ckpt')
  print('Saved model to {}'.format(save_path))

  print('RUN INITIALIZE TO RESET')
  sess.run(tf.global_variables_initializer())
  print('v1 = {}'.format(sess.run(v1)))
  step += 1
  summaries = sess.run(merged_summaries)
  create_summary(event_writer, step, summaries)

  saver.restore(sess, save_path)
  print('Restored model from {}'.format(save_path))
  print('v1 = {}'.format(sess.run(v1)))
  step += 1
  summaries = sess.run(merged_summaries)
  create_summary(event_writer, step, summaries)

  event_writer.close()
