import tensorflow as tf

list_of_variables_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # = []

v_float = tf.Variable(1.0, name='v_float')
# = <tf.Variable 'v_float:0' shape=() dtype=float32_ref>

# Variables will be added to tf.GraphKeys.GLOBAL_VARIABLES collection
# automatically to keep track of them
list_of_variables_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# = [<tf.Variable 'v_float:0' shape=() dtype=float32_ref>]

# Like tf.constant, variables can be multiplied, summed, ...
add_float = v_float + v_float  # = 2.0
# = Tensor("add:0", shape=(), dtype=float32)
add_variable_and_const = v_float + tf.constant(1.0)
# = Tensor("add_1:0", shape=(), dtype=float32)

MY_COLLECTION = 'my_collection'
tf.get_collection(MY_COLLECTION)
# We can create an own collection to add variables to to help us keep track of
# them
# = <tf.Variable 'v_float:0' shape=() dtype=float32_ref>

v_float_2 = tf.Variable(
    2.0, name='v_float_my_collection',
    collections=[MY_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES])
# = <tf.Variable 'v_float_my_collection:0' shape=() dtype=float32_ref>
# Add it to tf.GraphKeys.GLOBAL_VARIABLES as well. This is needed for
# tensorflow internal computations such as backpropagation

list_of_variables_3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# = [<tf.Variable 'v_float:0' shape=() dtype=float32_ref>,
#    <tf.Variable 'v_float_my_collection:0' shape=() dtype=float32_ref>]

list_of_my_collection = tf.get_collection(MY_COLLECTION)
# = [<tf.Variable 'v_float_my_collection:0' shape=() dtype=float32_ref>]

new_v_float_2 = tf.assign(v_float_2, 3.0)
# = Tensor("Assign:0", shape=(), dtype=float32_ref)
# After the assignment is executed (explained in graphs_and_sessions.py) for
# the first time the reference to v_float_2 will hold the same value as
# new_v_float_2.

# ### IGNORE EVERYTHING BELOW THIS LINE ### #
with tf.Session() as sess:
  print('1. Collection "GLOBAL_VARIABLES" = {}'.format(list_of_variables_1))
  print('Creating v_float as object = {}'.format(v_float))
  print('2. Collection "GLOBAL_VARIABLES" = {}'.format(list_of_variables_2))
  print('Creating add_float as object = {}'.format(add_float))
  print('Creating add_variable_and_const as object = {}'.format(
      add_variable_and_const))
  print('Creating v_float_2 as object = {}'.format(v_float_2))
  print('3. Collection "GLOBAL_VARIABLES" = {}'.format(list_of_variables_3))
  print('Collection "MY_COLLECTION" = {}'.format(list_of_my_collection))
  print('Assigning new value to v_float_2 = {}'.format(new_v_float_2))
  print('! Now running tf.global_variables_initializer() to init variables')
  sess.run(tf.global_variables_initializer())
  print('v_float = {}'.format(sess.run(v_float)))
  print('add_float = {}'.format(sess.run(add_float)))
  print('add_variable_and_const = {}'.format(sess.run(add_variable_and_const)))
  print('v_float_2 = {}'.format(sess.run(v_float_2)))
  print('new_v_float_2 = {}'.format(sess.run(new_v_float_2)))
  print('v_float_2 = {}'.format(sess.run(v_float_2)))
