import tensorflow as tf

# --------------------------Tensor Naming------------------------------------ #

c_int_1 = tf.constant(1, name='c_int_1')
c_int_2 = tf.constant(2, name='c_int_2')
c_int_3 = tf.constant(3, name='c_int_3')
c_int_4 = tf.constant(4, name='c_int_4')
v_int_1 = tf.Variable(1, name='v_int_1')

add_1 = tf.add(c_int_1, c_int_2, name='add_1')  # = 3
add_2 = tf.add(add_1, c_int_3, name='add_2')  # = 6
add_3 = tf.add(add_2, c_int_4, name='add_3')  # = 10

# ---------------------------Name Scopes------------------------------------- #

with tf.name_scope('my_scope'):
  add_4 = tf.add(add_3, v_int_1, name='add_4')  # = 11
  # name = my_scope/add_4:0
  with tf.name_scope('my_inner_scope'):
    add_5 = tf.add(add_4, v_int_1, name='add_5')  # = 12
    # name = my_scope/my_inner_scope/add_5:0

# A tf.Graph is a namespace that holds the operations
# There is always a default graph, but new ones can also be created
default_graph = tf.get_default_graph()
# Print the graph
print(default_graph.as_graph_def())

# --------------------------Variable Reuse----------------------------------- #

# A usual scenario is that you develop a helper method to setup some kind of
# function e.g. cell creation, convolution, ...
# This next section therefore explains reuse of variables


def get_weight():
  import random
  initial_value = [[random.randint(0, 100), 3, 3], [3, 3, 3], [3, 3, 3]]
  # tf.get_variable retrieves an existing variable from a variable scope or
  # creates a new one.
  return tf.get_variable('weight',
                         initializer=tf.constant_initializer(initial_value),
                         shape=(3, 3))


# A variable scope encapsulates variables
with tf.variable_scope('weight_scope_1'):
  weight_1 = get_weight()
# The next weight will be different from weight_1 as we do not reuse the scope
with tf.variable_scope('weight_scope_2'):
  weight_2 = get_weight()

with tf.variable_scope('weight_scope_3') as scope:
  weight_3 = get_weight()
  # Here we explicitly set the scope to reuse
  # Therefore we get weight_3 = weight_4
  scope.reuse_variables()
  weight_4 = get_weight()

# ----------------------------Placeholder------------------------------------ #

placeholder_1 = tf.placeholder(tf.float32, shape=(2, 2))
square_placeholder_1 = tf.matmul(placeholder_1, placeholder_1)
# session.run(square_placeholder_1) would fail
# At run time a placeholder has to be filled by using
# session.run(xyz, feed_dict={placeholder_ref: value})

# ---------------------------Session Execution------------------------------- #

# A session is a concrete instantiation of a graph
with tf.Session() as sess:
  # With sess.run(tensor_ref) we can execute the graph up to 'tensor'
  # This will also change the internal state of the graph.
  # So running things like tf.assign will have a permanent effect on the graph.

  add_1_result = sess.run(add_1)
  print('add_1 = {}'.format(add_1_result))

  # We can also pass in multiple graph nodes
  add_2_result, add_3_result = sess.run([add_2, add_3])
  print('add_2 = {}, add_3 = {}'.format(add_2_result, add_3_result))

  # To run add_4, which uses a tf.Variable we have to initialize the variable
  # first.
  # This initializes all variables
  sess.run(tf.global_variables_initializer())
  # This initializes a single variable
  sess.run(v_int_1.initializer)
  # This initializes not yet initialized variables
  sess.run(tf.report_uninitialized_variables())

  # Now we can run add_4
  add_4_result = sess.run(add_4)
  print('add_4 name = {}'.format(add_4.name))
  print('add_4 = {}'.format(add_4_result))

  add_5_result = sess.run(add_5)
  print('add_5 name = {}'.format(add_5.name))
  print('add_5 = {}'.format(add_5_result))

  print('weight_1 as object = {}'.format(weight_1))
  weight_1_result = sess.run(weight_1)
  print('weight_1 = \n{}'.format(weight_1_result))

  print('weight_2 as object = {}'.format(weight_2))
  weight_2_result = sess.run(weight_2)
  print('weight_2 = \n{}'.format(weight_2_result))

  print('weight_3 as object = {}'.format(weight_3))
  weight_3_result = sess.run(weight_3)
  print('weight_3 = \n{}'.format(weight_3_result))

  print('weight_4 as object = {}'.format(weight_4))
  weight_4_result = sess.run(weight_4)
  print('weight_4 = \n{}'.format(weight_4_result))

  placeholder_value = [[2, 2], [2, 2]]
  square_placeholder_1_result = sess.run(
      square_placeholder_1,
      feed_dict={placeholder_1: placeholder_value})
  print('square_placeholder_1 = \n{}'.format(square_placeholder_1_result))
