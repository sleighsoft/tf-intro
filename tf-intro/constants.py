import tensorflow as tf


c_int = tf.constant(1)  # = 1
c_float = tf.constant(1.0)  # = 1.
c_bool = tf.constant(True)  # = True

c_int_2x2 = tf.constant(1, shape=(2, 2))  # = [[1,1],[1,1]]
c_float_2x2 = tf.constant(
    1, shape=(2, 2), dtype=tf.float32)  # = [[1.,1.],[1.,1.]]

add_int = c_int + c_int  # = 2
add_float = c_float + c_float  # = 2.
# c_int + c_float does not work, no implicit casting

add_int_implicit = c_int + 2  # = 3
add_float_implicit = 2 + c_float  # = 3.0
# Tensorflow will automatically convert bool, int, float to tensors

add_int_2x2 = c_int_2x2 + c_int_2x2  # = [[2,2],[2,2]]
add_float_2x2 = c_float_2x2 + c_float_2x2  # = [[2.,2.],[2.,2.]]
# Other operators are *, /, //, -, % and probably some more

broadcast_add_int = c_int + c_int_2x2  # = [[2,2],[2,2]]
broadcast_add_float = c_float + c_float_2x2  # = [[2.,2.],[2.,2.]]
# Shapes are automatically matched, if one is of scalar shape
# c_int_2x2 + c_int_3x3 would not work

# ### IGNORE EVERYTHING BELOW THIS LINE ### #
with tf.Session() as sess:
  print('c_int as object = {}'.format(c_int))
  print('c_float as object = {}'.format(c_float))
  print('c_bool as object = {}'.format(c_bool))
  print('c_int = {}'.format(sess.run(c_int)))
  print('c_float = {}'.format(sess.run(c_float)))
  print('c_bool = {}'.format(sess.run(c_bool)))
  print('c_int_2x2 = {}'.format(sess.run(c_int_2x2)))
  print('c_float_2x2 = {}'.format(sess.run(c_float_2x2)))
  print('add_int = {}'.format(sess.run(add_int)))
  print('add_float = {}'.format(sess.run(add_float)))
  print('add_int_implicit = {}'.format(sess.run(add_int_implicit)))
  print('add_float_implicit = {}'.format(sess.run(add_float_implicit)))
  print('add_int_2x2 = {}'.format(sess.run(add_int_2x2)))
  print('add_float_2x2 = {}'.format(sess.run(add_float_2x2)))
  print('broadcast_add_int = {}'.format(sess.run(broadcast_add_int)))
  print('broadcast_add_float = {}'.format(sess.run(broadcast_add_float)))
