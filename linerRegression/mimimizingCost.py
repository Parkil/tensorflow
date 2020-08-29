import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os

# 해당코드는 tensorflow v1으로 작성되었으며 어느정도 수준이 올라갈때 까지는 v1을 쓰고 나중에 v2로 변경

# [Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2] 오류 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [`loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.] 오류방지
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.8
    current_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(current_cost)

plt.plot(W_val, cost_val)
plt.show()

