import tensorflow.compat.v1 as tf
import numpy as np
import os

# 해당코드는 tensorflow v1으로 작성되었으며 어느정도 수준이 올라갈때 까지는 v1을 쓰고 나중에 v2로 변경

# [Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2] 오류 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [`loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.] 오류방지
tf.disable_v2_behavior()

# X and Y data
xy = np.loadtxt('c:/dev/PyCharm workspace/tensorflow/dataFile/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost - 가설과 실제값의 차이 cost 가 적을 수록 정확한 값

# Minimize - cost의 값을 최소화 하기 위한 Optimizer 설정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
print("Your score will be ", sess.run(hypothesis,
                                      feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
