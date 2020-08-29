import tensorflow.compat.v1 as tf
import numpy as np
import os

# 해당코드는 tensorflow v1으로 작성되었으며 어느정도 수준이 올라갈때 까지는 v1을 쓰고 나중에 v2로 변경

# [Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2] 오류 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [`loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.] 오류방지
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.DEBUG)  # 디버깅 로그 설정

tf.set_random_seed(777)

# file_queue를 쓰게 되면 제목라인을 거르고 가져오지 못해서 제목을 다른 type으로 인식하여 오류를 발생하는 경우가 생긴다
file_name_queue = tf.train.string_input_producer(
    [
        'c:/dev/PyCharm workspace/tensorflow/dataFile/data-01-test-score.csv',
        'c:/dev/PyCharm workspace/tensorflow/dataFile/data-02-test-score.csv',
        'c:/dev/PyCharm workspace/tensorflow/dataFile/data-03-test-score.csv'
    ],
    shuffle=True,
    name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(file_name_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost - 가설과 실제값의 차이 cost 가 적을 수록 정확한 값

# Minimize - cost의 값을 최소화 하기 위한 Optimizer 설정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Fit the line
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ", sess.run(hypothesis,
                                      feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
