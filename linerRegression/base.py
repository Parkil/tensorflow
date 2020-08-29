import tensorflow.compat.v1 as tf
import os

# 해당코드는 tensorflow v1으로 작성되었으며 어느정도 수준이 올라갈때 까지는 v1을 쓰고 나중에 v2로 변경

# [Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2] 오류 방지
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# [`loss` passed to Optimizer.compute_gradients should be a function when eager execution is enabled.] 오류방지
tf.disable_v2_behavior()

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
hypothesis = x_train * W + b  # 가설 - 찾고자 하는 값

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # cost - 가설과 실제값의 차이 cost 가 적을 수록 정확한 값

# Minimize - cost의 값을 최소화 하기 위한 Optimizer 설정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)  # optimizer 를 설정
    if step % 20 == 0:
        # print(sess.run(cost))  # 주어진 수식을 계산 optimizer 가 설정되어 있을 경우 optimizer 에 따라서 수식값이 변경된다
        """
            sess.run은 들어가는 값에 따라서 수식계산/추가 option 설정 등의 다양한 역할을 수행하도록 설정되어 있는것으로 보임
            1)sess.run(train)
            print(step, 2)sess.run(cost), 3)sess.run(W), 4)sess.run(b))
            ->
            1) optimizer를 설정
            2) cost에 지정된 수식을 계산(optimizer가 설정되어 있지 않으면 계산회수와 상관없이 동일한 값이 나옴)
            3) 2)hypothesis에 지정된 W값을 random으로 설정
            4) 2)hypothesis에 지정된 b값을 random으로 설정
        """
        print(step, sess.run(cost), sess.run(W), sess.run(b))
