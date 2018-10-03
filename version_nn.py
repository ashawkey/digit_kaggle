import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
import time

np.random.seed(42)

### data preprocessing -----

train = pd.read_csv("./digit_recognizer/train.csv")
train_X = train.values[:,1:]/255.0
#train_X = train_X.reshape([-1,28,28,1])
train_y = train.values[:,0]

test_X = pd.read_csv("./digit_recognizer/test.csv").values/255.0
#test_X = test_X.reshape([-1,28,28,1])

n_classes = len(set(train_y))
n_train = train_X.shape[0]
n_test = test_X.shape[0]

### nn version

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

ph_x = tf.placeholder(tf.float32, [None, 784])
x_im = tf.reshape(ph_x, [-1, 28, 28, 1])
ph_y = tf.placeholder(tf.int64, [None])
y_onehot = tf.one_hot(ph_y, n_classes)
ph_dp = tf.placeholder(tf.float32)

# conv1
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_im, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, ph_dp)

# logits
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_variable([10])
logits = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
preds = tf.argmax(logits, 1)

# model training
cross_entropy = -tf.reduce_sum(y_onehot * tf.log(logits))
mean_loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(preds, ph_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

steps = 1001
batch_size = 128

for step in range(steps):
    idx = np.random.choice(n_train, batch_size)
    batch_X = train_X[idx]
    batch_y = train_y[idx]
    train_feed_dict={
        ph_x: batch_X,
        ph_y: batch_y,
        ph_dp: 0.5
    }
    _, l, acc = sess.run([optimizer, mean_loss, accuracy], feed_dict = train_feed_dict)
    if step % 100 == 0:
        print("step %d, training accuracy %g"%(step, acc))
        

kf = KFold(40)
res = np.zeros([n_test], np.int32)
for train_idx, test_idx in kf.split(test_X):
    batch_X = test_X[test_idx]
    test_feed_dict = {
        ph_x: batch_X,
        ph_dp: 1.0
    }
    p = sess.run(preds, feed_dict=test_feed_dict)
    res[test_idx] = p

submission = pd.DataFrame({"ImageId":range(1,n_test+1), "Label":res})
submission.to_csv("nn_output.csv", index=False)
