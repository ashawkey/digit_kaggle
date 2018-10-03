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
train_X = train_X.reshape([-1,28,28,1])
train_y = train.values[:,0]

test_X = pd.read_csv("./digit_recognizer/test.csv").values/255.0
test_X = test_X.reshape([-1,28,28,1])

n_classes = len(set(train_y))
n_train = train_X.shape[0]
n_test = test_X.shape[0]

### layers version -----
ph_x = tf.placeholder(tf.float32, [None, 28, 28, 1])
ph_y = tf.placeholder(tf.int64, [None])
y_onehot = tf.one_hot(ph_y, n_classes)
ph_dp = tf.placeholder(tf.bool)

conv0 = tf.layers.conv2d(ph_x, 32, 5, activation=tf.nn.relu)
conv1 = tf.layers.conv2d(conv0, 64, 5, activation=tf.nn.relu)
flat = tf.layers.flatten(conv1)
fc0 = tf.layers.dense(flat, 256, activation=tf.nn.relu)
dp = tf.layers.dropout(fc0, 0.5, training=ph_dp)
logits = tf.layers.dense(dp, n_classes, activation=tf.nn.softmax)

preds = tf.argmax(logits, 1)

### model training -----
cross_entropy = -tf.reduce_sum(y_onehot * tf.log(logits))
mean_loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(preds, ph_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
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
        ph_dp: True
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
        ph_dp: False
    }
    p = sess.run(preds, feed_dict=test_feed_dict)
    res[test_idx] = p

submission = pd.DataFrame({"ImageId":range(1,n_test+1), "Label":res})
submission.to_csv("layers_output.csv", index=False)
