import tensorflow as tf
import numpy as np
# coding:utf-8
from time import sleep

from sklearn.manifold import TSNE
from tensorflow import keras
from ovs_preprocess import prepro
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from datetime import datetime
# from tensorflow_core.python.keras import layers
# from resnext import ResNeXt50
import numpy as np
import tensorflow as tf


def subtime(date1, date2):
    return date2 - date1


num_classes = 10
length = 484
number = 800  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.6, 0.2, 0.2]  # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = prepro(
    d_path=path,
    length=length,
    number=number,
    normal=normal,
    rate=rate,
    enc=False, enc_step=28)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)
# y_test = tf.squeeze(y_test)

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)
# sleep(60000)

y_train = [int(i) for i in y_train]
y_valid = [int(i) for i in y_valid]
y_test = [int(i) for i in y_test]

# 打乱顺序
index = [i for i in range(len(x_train))]
random.seed(1)
random.shuffle(index)
x_train = np.array(x_train)[index]
y_train = np.array(y_train)[index]

index1 = [i for i in range(len(x_valid))]
random.shuffle(index1)
x_valid = np.array(x_valid)[index1]
y_valid = np.array(y_valid)[index1]

index2 = [i for i in range(len(x_test))]
random.shuffle(index2)
x_test = np.array(x_test)[index2]
y_test = np.array(y_test)[index2]

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train)
print(y_valid)
print(y_test)
print("x_train的最大值和最小值：", x_train.max(), x_train.min())
print("x_test的最大值和最小值：", x_test.max(), x_test.min())

x_train = tf.reshape(x_train, (len(x_train), 484, 1))
x_valid = tf.reshape(x_valid, (len(x_valid), 484, 1))
x_test = tf.reshape(x_test, (len(x_test), 484, 1))
# print(x_train.shape)
# print(x_valid.shape)
# print(x_test.shape)
# sleep(60000)

# 保存最佳模型
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


# t-sne初始可视化函数
def start_tsne():
    print("正在进行初始输入数据的可视化...")
    x_train1 = tf.reshape(x_train, (len(x_train), 484))
    X_tsne = TSNE().fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()

# start_tsne()
# sleep(600000)

# 生成高斯噪声函数
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / 484
    npower = xpower / snr
    random.seed(1)
    noise1 = np.random.randn(484) * np.sqrt(npower)
    return x + noise1

# 绘制高斯原始图像和添加高斯噪声之后的图像
# for i in range(0,200):
#     if y_train[i] == 0:
#         print(y_train[i])
#         plt.plot(x_train[i])
#         plt.show()
#         sleep(300)
# plt.plot(x_train[600])
# plt.show()

x_valid = tf.squeeze(x_valid)
x_valid = np.array(x_valid)
for i in range(0, len(x_valid)):
    x_valid[i] = wgn(x_valid[i], 4)
x_valid = tf.reshape(x_valid, (len(x_valid), 484, 1))

x_test = tf.squeeze(x_test)
x_test = np.array(x_test)
for i in range(0, len(x_test)):
    x_test[i] = wgn(x_test[i], 4)
x_test = tf.reshape(x_test, (len(x_test), 484, 1))
#
# x_train = tf.reshape(x_train, (len(x_train), 22, 22, 1))
# x_valid = tf.reshape(x_valid, (len(x_valid), 22, 22, 1))
# x_test = tf.reshape(x_test, (len(x_test), 22, 22, 1))

# drsn 原始：96%  4db：85%, 73%
# cnn  原始： 97% 4db: 50%
# plt.plot(x_train[600])
# plt.show()
# sleep(300)

def residual_shrinkage_block(inputs, out_channels, downsample_strides=1):
    in_channels = inputs.shape[-1]

    residual = tf.keras.layers.BatchNormalization()(inputs)
    residual = tf.keras.layers.Activation('relu')(residual)
    residual = tf.keras.layers.Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
                                      padding='same')(residual)

    # residual = tf.keras.layers.BatchNormalization()(residual)
    residual = tf.keras.layers.Activation('relu')(residual)
    residual = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(residual)

    residual_abs = tf.abs(residual)
    abs_mean = tf.keras.layers.GlobalAveragePooling2D()(residual_abs)

    scales = tf.keras.layers.Dense(out_channels, activation=None)(abs_mean)
    # scales = tf.keras.layers.BatchNormalization()(scales)
    scales = tf.keras.layers.Activation('relu')(scales)
    scales = tf.keras.layers.Dense(out_channels, activation='sigmoid')(scales)

    thres = tf.keras.layers.multiply([abs_mean, scales])

    sub = tf.keras.layers.subtract([residual_abs, thres])
    zeros = tf.keras.layers.subtract([sub, sub])
    n_sub = tf.keras.layers.maximum([sub, zeros])
    residual = tf.keras.layers.multiply([tf.sign(residual), n_sub])

    out_channels = residual.shape[-1]

    if in_channels != out_channels:
        identity = tf.keras.layers.Conv2D(out_channels, 1, strides=(downsample_strides, downsample_strides),
                                          padding='same')(inputs)

    residual = tf.keras.layers.add([residual, identity])

    return residual


# # inputs = np.zeros((1, 224, 224, 3), np.float32)
# inputs = np.shape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
#
# tf.squeeze(inputs)
# # inputs = (x_train.shape[0], x_train.shape[1], x_train.shape[2])
# # tf.squeeze(inputs)
# print(inputs.shape)
# print(x_train.shape[0])
# print(x_train.shape[1])
# print(x_train.shape[2])
# sleep(3000)

def mymodel():
    # inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    # h1 = residual_shrinkage_block(inputs, 10)

    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Flatten()(h1)
    h1 = layers.Dropout(0.7)(h1)
    # h1 = layers.Dense(32, activation='relu')(h1)
    h1 = layers.Dense(10, activation='softmax')(h1)

    deep_model = keras.Model(inputs, h1, name="cnn")
    return deep_model

model = mymodel()
model.summary()
startdate = datetime.utcnow()  # 获取当前时间

# 编译模型
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
#
history = model.fit(x_train, y_train,
              batch_size=256, epochs=200, verbose=1,
              validation_data=(x_valid, y_valid))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


def acc_line():
    # 绘制acc和loss曲线
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # 画accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # 画loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    # plt.figure()
    plt.show()


acc_line()
