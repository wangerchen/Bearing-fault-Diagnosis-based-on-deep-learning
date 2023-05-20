# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
#
# 该项目涉及数据集以及相关安装包在公众号《秋雨行舟》回复轴承即可领取。
# 对于该项目有疑问的可以在B站留言（免费答疑），或者联系微信（有偿）：LettersLive23
# 该项目对应的视频可在B站搜索《秋雨行舟》进行观看学习。
# 欢迎交流学习，共同进步

from time import sleep
from tensorflow import keras
from OriginalVibrationSignal import ovs_preprocess
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model


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

def subtime(date1, date2):
    return date2 - date1

num_classes = 10    # 样本类别
length = 784        # 样本长度
number = 300  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.5, 0.25, 0.25]  # 测试集验证集划分比例

path = r'data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = ovs_preprocess.prepro(
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

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

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
# print(y_train)
# print(y_valid)
# print(y_test)
print("x_train的最大值和最小值：", x_train.max(), x_train.min())
print("x_test的最大值和最小值：", x_test.max(), x_test.min())

x_train = tf.reshape(x_train, (len(x_train), 784, 1))
x_valid = tf.reshape(x_valid, (len(x_valid), 784, 1))
x_test = tf.reshape(x_test, (len(x_test), 784, 1))


# 模型定义
def mymodel():
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)
    h1 = layers.Dropout(0.6)(h1)
    h1 = layers.Flatten()(h1)

    h1 = layers.Dense(32, activation='relu')(h1)
    h1 = layers.Dense(10, activation='softmax')(h1)

    deep_model = keras.Model(inputs, h1, name="cnn")
    return deep_model

model = mymodel()

# 编译模型
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=256, epochs=50, verbose=1,
#                     validation_data=(x_valid, y_valid),
#                     callbacks=[CustomModelCheckpoint(
#   model, r'best_sign_cnn.h5')])
#
# # 编译模型
# model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# # 评估模型
# scores = model.evaluate(x_test, y_test, verbose=1)
# print("原始准确率")
# print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
# sleep(1000)


# 冻结模型的所有层
model.load_weights(filepath='best_sign_cnn.h5')

for layer in model.layers:
    layer.trainable = False

# 定义新的1D-CNN网络
inputs = Input(shape=(10, 1))
x = Conv1D(4, 3, activation='relu')(inputs)
x = MaxPooling1D(2)(x)
x = Conv1D(8, 3, activation='relu')(x)
x = Flatten()(x)
x = Dense(20, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 定义新的模型
model_new = Model(inputs=inputs, outputs=predictions)

transfer = model.predict(x_test)
print("transfer", transfer.shape)
print("x_test", x_test.shape)
# 训练新的模型
model_new.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_new.fit(transfer, y_test, epochs=50, batch_size=64)

# 把之前的model的输出作为新模型的输入
transfer_test = model.predict(x_test)
# 在测试数据上评估新的模型
scores = model_new.evaluate(transfer_test, y_test)
print("result")
print('%s: %.2f%%' % (model_new.metrics_names[1], scores[1] * 100))
