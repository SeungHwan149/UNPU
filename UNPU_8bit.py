# Tensorflow 임포트
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import numpy as np
import math
from time import time

from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot as plt

# MNIST 데이터셋을 2차원으로 다운로드 하고 준비
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[..., None]
test_images = test_images[..., None]

# 이미지를 [0, 1] 범위로 변경하기.
train_images = train_images / np.float32(255)
test_images = np.int32(test_images / np.float32(1))

tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 1

EPOCHS = 10


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE * 8) 
test_dataset_v2 = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE * 16) 
    
#서브 클래싱 모델
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape = (28, 28, 1), padding = 'same', dtype = tf.float32)
        self.maxpooling1 = MaxPooling2D((2,2))
        self.conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', dtype = tf.float32)
        self.maxpooling2 = MaxPooling2D((2,2))
        self.conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', dtype = tf.float32)
        self.flatten = Flatten()
        self.d1 = Dense(64, activation = 'relu', dtype = tf.float32)
        self.d2 = Dense(10, activation='softmax')
        self.conv1_res = CustomConv2(32, 3, activation = 'relu', padding = 'same')
        self.conv2_res = CustomConv2(64, 3, activation = 'relu', padding = 'same')
        self.conv3_res = CustomConv2(64, 3, activation = 'relu', padding = 'same')
        self.d1_res = CustomDense(64, activation = 'relu')
        self.d2_res = CustomDense(10, activation = 'softmax')
        self.conv1_res_v2 = NewCustomConv2(32, 3, activation = 'relu', padding = 'same')
        self.conv2_res_v2 = NewCustomConv2(64, 3, activation = 'relu', padding = 'same')
        self.conv3_res_v2 = NewCustomConv2(64, 3, activation = 'relu', padding = 'same')
        self.d1_res_v2 = NewCustomDense(64, activation = 'relu')
        self.d2_res_v2 = NewCustomDense(10, activation = 'softmax')
    
    def call(self, x, training = None, v2 = None):
        if training:
            x = self.conv1(x)
            x = self.maxpooling1(x)
            x = self.conv2(x)
            x = self.maxpooling2(x)
            x = self.conv3(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)
        if v2:
            x = self.conv1_res_v2(x, 8)
            x = self.maxpooling1(x)
            x = self.conv2_res_v2(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            x = self.maxpooling2(x)
            x = self.conv3_res_v2(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            x = self.flatten(x)
            x = self.d1_res_v2(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            return self.d2_res_v2(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            
        else:
            x = self.conv1_res(x, 8)
            x = self.maxpooling1(x)
            x = self.conv2_res(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            x = self.maxpooling2(x)
            x = self.conv3_res(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            x = self.flatten(x)
            x = self.d1_res(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
            return self.d2_res(tf.cast(x / tf.reduce_max(x) * 255, tf.int32), 8)
    def set(self):
        self.conv1_res.kernel = tf.cast(self.conv1.kernel * 127, tf.int32)
        self.conv1_res.bias = tf.cast(self.conv1.bias * 127, tf.int32)
        self.conv1_res_v2.kernel = tf.cast(self.conv1.kernel * 127, tf.int32)
        self.conv1_res_v2.bias = tf.cast(self.conv1.bias * 127, tf.int32)
        self.conv2_res.kernel = tf.cast(self.conv2.kernel * 127, tf.int32)
        self.conv2_res.bias = tf.cast(self.conv2.bias * 127, tf.int32)
        self.conv2_res_v2.kernel = tf.cast(self.conv2.kernel * 127, tf.int32)
        self.conv2_res_v2.bias = tf.cast(self.conv2.bias * 127, tf.int32)
        self.conv3_res.kernel = tf.cast(self.conv3.kernel * 127, tf.int32)
        self.conv3_res.bias = tf.cast(self.conv3.bias * 127, tf.int32)
        self.conv3_res_v2.kernel = tf.cast(self.conv3.kernel * 127, tf.int32)
        self.conv3_res_v2.bias = tf.cast(self.conv3.bias * 127, tf.int32)
        self.d1_res.kernel = tf.cast(self.d1.kernel * 127, tf.int32)
        self.d1_res.bias = tf.cast(self.d1.bias * 127, tf.int32)
        self.d1_res_v2.kernel = tf.cast(self.d1.kernel * 127, tf.int32)
        self.d1_res_v2.bias = tf.cast(self.d1.bias * 127, tf.int32)
        self.d2_res.kernel = tf.cast(self.d2.kernel * 127, tf.int32)
        self.d2_res.bias = tf.cast(self.d2.bias * 127, tf.int32)
        self.d2_res_v2.kernel = tf.cast(self.d2.kernel * 127, tf.int32)
        self.d2_res_v2.bias = tf.cast(self.d2.bias * 127, tf.int32)
#UNPU를 구현하기 위한 layer

class CustomConv2(Layer):
    def __init__(self,
                 filters, 
                 kernel_size,
                 activation,
                 padding):
        super(CustomConv2, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        if padding == 'same':
            self.padding = True
        else :
            self.padding = False
    def call(self, inputs, bits):
        input_size = inputs.shape  
        i = 1
        t = 0
        if i < 2:
            print(inputs.shape)
            print(self.kernel.shape)
            i = i + 1
        if self.padding:
            padding_size = 1
            padding = tf.constant([[0, 0],
                                   [padding_size, padding_size],
                                   [padding_size, padding_size], 
                                   [0, 0]], 
                                   dtype = tf.int32)
            inputs = tf.pad(inputs, padding, "CONSTANT")
        for row in range(input_size[1]):
            for col in range(input_size[2]):
                split0, split_col, split2 = tf.split(inputs, [col, 3, inputs.shape[2] - col - 3], 2)
                for i in range(3):
                    split0, split_row, split2 = tf.split(split_col, [row + i, 1, inputs.shape[1] - row - i - 1], 1)
                    input = tf.squeeze(split_row, [1])
                    w = tf.Variable(self.kernel[i])
                    input = tf.expand_dims(input, 3)
                    w_sign = tf.sign(w)
                    if t < 1:
                        print("input shape : ", end="")
                        print(input.shape)
                        print("weight shape : ", end="")
                        print(w.shape)
                    for bit in range(bits):
                        if bit == 0:
                            dim0 = input * (w % 2 * w_sign)
                            if t < 1 :
                                print("dim0 shape : ", end="")
                                print(dim0.shape)
                                t = t + 1
                        else:
                            dim0 = dim0 + input * (w % 2 * w_sign)
                        w = tf.cast(w / 2, tf.int32)
                        dim0 = tf.cast(dim0 / 2, tf.int32)
                    if i == 0:
                        dim1 = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(dim0, 1), 1), 1)
                    else:
                        dim1 = dim1 + tf.expand_dims(tf.reduce_sum(tf.reduce_sum(dim0, 1), 1), 1)
                if col == 0:
                    dim2 = tf.expand_dims(dim1, 2)
                else:
                    dim2 = tf.concat([dim2, tf.expand_dims(dim1, 2)], 2)
            if row == 0:
                dim3 = tf.Variable(dim2)
            else:
                dim3 = tf.concat([dim3, dim2], 1)
        if self.activation == 'relu':
            return tf.nn.relu(dim3 + self.bias)
        else:
            return dim3 + self.bias
        
class CustomDense(Layer):
    def __init__(self,
                  units,
                  activation):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = activation
    def call(self, inputs, bits):
        
        for i in range(int((inputs.shape[1] + 3) / 3)):
            if i * 3 + 2 < inputs.shape[1]:
                split0, input, split1 = tf.split(inputs, [i * 3, 3, inputs.shape[1] - 3 - i * 3], 1)
                input = tf.expand_dims(input, 2)
                w = tf.expand_dims(self.kernel[i * 3:i * 3 + 3], 0)
            elif i * 3 + 2 == inputs.shape[1]:
                split0, input, split1 = tf.split(inputs, [i * 3, 2, inputs.shape[1] - 2 - i * 3], 1)
                input = tf.expand_dims(input, 2)
                w = tf.expand_dims(self.kernel[i * 3:i * 3 + 2], 0)
            elif i * 3 + 1 == inputs.shape[1]:
                split0, input, split1 = tf.split(inputs, [i * 3, 1, inputs.shape[1] - 1 - i * 3], 1)
                input = tf.expand_dims(input, 2)
                w = tf.expand_dims(tf.expand_dims(self.kernel[i * 3], 0), 0)
            w_sign = tf.sign(w)
            for bit in range(bits): 
                if bit == 0:
                    dim1 = input * (w % 2 * w_sign)
                else:
                    dim1 = dim1 + input * (w % 2 * w_sign)
                w = tf.cast(w / 2, tf.int32)
                dim1 = tf.cast(dim1 / 2, tf.int32)
            if i == 0:
                res = tf.reduce_sum(dim1, 1)
            else:
                res = res + tf.reduce_sum(dim1, 1)
        if self.activation == 'relu':
            return tf.nn.relu(res + self.bias) 
        elif self.activation == 'softmax':
            return tf.nn.softmax((res + self.bias) / tf.reduce_max(res + self.bias))
        
class NewCustomConv2(Layer):
    def __init__(self,
                 filters, 
                 kernel_size,
                 activation,
                 padding):
        super(NewCustomConv2, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        if padding == 'same':
            self.padding = True
        else :
            self.padding = False
    def call(self, inputs, bits):
        input_size = inputs.shape
        if self.padding:
            padding_size = 1
            padding = tf.constant([[0, 0],
                                   [padding_size, padding_size],
                                   [padding_size, padding_size], 
                                   [0, 0]], 
                                   dtype = tf.int32)
            inputs = tf.pad(inputs, padding, "CONSTANT")
        for row in range(input_size[1]):
            for col in range(input_size[2]):
                split0, split_col, split2 = tf.split(inputs, [col, 3, inputs.shape[2] - col - 3], 2)
                for i in range(3):
                    split0, split_row, split2 = tf.split(split_col, [row + i, 1, inputs.shape[1] - row - i - 1], 1)
                    for j in range(3):
                        split0, input, split2 = tf.split(split_row, [j, 1, 2 - j], 2)                    
                        input = tf.squeeze(input, [1])
                        input = tf.expand_dims(input, 3)
                        w = tf.Variable(self.kernel[i][j])
                        w_sign = tf.sign(w)
                        w = tf.abs(w)
                        for bit in range(int(bits / 2)):
                            if bit == 0:
                                dim0 = input * (w % 4) * w_sign
                            else:
                                dim0 = dim0 + input * (w % 4) * w_sign
                            w = tf.cast(w / 4, tf.int32)
                            dim0 = tf.cast(dim0 / 4, tf.int32)
                        if i == 0 and j == 0:
                            dim1 = tf.reduce_sum(dim0, 2)
                        else:
                            dim1 = dim1 + tf.reduce_sum(tf.cast(dim0, tf.int32), 2)
                if col == 0:
                    dim2 = tf.expand_dims(dim1, 2)
                else:
                    dim2 = tf.concat([dim2, tf.expand_dims(dim1, 2)], 2)
            if row == 0:
                dim3 = tf.Variable(dim2)
            else:
                dim3 = tf.concat([dim3, dim2], 1)
        if self.activation == 'relu':
            return tf.nn.relu(dim3 + self.bias)
        else:
            return dim3 + self.bias

class NewCustomDense(Layer):
    def __init__(self,
                  units,
                  activation):
        super(NewCustomDense, self).__init__()
        self.units = units
        self.activation = activation
    def call(self, inputs, bits):
        res = tf.zeros(shape=(inputs.shape[0], self.kernel.shape[1]), dtype=tf.int32)
        for i in range(inputs.shape[1]):
            split0, input, split1 = tf.split(inputs, [i, 1, inputs.shape[1] - 1 - i], 1)
            input = tf.expand_dims(input, 2)
            w = tf.expand_dims(tf.expand_dims(self.kernel[i], 0), 0)
            w_sign = tf.sign(w)
            w = tf.abs(w)
            for bit in range(int(bits / 2)):
                if bit == 0:
                    dim1 = input * (w % 4 * w_sign)
                else:
                    dim1 = dim1 + input * (w % 4 * w_sign)
                w = tf.cast(w / 4, tf.int32)
                dim1 = tf.cast(dim1 / 4, tf.int32)
            res = res + tf.reduce_sum(dim1, 1)
        if self.activation == 'relu':
            return tf.nn.relu(res + self.bias) 
        elif self.activation == 'softmax':
            return tf.nn.softmax((res + self.bias) / tf.reduce_max(res + self.bias))

# 훈련에 필요한 옵티마이저와 손실함수 선택
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

#모델의 손실과 성능 측정할 지표 선택. 수집된 정보 바탕으로 최종결과 출력
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')  
test_loss_v2 = tf.keras.metrics.Mean(name = 'test_loss_v2')
test_accuracy_v2 = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_v2')  

model = CustomModel()
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape를 이용하여 모델 훈련
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
    
def test_step(images, labels, v2 = None):
    if v2:
        predictions = model(images, training = False, v2 = True)
        t_loss_v2 = loss_object(labels, predictions)
        test_loss_v2(t_loss_v2)
        test_accuracy_v2(labels, predictions)
    else:
        predictions = model(images, training = False, v2 = False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
EPOCHS = 10
print("시뮬레이션 시작")
acc_arr = []
test_acc_arr = []
test_acc_arr_v2 = []
loss_arr = []
test_loss_arr = []
test_loss_arr_v2 = []
time_total_arr = []
time_total_arr_v2 = []
for epoch in range(EPOCHS):
    for images, labels in train_dataset:
        train_step(images, labels)
    template = '에포크: {}, 손실: {}, 정확도: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))
    acc_arr.append(train_accuracy.result())
    loss_arr.append(train_loss.result())
    model.set()
    print("테스트 시작")
    time_arr = []
    for test_images, test_labels in test_dataset:
        start = time()
        test_step(test_images, test_labels, v2 = False)
        end = time()
        time_arr.append((end - start) / 60)    
    template = '기존 - 테스트 손실: {}, 테스트 정확도:{}, 테스트 소요 시간: {}분'
    print(template.format(test_loss.result(),
                          test_accuracy.result()*100,
                          sum(time_arr)))
    time_total_arr.append(sum(time_arr))
    test_acc_arr.append(test_accuracy.result())
    test_loss_arr.append(test_loss.result())
    time_arr = []
    for test_images, test_labels in test_dataset_v2:
        start = time()
        test_step(test_images, test_labels, v2 = True)
        end = time()
        time_arr.append((end - start) / 60)    
    template = '개선 - 테스트 손실: {}, 테스트 정확도:{}, 테스트 소요 시간: {}분\n'
    print(template.format(test_loss_v2.result(),
                          test_accuracy_v2.result()*100,
                          sum(time_arr)))
    time_total_arr_v2.append(sum(time_arr))
    test_acc_arr_v2.append(test_accuracy_v2.result())
    test_loss_arr_v2.append(test_loss_v2.result())
template = '기존 방식 소요시간 : {}, 개선 방식 소요시간 : {}'
print(template.format(sum(time_total_arr) / len(time_total_arr),
                      sum(time_total_arr_v2) / len(time_total_arr_v2)))
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_arr, label='Training')
plt.plot(epochs_range, test_acc_arr, label='Test(original)')
plt.plot(epochs_range, test_acc_arr_v2, label='Test(new)')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_arr, label='Training')
plt.plot(epochs_range, test_loss_arr, label='Test(original)')
plt.plot(epochs_range, test_loss_arr_v2, label='Test(new)')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()
