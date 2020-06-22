from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12



#입력 이미지의 사이즈를 설정합니다.
img_rows, img_cols = 28, 28




#train 데이터와 test 데이터의 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)




"""
DNN과이 차이입니다.
1. 2차원 이미지를 1차원 벡터로 변환하지 않고 그대로 사용한다.
2. 흑백 이미지에 대한 정보 처리를 위해서 추가적인 차원을 포함시켜야 한다.
"""


# 이미지 배열의 앞 단에 추가해야하는 경우(channel_first) 반대(channel_last)
if K.image_data_format() == 'channels_first':
    
 # 이미지의 샘플수, 채널 수, 이미지의 가로 길이, 이미지의 세로 길이
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    #reshape로 행렬크기 변경
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



#전처리
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 클래스 벡터를 이진 클래스 행렬로 변환
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



#CNN을 이용, 모델 이용
# Convolution레이어를 추가
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#print(model.output_shape)

# Convolution레이어 추가


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#print(model.output_shape)



#Keras에서는 모형을 컴파일하는 과정이 필요한데,
#이 과정에서 loss function이나 optimizer 등을 설정해주게 됩니다.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])




#모델 학습
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,    
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



#학습이 잘 됐는지 평가하는 과정.
model.save('mnist_model.h5')
model.save_weights('mnist_model_weights.h5')
