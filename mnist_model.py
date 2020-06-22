from keras.models import load_model, Sequential
class MNIST_MODEL(object):
    
    def __init__(self):
        self.input_shape = (28,28,1)
        self.num_classes = 10
        self.model = load_model('mnist_model.h5')


    def network(self, weights=None):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
        return model.load_weights()

'''
케라스에서 제공되는 convolution 레이어 종류에도 여러가지가 있으나 영상처리에 주로 사용되는 conv2D 레이어를 사용했습니다.
레이어는 주로 영상 인식에 사용되고 필터가 탑재되어 있습니다.

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))

conv2D() 함수의 인자로 첫번째 숫자인 32는 32개의 필터를 적용하겠다는 의미입니다.
kernel_size는 필터의 크기입니다.

Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')
첫번째 인자는 convolution 필터의 수 입니다.
두번째 인자는 convolution 커널의 행과 열입니다.
padding = 경계 처리 방법을 나타냅니다.
-> valid = 유효한 영역만 출력. 따라서 출력 이미지사이즈는 입력 사이즈보다 작습니다.
-> same = 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.

input shape = 샘플 수를 제외한 입력 형태를 정의. 모델에서는 첫 레이어일때만 정의하면 됩니다.
(행, 열, 채널 수)로 정의. 흑백영상인 경우 채널이 1, 컬러 영상인 경우 채널을 3으로 설정

activation = 활성화 함수 설정
linear = 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력됩니다.
relu(Rectified Linear Unit) = rectifier 함수, 딥러닝에서 사용하는 활성화 함수.
x>0이면 기울기가 1인 직선, x<0이면 함수값이 0이 됩니다.
sigmoid, tanh 함수와 비교했을때 학습이 훨씬 빠릅니다.
연산 비용이 크지 않고 구현이 간단하다는 것이 장점입니다.
x<0인 값들에 대해서는 기울기가 0이기 때문에 뉴런이 죽을 수 있는 단점이 존재합니다.

sigmoid = 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰이는 함수입니다.
softmax = 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

image_data_format이 channels_first인 경우 (샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서
image_data_format이 channels_last인 경우 (샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서

출력형태는 행과 열의 크기는 padding = same인 경우 입력 형태의 행과 열의 크기가 동일하게 출력됩니다.

경계 처리 방법으로 convolution 레이어 설정 옵션에는 border_mode가 있는데 valid와 same으로 설정할 수 있습니다.
valid인 경우 입력 이미지 영역에 맞게 필터를 적용합니다. 그래서 출력 이미지 크기가 입력 이미지 크기보다 작아집니다. 반면
same은 출력 이미지와 입력 이미지 사이즈가 동일하도록 입력 이미지 경계에 빈 영역을 추가해 필터를 적용합니다.
same으로 설정 시, 입력 이미지에 경계를 학습시키는 효과가 있습니다.


max pooling 레이어
convolution 레이어의 출력 이미지에서 주요값만 뽑아 크기가 작은 출력 영상을 만듭니다. 이것은 지역적으로 사소한 변화가 영향을 미치지 않게 합니다.
pool_size = 수직, 수평 축소 비율을 지정. (2,2)면 출력 영상 크기는 입력 영상 크기의 반으로 줄어듭니다.
영상의 작은 변화라던지 사소한 움직임이 특징을 추출할 때 크게 영향을 미치지 않도록 합니다.

Flatten 레이어
CNN에서 convolution 레이어나 맥스풀링 레이어를 반복적으로 거치면 주요 특징만 추출됩니다. convolution 레이어나 맥스풀링 레이어는 주로 2차원 자료를 다룹니다. 하지만 전결합층에 전달하기 위해서
1차원 데이터로 바꿔줘야 합니다. 이때 Flatten 레이어가 사용됩니다.
이전 레이어의 출력 정보를 이용하여 입력 정보를 자동으로 설정해주기 때문에 별도로 개발자가 파라미터를 지정해주지 않아도 됩니다.

Dense 레이어
활성화 함수로는 relu와 softmax가 있습니다. softmax가 relu보다 작은 단위의 활성화 함수입니다


Dropout()는 특정 노드에 학습이 지나치게 몰리는 것을 방지하기 위해 랜덤하게 일부 노드를 꺼주는 역할을 합니다.
과적합을 조금 더 효과적으로 회피할 수 있습니다.

'''
