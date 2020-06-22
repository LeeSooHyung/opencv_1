import cv2 
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D




'''
contour의 면적은 cv2.contourArea 함수로 구할 수 있습니다.
'''
def find_max_area_contour(contours):
    
    area = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > area:
            area = cv2.contourArea(contours[i])
            index = i
    return index

def warp_coord(pts1):
    pts2 = [[0,0],[0,297],[297,0],[297,297]]
    ref = np.amax(pts1)//2
    pts1 = list(pts1)
    for i in range(len(pts1)):
        if pts1[i][0] < ref:
            if pts1[i][1] < ref:
                
                pts2[i] = [0 , 0]
            elif pts1[i][1] > ref:
               
                pts2[i] = [0, 297]
        elif pts1[i][0] > ref:
            if pts1[i][1] < ref:
                
                pts2[i] = [297, 0]
            elif pts1[i][1] > ref:
                
                pts2[i] = [297, 297]
    return np.float32(pts2)
'''
grayscale = 영상이나 이미지의 색상을 흑백 색상으로 변환하기 위해서 사용합니다.

'''





def preprocessing_image(image):
    #cv2.cvtcolor(원본 이미지, 색상 변환코드)를 이용해 이미지의 색상 공간을 변경 가능.
    #여기서 색상 변환 코드는 원본 이미지 색상 공간과 결과 이미지 색상 공간을 의미.
    #원본 이미지 색상 공간은 원본 이미지와 일치해야 합니다.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    #opencv에서는 GaussianBlur 함수가 있습니다.
    #GaussianBlur(원본 이미지, 커널 크기, 표준편차)
    #가우시간 블러로 잡음 제거
    blur_image = cv2.GaussianBlur(gray_image, (9,9), 0)


    '''
이진화 처리(흑/백으로 분류하여 처리하는 것)는 간단해 보이지만 쉽지 않습니다. 이때 기준이 되는
임계값이 가장 중요한 문제가 됩니다. 임계값보다 크면 백, 작으면 흑이 됩니다. 기본 임계처리는 개발자가 고정된 임계값을 결정하고
그 결과를 보여주는 단순한 형태입니다. 이때 사용되는 함수는 threshold()입니다.
그렇지만 threshold 함수는 임계값을 이미지 전체에 적용하여 처리하기 때문에 하나의 이미지에 음영이 다르다면 일부 영역이 모두 흑/백으로 출력됩니다.
그것을 해결하기 위해 cv2.adaptiveThreshold()를 사용했습니다.

    '''
    thresh = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #thresh = cv2.adaptiveThreshold(grayscaleimage, 임계값, thresholding value를 결정하는 계산 방법, threshole type, 영역 사이즈, 평균이나 가중평균에서 차감할 값)



    '''
bitwise 비트 연산 진행,
이미지에서 백을 true(1) / 흑을 False(0)
    '''
    binary_image = cv2.bitwise_not(thresh, thresh)

    #개발자의 커널 크기에 따라 외곽에서 0이 되는 픽셀의 정도가 달라집니다.
    kernel = np.ones((1,1), np.uint8)

    #바이너리 이미지에서 흰색 오브젝트의 외곽 픽셀 주변에 흰색(1)을 추가합니다.
    #추가로 노이즈를 없애기 위해 사용한 Erosion에 의해 작아졌던 오브젝트를 원래대로 돌리거나 할때 사용할 수 있습니다.
    dilated_image = cv2.dilate(binary_image, kernel)

    return dilated_image



'''
contours는 동일한 색 또는 동일한 강도를 가지로 있는 영역의 경계선을 연결한 선입니다. 찾아보니 우리 실생활에선 일기예보나 등고선가 이 예에 해당했습니다.
대상의 외형을 파악하는데 유용하게 사용됩니다.
정확도를 높이기 위해 바이너리 이미지를 사용합니다. threshold로 선처리를 수행 한 후 cv2.findContours 함수를 사용해서 원본 이미지를 직접 수정합니다.
opencv에서 coutour를 찾는 것은 검은색 배경에서 흰색 대상을 찾는 것과 비슷합니다. 그래서 대상은 흰색, 배경은 검은색으로 합니다.


cv2.RETR_EXTERNAL : contours line중 가장 바같쪽 Line만 찾음.
cv2.RETR_LIST : 모든 contours line을 찾지만, hierachy 관계를 구성하지 않음.
cv2.RETR_CCOMP : 모든 contours line을 찾으며, hieracy관계는 2-level로 구성함.
cv2.RETR_TREE : 모든 contours line을 찾으며, 모든 hieracy관계를 구성함


cv2.CHAIN_APPROX_NONE : 모든 contours point를 저장.
cv2.CHAIN_APPROX_SIMPLE : contours line을 그릴 수 있는 point 만 저장. (ex; 사각형이면 4개 point)
cv2.CHAIN_APPROX_TC89_L1 : contours point를 찾는 algorithm
cv2.CHAIN_APPROX_TC89_KCOS : contours point를 찾는 algorithm

'''

def find_sudoku_border(image):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

         #contours, hierarchy = cv2.findContours(이미지,contour 탐색 방법, contours를 찾을 때 사용하는 근사치 방법)
       #cv2.CHAIN_APPROX_NONE 는 모든 point를 저장하고 cv2.CHAIN_APPROX_SIMPLE 는 4개의 point만을 저장하여 메모리를 절약합니다.
       #hierachy는 contours line의 계층 구조(contour의 상하구조)
      
        index = find_max_area_contour(contours)

        
          #close : 닫힘 여부 (테두리를 닫고 싶으면 True) / contour의 둘레를 구하는 과정
        epsilon = 0.1*cv2.arcLength(contours[index], True)

               #close : 닫힘여부 / 테두리를 열고 싶으면 True
        approx = cv2.approxPolyDP(contours[index], epsilon, True)

        return approx

def align_sudoku(image, approx):
        try:
            #pts1을 pts2까지 이동시키는 M을 만든다.
            pts1 = np.float32(approx.reshape(4,2))
            pts2 = warp_coord(pts1)
            '''
            Perspective Transformation
            원근법 변환은 직선의 성질만 유지되며 선의 평행성은 유지가 되지 않는 변환입니다.
            하나의 예로 기차길을 들 수 있는데 서로 평행하지만 원근변환을 거치면 평행성은 유지 되지 못하고
            나중에 하나의 점에서 만나는 것처럼 보이게 됩니다.
            이처럼 Perspective Transformation은 4개의 포인트가 필요합니다.
            변환행렬을 구하기 위해 cv2.getPerspectiveTransform() 함수가 필요하고 cv2.warpPerspective() 함수에 변환 행렬값을 적용하면
            최종 결과 이미지를 얻을 수 있습니다.

            '''
            
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(image, M, (297,297))
        except:
            return image
        else:
            return dst

def obtain_squares_list(image):
        x_size, y_size = 33,33
        squares = []
        for y in range(1,10):
            for x in range(1,10):
                squares.append(image[(y-1)*y_size:y*y_size, (x-1)*x_size:x*x_size])
        return squares

def detect_numbers(squares):
    numbered_squares = list()
    for i in range(81):
        if np.var(squares[i][10:23, 10:23]) > 10000:
            numbered_squares.append(i)
    return numbered_squares

def create_model():
        input_shape = (28,28,1)
        num_classes = 10
        model = Sequential() #케라스 모델 시작 Sequential()
         #add() 메소드를 통해서 쉽게 레이어를 추가할 수 있습니다.
       
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
        model = load_model('mnist_model.h5')

        return model



#이미지 사이즈 변경하기
def predict_digits(squares, numbered_squares, model):
        my_dict = {}
        for index in numbered_squares:
            buff = squares[index][3:30,3:30]
            buff = cv2.resize(buff, (28,28))
            buff = buff.reshape(1,28,28,1)

            #모델 사용ㅇ하기
            label = model.predict_classes(buff)
            my_dict[index] = label[0]
            
        return my_dict

def create_string(detected_numbers_dictionary):
        sudoku = ['0' for i in range(81)]
        for key, value in detected_numbers_dictionary.items():
            sudoku[key] = str(value)
        grid1 = ''.join(sudoku)

        return grid1

def write_to_image(image, item):
    row = item[0][0]
    column = item[0][1]
    value = item[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
     #cv2.putText(이미지 파일, 출력문자, 출력문자 시작 위치 좌표, 폰트, 폰트 크기, 폰트 색상, 라인 파)
    
    cv2.putText(image, value, (9+33*(int(column)-1), 25+33*(ord(row)-65)), font, 0.7, (0,255,0),2, cv2.LINE_AA)
    return image

def write_answers_on_image(grid1, image, answers):
    for i in range(81):
        if grid1[i]=='0' or grid1[i]=='.':
            frame = write_to_image(image, answers[i] )
    return frame

def inverse_perspective(answer_image, approx, image):
    pts1 = np.float32(approx.reshape(4,2))
    pts2 = warp_coord(pts1)

    N = cv2.getPerspectiveTransform(pts2, pts1)
    dst = cv2.warpPerspective(answer_image, N, image.shape[1::-1])

    return dst
