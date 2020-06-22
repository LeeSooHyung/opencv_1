import cv2
import numpy as np
from solver import *
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from image_utils import *

model = create_model()

cap = cv2.VideoCapture(0)
#특정 키를 누를때까지 무한 반복을 위해 사용했습니다
while True:
    #비디오의 한 프레임씩 읽습니다. 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타납니다.
    #fram에 읽은 프레임이 나옵니다
    ret, frame = cap.read()

    dilated_image = preprocessing_image(frame)

    approx = find_sudoku_border(dilated_image)

    if len(approx)==4:
        aligned_image = align_sudoku(dilated_image, approx)
        
        square_images_list = obtain_squares_list(aligned_image)

        numbered_squares_list = detect_numbers(square_images_list)

        digits_dict = predict_digits(square_images_list, numbered_squares_list, model)
        
        sudoku_string = create_string(digits_dict)

        answer_dict = solve('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    
        answers_list = list(answer_dict.items())

        aligned_original_image = align_sudoku(frame, approx)

        answered_image = write_answers_on_image(sudoku_string, aligned_original_image, answers_list)

        aligned_answered_image = inverse_perspective(answered_image, approx, frame)

        final_answered_image = cv2.addWeighted(frame, 0.5, aligned_answered_image, 0.5, 0)

        cv2.imshow('Sudoku_1T', final_answered_image)

    else:
        cv2.imshow('Sudoku_1T',frame)
    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release() #오픈한 캡쳐 객체를 해제합니다
cv2.destroyAllWindows() #화면에 나타난 윈도우를 종료합니다.


'''
cv2.VideoCapture()를 사용해 비디오 캡쳐 객체를 생성할 수 있습니다. 안의 숫자는 장치 인덱스(어떤 카메라를 사용할 것인가)입니다.
1개만 부착되어 있으면 0, 2개 이상이면 첫 웹캠은 0, 두번째 웹캠은 1으로 지정합니다


cv2.imread(fileName, flag) : fileName은 이미지 파일의 경로를 의미하고 flag는 이미지 파일을 읽을 때 옵션입니다

flag는 총 3가지가 있습니다.
cv2.IMREAD_COLOR(1) : 이미지 파일을 Color로 읽음. 투명한 부분은 무시하며 Default 설정입니다
cv2.IMREAD_GRAYSCALE(0) : 이미지 파일을 Grayscale로 읽음. 실제 이미지 처리시 중간 단계로 많이 사용합니다
cv2.IMREAD_UNCHAGED(-1) : 이미지 파일을 alpha channel 까지 포함해 읽음

cv2.imshow(tital, image) : title은 윈도우 창의 제목을 의미하며 image는 cv2.imread() 의 return값입니다



'''








