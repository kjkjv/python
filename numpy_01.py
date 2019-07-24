# numpy_01.py

# pip install '모듈명'
# pip uninstall '모듈명'
#
# file => settings => 프로젝트 인터프린터 => 플러스 눌러서 설치 가능

data1 = [0,1,2,3,4,5]
import numpy as np
a1 = np.array(data1)
print(type(a1))
print(a1)

# ====================numpy 실습=============================================================
# 1. numpy로 5행 6열 2차원 배열을 임의로 만들고 아래 지시대로 출력해보세요
#
# 	1.1 데이터를 거꾸로 출력해보세요.
# 	1.2 마지막 열을 제외한 모든 열을 출력해보세요.
# 	1.3 전치(transpose) 행렬을 출력해보세요
# 	1.4 2차원을 1차원 배열의 형태로 변형하여 출력하세요

import numpy as np
data_56 = np.arange(30).reshape(5,6)
print(data_56)

print(data_56[::-1,::-1])
print(data_56[:,:-1])
re = data_56.transpose()
print(re)
f = data_56.flatten()
print(f)


# 2. numpy를 사용하여 아래 두개의 행렬을 만들고 지시대로 출력해보세요
# 	2.1 두개의 행렬을 수평으로 합쳐 결과를 출력하세요
# 	2.2 두개의 행렬을 수직으로 합쳐 결과를 출력하세요
# 	2.3 두개의 행렬을 열로 합쳐 결과를 출력하세요
# 	2.4 두개의 행렬을 행으로 합쳐 결과를 출력하세요

data_2_1 = np.arange(16).reshape(4,-1)
print(data_2_1)
d2 = data_2_1 * 2
print(d2)

result = np.hstack((data_2_1,d2))
print(result)

result2 = np.vstack((data_2_1,d2))
print(result2)

worhkd = np.column_stack((data_2_1,d2))
rhkdwo = np.row_stack((data_2_1,d2))
print(worhkd)
print(rhkdwo)


# 3. scipy.misc 모듈의 face()함수로 얻어온 미국 너구리(raccoon)
#    얼굴 사진(face)을 아래 내용대로 변환 하는 코드를 구현하세요
#    subplot() 함수를 사용하여 4개를 같은  윈도우 창에 출력하게 하세요
#    face.shape : (768,1024,3)
#
# 	3.1  Red 색상을 모두 0 으로 변경하여 출력한다
# 	3.2  Green 색상을 모두 0 으로 변경하여 출력한다
# 	3.3  Blue 색상을 모두 0 으로 변경하여 출력한다
# 	3.4  Red, Green, Blue 색상 중  100보다 작은 경우
#      	     모두 0 으로 변경하여 출력한다

import scipy


import matplotlib.pyplot as plt

scipy.misc.face()

import scipy.misc
import matplotlib.pyplot as plt

raccoon = scipy.misc.face()
raccoon2 = raccoon.copy()
print(type(raccoon2))
print(raccoon2.shape)
print(raccoon2.nbytes)
print(raccoon2.size)
print(raccoon2.dtype)
print(raccoon2.ndim)
plt.imshow(raccoon2)
plt.show(raccoon2)
plt.imshow(raccoon2.view)
plt.show(raccoon2)
plt.subplot(221)
plt.imshow(raccoon2)
plt.subplot(222)
plt.imshow(raccoon2)
plt.subplot(223)
plt.imshow(raccoon2)


print(raccoon2)
raccoon2[:,:,0] = 0
raccoon2[raccoon2[:,:,0]<10] = 255             # 요런 방법도 있다.
raccoon2[:,:,1] = 0
raccoon2[:,:,2] = 0
plt.imshow(raccoon2)











# =====================================================================================


data2 = [0,1,2,3,4,5]
print(data2)              # [0, 1, 2, 3, 4, 5]
print(type(data2))        # <class 'list'>

a2 = np.array(data2)      # 배열, array = 정렬
print(a2)                 # [0 1 2 3 4 5]
print(type(a2))           # <class 'numpy.ndarray'>

a3 = np.arange(6)
print(a3)                 # [0 1 2 3 4 5]
print(a3.shape)           # (6,), 1차원

a3 = np.arange(12).reshape(3,4)
print(a3)                  # [[ 0  1  2  3]
                           # [ 4  5  6  7]
                           # [ 8  9 10 11]]
print(a3.shape)            # (3, 4)

# 대괄호의 갯수로 차원을 구하고
# 그 안의 갯수로 모양을 구한다
# 마지막은 괄호 안의 객체의 갯수를 파악

print(a3.shape)
print(a3.ndim)             # 2 (차원)

a4 = np.arange(12).reshape(2,2,-1)
print(a4)
# [[[ 0  1  2]
#   [ 3  4  5]]
#  [[ 6  7  8]
#   [ 9 10 11]]]
print(a4.shape)            # (2, 2, 3)
print(a4.ndim)             # 3 (차원)

# 인덱싱
a4[0,0,0],a4[0,0,1],a4[0,0,2]        # (0, 1, 2)
a4[0,1,0],a4[0,1,1],a4[0,1,2]        # (3, 4, 5)
a4[1,0,0],a4[1,0,1],a4[1,0,2]        # (6, 7, 8)
a4[1,1,0],a4[1,1,1],a4[1,1,2]        # (9, 10, 11)

import time
start_time = time.time()
# for 문을 사용하여 인덱싱하는 방법 두 개가 같은 것이지만 시간이 오래 걸린다.
print('-'*50)
for x in range(2):
    for y in range(2):
        for z in range(3):
            print(a4[x],[y],[z])
            print(a4[x,y,z])
end_time = time.time()
tiime = end_time - start_time
print('경과된 시간은 = ',tiime)


print('-'*50)
for x in range(a4.shape[0]):
    for y in range(a4.shape[1]):
        for z in range(a4.shape[2]):
            print(a4[x],[y],[z])
            print(a4[x,y,z])
print()

print(a4[0])
# [[0 1 2]
#  [3 4 5]]        ==> 2차원 배열

print(a4[0][0])
# [0 1 2]          ==> 1차원 배열

print(a4[0][0][0])
# 0                ==> 스칼라(0차원)
print(a4[0,0,0])
# 0                ==> a4[0][0][0]와 같은 값


# 조건 인덱싱
print(a4[a4>5])      # [ 6  7  8  9 10 11]
print(a4[a4<3])      # [0 1 2]
# for문을 돌려서 요소 하나하나를 검색해야 하는데
# 파이썬에서는 바로 할 수 있다
# 배열의 요소가 조건에 걸리는 것이지 배열이 조건에 걸린 것은 아니다.


# 슬라이싱
print(a4[:,:,:2])
# 행과 열은 전체 면은 2개만
# [[[ 0  1]
#   [ 3  4]]
#  [[ 6  7]
#   [ 9 10]]]

print(a4[:-1,:-1,:-1])
# [[[0 1]]]
# 끝에만 제거하기



