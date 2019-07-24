# 05-4 예외처리_외장함수 .py

# 예외(exception)
# => class로 구성되어 있다.
# try :
#       ...
# except : <예외 종류> :
#       ...
# finally :
#       ...


a, b = 5,0
try :
    c = a / b                 # ZeroDivisionError
except ZeroDivisionError :    # except를 안다면 쓰고 모르면 빈칸으로!
    print(b, '로 나누었습니다.(ZeroDivisionError)')
    pass

try :
    d = a + value             # NameError(value에 값이 없으니)
except NameError :
    print('틀렸다 재광아(NameError)')
    pass

score = '결석'
score = 10
try :
    e = score + a             # TypeError
except TypeError :
    e = 0 + a
    print(e, '(TypeError)')
finally:
    print('finally:',e)
print('정상 종료되었습니다.')


try:
    f = open('hello.txt','r')     # FileNotFoundError
except FileNotFoundError :
    pass
    print('(FileNotFoundError)')


# 재귀함수
def add(a,b):
    c = a + b
    add(a,c)
print(add(1,3))                 # RecursionError

# 파이썬 문서에 가면 예외 종류 출력이 가능



# 파이썬 외장함수

# sys 모듈 : 터미널 명령모드에서 인자 정보와 시스템 경로(path) 변수


import sys
print(sys.argv)
# 터미널에서 아래와 같이 실행
print(sys.argv[0])    # 예외처리_외장함수.py
print(sys.argv[1])    # hello

print(sys.path)       # 시스템의 환경 변수를 출력
sys.path.append('c:/myforder/경로설정')    # 시스템의 환경변수에 등록


# os 모듈 : 파일, 디렉토리, 환경변수, os 자원

import os
os.system('dir')      # MS-DOS 명령어를 직접 실행할 수 있다.
os.system('cls')



# time 모듈 : 시간 관련 모듈
import time
print(time.time())
start_time = time.time()
time.sleep(10)                 # 지연한숨
end_time = time.time()

rlrks = end_time - start_time
print(round(rlrks),'sec')      # round 는 소수점을 빼는 함수


# range 함수 걸리는 시간도 알 수 있다.
import time
print(time.time())
start_time = time.time()
# time.sleep(10)
for k in range(10000):
    print(k)
end_time = time.time()
print()
rlrks = end_time - start_time
print(rlrks,'sec')

# 시간을 알아보는 함수
print(time.ctime(),'sec')

time_now = time.ctime()
print(type(time_now))
print(time_now)
titititi = time_now[11:19]
print(titititi)


import calendar
print(calendar.calendar(2019))
print(calendar.prmonth(2019,7))
print(calendar.monthrange(2019,8))


# random 모듈
import random

# random() 함수 0 <= 0.0 <= 1
num = random.random()
print(num)

for k in range(10):
    num = random.random()
    print(num)
print()
# 랜덤값이 무엇이냐에 따라 성능이 달라질 수 있다.

random.seed(3)
num = random.random()
print(num)
# seed 값이 같기 때문에 결과값이 같다.

random.seed(time.time())
num = random.random()
print(num)


# 랜덤 수를 뽑는다
# randint(a,b)   # a <= num <= b , 정수값을 생성
num = random.randint(1,24)
print(num)


# choice(시퀀스형)
mylist = [14,5,43,534,242,4,24,445,12]
mylist2 = ['a','m','f','r','d']

num = random.choice(mylist)
string = random.choice(mylist2)
print(num,string)

# sample(시퀀스형, 갯수) : 중복되지 않은 요소를 추출
num_list = random.sample(mylist,3)
string_list = random.sample(mylist2,3)
print(num_list,string_list)

# shuffle() : 원본 시퀀스형 데이터를 무작위로 섞는다.
num_shuffle = random.shuffle(mylist)
print(mylist)

