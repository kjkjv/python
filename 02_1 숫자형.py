# ====================0701==========================

#02_1 숫자형.py

# 숫자형 변수

# 표준 입출력
# 표준 입력 : 변수 = input("문자열")

# a=input()
# print(a)

math = input("수학점수 : ")        # str(문자열) 타입을 반환
english = input("영어점수: ")      # str(문자열) 타입을 반환
total = int(math) + int(english)  # 문자열을 숫자로 형변환
print('총점:',total )

# 표준 출력 : print()

print('add: ', 4 + 5, 'sub = ', 4- 2) # 마지막에 \n
print(1,2) ; print(3,4)               # 줄을 알아서 바꾼다, # 마지막에 자동으로 \n
print(1,2,end = ' ')                  # 마지막에 \n 대신 space로 출력
print(5,6,7)                          # ,(콤마) 사이 문자를 빈칸으로 구분하여 출력
print(5,6,7, sep= ',')
print(5,6,7, sep= ':')                # ,(콤마) 사이 문자를 : 으로 구분하여 출력


# print() formatting
num_1 = 10
num_2 = 3.124
str_1 = 'hello python'
print("int형:%d"%num_1)               # %d int 형
print("float형:%f"%num_2)             # %f float 형
print("16진수:0x%x"%num_1)            # %x 16진수 형
print("%f"%num_2)                     # int 형
print('%f'%num_1)                     # float 형
print('%d'%num_1)                     # %d  int 형
print('%f'%num_2)                     # %f  flat,실수형
print('%x'%num_1)                     # %x 16진수
print('%o'%num_1)                     # %o 8진수
print('문자열:%s'%str_1)

# print("int형:%d"%str_1)             # %d int형 ==> 문법오류

# format() 함수 사용 (질문하기)
print( format(1.23451, "4.3f"))       #+ 전체 4자리, 소숫점 3자리

print('{0} {1}'.format('apple',7.77))

print('{0:^10}{1:5.2f}'.format('apple',7.77))   # ^ 가운데 정렬
print('{0:<10}{1:5.2f}'.format('apple',7.77))
print('{0:=>10}{1:5.2f}'.format('apple',7.77))

# 숫자형 변수
a=12345
# int형
print(int(a))
print(bin(15))        # 2진수
print(hex(15))        # 2진수
print(oct(15))        # 2진수

# float형  : 실수형, 6자리까지, float < double < long double, long double는 더 길게 표시 가능, 저장 길이가 다름, 실수를 저장하는 공통점
print(float(a))

# complex형  : 허수형(복소수형)
print(complex(a))     # (12345+0j)

print(float('inf'))   # 무한대

num = float('inf')
print(num/1000)       # inf / 정수 ==> inf
print(num/num)        #inf / inf ==> nan, Not a Number
print(float('nan'))   # ==> 'nan'


# 사칙연산
#  : +, -, *, /, %, //

print("김재광"*4)


# ctrl + / : 주석문 설정/해제

# tab : 들여쓰기, 파이썬은 들여쓰기를 잘해야 한다.
# shift + tab : 들여쓰기 해제


# --------------run 실행-------------------
# shift + f10 : run (기존 수행했던 소스코드를 실행)
# ctrl + shift + f10 : run (새로운 코딩을 시작할 때)
# Alt + Shift + E : 한줄씩 또는 블럭한 것만 실행
# ==>오른쪽 마우스 버튼을 클릭하여 실행

# F8 : 디버거 모드에서 STEP OVER 를 할 때 사용


# # --------------- 디버거 ----------------
# 디버거 실행은 shift + f9
# 런이랑 실행이 다르다 !!!
# 파일 처음 실행할떄는 alt 추가
#  Shift + F9 : 현재 파일 디버거 모드
#  Alt + Shift + F9 : 현재 파일 디버거 모드
#  F8 : 디버거 모드에서 Step Over



# ====================0702==========================

# bool : true는 1  flase는 0 정수 값으로 간주된다.
a = 1
print( a>0 )     #True
print( a<0 )     #False

b = a > 0
print(b)
print(type(b))  #<class 'bool'>
print(type(a))  #<class 'int'>

c = True
print(type(c))  #<class 'bool'>
c = True + True
print(c, type(c))  #2, <class 'int'>, TRUE는 1이니까 사칙연산도 가능해지고 그때느 정수인 int 취급을 한다.
# 데이터를 숫자로 인식하는 기술로 가능하다.

d = bool(3)
print(d, type(d))  # True <class 'bool'>
d = bool([])
print(d, type(d))  # False <class 'bool'>
d = bool(0)
print(d, type(d))  # False <class 'bool'>, 0은 거짓.
d = bool([0])
print(d, type(d))  # True <class 'bool'>, 리스트에 0이 들어있으면 트루.
d = bool(-3)
print(d, type(d))  # True <class 'bool'>
# 0이 아니면 True, 거짓이 아니면 다 참이다. 비어있는 리스트도 False