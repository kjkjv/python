#1번
g = input("거리 : ")
s = input("속도 : ")
total = int(g) / int(s)
print(total)

#2번
g = input("거리 : ")
n = input("너비 : ")
m = int(g)*int(s)
print(m)

d = (int(g)*2) + (int(n)*2)
print(d)


#3번
h = input("화씨 : ")
s = (int(h)-32)/1.8
print(s)

#4번
a = input("a : ")
b = input("b : ")
print(("덧셈 :", int(a) + int(b)), ("뺄셈 : ", int(a) - int(b)), ("나눗셈 : ", int(a) * int(b)), ("곱셈 : ",int(a) / int(b)))

#print(total)

# eval() 문자열을 int로 안바꿔도 숫자로 계산해주는 식.



# =======================강사님 답안===============================================#
#numeric_ex.py

#1번
print( '{0:=^50}'.format( '1번' ) )
velocity = input( 'Input velocity : ' )
distance = input( 'Input distance : ' )

#time = eval( distance + '/' + velocity )
time = int(distance) / int(velocity)

print()
print( 'velocity : {0:<6.2f}'.format( float( velocity ) ) )
print( 'distance : {0:<6.2f}'.format( float( distance ) ) )
print( 'time     : {0:<6.2f}'.format( time ) )


#2번
print( '{0:=^50}'.format( '2번' ) )
length = input( 'Input length : ' )
width = input( 'Input width : ' )

#area = eval( length + '*' + width )
area =  int(length) * int(width)
#circumference = eval( length + '*' + '2' + '+' + width + '*' + '2' )
circumference =  int(length) * 2 + int(width) *2

print()
print( 'length : {0:<6.2f}\twidth : {1:<6.2f}'.format( float( length ), float( width ) ) )
print( 'area : {0:<6.2f}'.format( area ) )
print( 'circumference : {0:<6.2f}'.format( circumference ) )

#3번
print( '{0:=^50}'.format( '3번' ) )
fahrenheit = float( input( 'Input fahrenheit : ' ) )

celsius = ( fahrenheit - 32 ) / 1.8

print()
print( 'fahrenheit : {0:<6.2f} -> celsius : {1:<6.2f}'.format( fahrenheit, celsius ) )

#4번
print( '{0:=^50}'.format( '4번' ) )
number1 = int( input( 'Input number1 : ' ) )
number2 = int( input( 'Input number2 : ' ) )

add = number1 + number2
subtract = number1 - number2
multiple = number1 * number2
divide = number1 / number2

print()
print( '{0:^6} + {1:^6} = {2:<6}'.format( number1, number2, add ) )
print( '{0:^6} - {1:^6} = {2:<6}'.format( number1, number2, subtract ) )
print( '{0:^6} * {1:^6} = {2:<6}'.format( number1, number2, multiple ) )
print( '{0:^6} / {1:^6} = {2:<6.2f}'.format( number1, number2, divide ) )




# =========문자열 실습과제==============#

# 1번
a = 'hong gil dong201912121623210'
num = a.find('20191212')      #13
name ='Name : '+ (a[:13])
birthday ='Birthday : ' + (a[13:17] + '/' + a[17:19] + '/' + a[19:21])
id_number ='ID Number : ' + (a[13:21] + '-' + a[21:28])
print(name)
print(birthday)
print(id_number)


#  2번
a = 'PythonProgramming'
s2 = (a[:6])
s1 = (a[6:])

s3 = s1 + s2
print(s3)


# 3번
s = 'hello world'
a = s.replace('hello', 'hi')
print(a)



# =========[ list 실습과제 ]==============#

# 1번

a = input('a:')
b = input('b:')
list = ['+', '-', '*', '/']
op_select = int( input( 'Input operator( 1:+, 2:-, 3:*, 4:/ ) : ' ) )
c = list[op_select-1]      #생각하지 못했던 개념
eval(a+c+b)
print(eval)


# 2번

n = int(input('n :'))+1
a = range(1,n)      #range는 list로 하면 안된다. 그리고 n은 문자열이므로 숫자열로 바꿔줘야 한다.
print(sum(a))


# 3번

# 1 ~ n까지 짝수합과 홀수합을 출력하는 프로그램을 리스트를 이용하여 작성하시오.
# (최대값 n을 input()함수로 입력 받아 사용하세요)

n = int(input('n:'))+1     #숫자열과 문자열을 분명히 구분하기
a = range(1,int(n),2)
print('홀수합',sum(a))
a = range(0,int(n),2)
print('짝수합',sum(a))

# 4번

# 1 ~ n까지 3의 배수와 5의 배수를 제외한 수를 출력하고 그 합을 출력하는 프로그램을 작성하시오.
# (최대값 n을 input()함수로 입력 받아 사용하세요)


# 풀이 1번째
n = input('n:')
a = range(1,int(n))          # 파이썬은 tab으로 들여쓰기가 중요.
list=[]                      # 리스트 함수를 만듬.
for b in a :                 # for 변수 in 리스트 :
    if b%3!=0 and b%5!=0 :   # if 에서 true면 바로 밑에 줄로 이동.
      list.append(b)         # .append로 추가
print(sum(list))



#  풀이 2번째
n = int(input('n:'))+1          # n의 값을 입력하는데 숫자열로 바꿔주고 range를 의식해서 +1로
a = range(1,n)                  # range 범위를 설정
c = 0                           # 리스트 대신 변수로 만들고 싶으면 0의 변수를 만들어준다
for b in a:                     # for 문은 in 과 함께 for 새로운 변수 in 리스트
    if b%3!=0 and b%5!=0 :      # 3의 배수가 아니고 5의 배수가 아니면 true, true가 나오면 바로 밑으로 간다. 나머지는 버림
        c = c+b                 # 나오는 b값에 c값을 계속해서 추가하여 더한다. sum 함수를 사용하지 않아도 괜춘

print(c)

print('5의 배수가 아닌 것의 합',sum(list_1))




# ========================LIST실습문제/강사님 답안===========================================================
# 1번 답안
print( '{0:=^50}'.format( '4' ) )
op = [ '+', '-', '*', '/' ]
number1 = input( 'Input number1 : ' )
number2 = input( 'Input number2 : ' )
op_select = int( input(
    'Input operator( 1:+, 2:-, 3:*, 4:/ ) : ' ) )

index = op_select - 1
result = eval( number1 + op[ index ] + number2 )

print()
print( 'number1 : {0:^8.2}'.format( number1 ) )
print( 'number2 : {0:^8.2}'.format( number2 ) )
print( '{0:^6} {2:^3} {1:^6} = {3:<.2f}'.format(
    number1, number2, op[ index ], result ) )

# 2번 답안
print( '{0:=^50}'.format( '5' ) )
max_number = int( input( 'Input max number : ' ) )

l = list( range( 1, max_number + 1 ) )

print()
print( l )
print( '1 ~ {0:^6} = {1:<8}'.format(max_number, sum( l )))


# 3번 답안
print( '{0:=^50}'.format( '6' ) )
max_number = int( input( 'Input max number : ' ) )

even = list( range( 2, max_number + 1, 2 ) )
odd = list( range( 1, max_number + 1, 2 ) )

print()
print( 'even number : ', even )
print( '1 ~ {0:^6} = {1:<8}\n'.format( max_number, sum( even ) ) )

print( 'odd number : ', odd )
print( '1 ~ {0:^6} = {1:<8}'.format( max_number, sum( odd ) ) )

# 4번 답안
print( '{0:=^50}'.format( '7' ) )
max_number = int( input( 'Input max number : ' ) )

l3 = [ x for x in range( 1, max_number + 1 ) if x % 3 == 0 ]
l5 = [ x for x in range( 1, max_number + 1 ) if x % 5 == 0 ]
l = [ x for x in range( 1, max_number + 1 ) if x % 3 != 0 and x % 5 != 0 ]

print()

print( 'Multiple of 3 : ', l3, '\n' )
print( 'Multiple of 5 : ', l5, '\n' )
print( 'Excluding Multiple of 3 and 5 : ', l )
print( 'sum = {0:<6}'.format( sum( l ) ) )





# ====================튜플문제=========================
# 1번
a=('a1','a2','a3','a4')
b=('b1','b2','b3','b4')

# (1) q, w, e, r 변수에 튜플 a의 구성요소들을 차례대로 하나씩 넣으시오.(ex) q='a1'
q = a[0]
print(q)
w = a[1]
print(w)
e = a[2]
print(e)
r = a[3]
print(r)

# 1
q, w, e, r = ('a1','a2','a3','a4')
print(q,w,e,r)


# (2) a와 b를 더한 값을 c에 넣어보세요
c = (a+b)
print(c)


# (3) c의 3번째 자리의 구성요소는 무엇인가?
print(c[2])

# (4) 6번째 부터 끝까지의 구성요소는 무엇인가?
print(c[5:])

# (5) 처음부터 3번째의 구성요소는 무엇인가?
print(c[:3])

# (6) 4번째 구성요소 제거해 보세요 ==>에러 발생
del a[3]

# (7) 5번째 구성요소의 값을 'c1'로 수정해보세요 ==>에러 발생
c[4] = 'c1'

c.replace(c[4], 'c1')


#======================= 튜플실습문제.강사님 정답 ==============================

a=('a1','a2','a3','a4')
b=('b1','b2','b3','b4')

# 1,언패킹
q, w, e, r = ('a1','a2','a3','a4')
print(q,w,e,r)

# 2, + 연산
c = a + b
print(c)

# 3, 인덱싱
print(c[2])

# 4, 슬라이싱
print(c[5:])

# 5, 슬라이싱
print(c[:3])

# 6
del a[3]
# TypeError: 'tuple' object doesn't support item deletion

#7,
c[4] = 'c1'
# TypeError: 'tuple' object does not support item assignment



# ======dic 연습문제==============
srp={'가위':'보','바위':'가위','보':'바위'}

# (1) srp의 key list 생성
x = srp.keys()

# (2) srp의 value list 생성
y= srp.values()

# (3) srp의 key와 value 의 한쌍으로된 리스트 생성
srp.items()

# (4) srp의 key '가위'에 해당하는 value 출력
srp.get('가위')

# (5) srp의 value '바위'에 해당하는 key 출력
for key, value in srp.items():
    if value == '바위':
        print(key)
type(key)               # <class 'str'>

x = [key for key, value in srp.items() if value == '바위']   # pop() 사용
print(x)
type(x)

# (6) srp에 '찌':'빠', '묵':'찌', '빠':'묵' 추가
x = {'찌':'빠', '묵':'찌', '빠':'묵'}
srp.update(x)
print(srp)

# (7) srp 보자기 라는 키가 있는지 확인
'보자기' in srp
# False

# (8) srp의 key 와 value를 서로 바꾸어서 새로운 사전 srp2를 생성
srp2 = {y:x for x,y in zip(x,y)}   # dic={}, 내장함수가 자료형에 감싸있어야 제기능 가능.
print(srp2)                        # 변수 이름을 다르게

type(srp2)                         # <class 'dict'>
# comprehension 은 list, dic에 따라 괄호가 달라진다. list=[], dic={}

# 내장함수는 기존의 for 문 등과 비교하여 처리 속도가 비교가 안된다.




#===================== DICT실습문제_강사님 답안==========================================

#
srp = {'가위':'보','바위':'가위','보':'바위'}

# 1
print(list(srp.keys()))

# 2
print(list(srp.values()))

# 3
print(list(srp.items()))

# 4
print(srp['가위'])

# 5
# 파이선 스타일 방식
a = [x for x,y in srp.items() if y == '바위']
print(a[0])

# 전통적인 언어의 방식
for x,y in srp.items():
    if y == '바위':
        a = x
print('key =',a)

# 6
b = {'찌':'빠', '묵':'찌', '빠':'묵'}
srp.update(b)
print(srp)

# 7
print('보자기' in srp)

# 8

# 파이선 스타일 방식
#srp = {1: '보',2:'바위', 3:'가위', 4:'묵', 5:'찌', 6:"빠"}
srp2 = { y:x for x,y in srp.items() }
print(srp2)

# 전통적인 언어의 방식
srp2 = {}
for x,y in srp.items():
    srp2.update({y:x})

print('srp2 =',srp2)



# =================[집합실습]============================

# (1) a = [1,2,3,4] 로 set s1을 생성하시오. b = "aabbccddeeff"로 set s2를 생성하시오.
s1 = {1,2,3,4}
s2 = {'aabbccddeeff'}
type(s2)

# (2) s1 에 a,b,c 를 추가하시오.
s1.update({'a,b,c'})
print(s1)

# (3) s2 에 1,2를 추가하시오.
s2.update({1,2})
print(s2)

# (4) s1과 s2의 교집합을 구하시오.(2가지 방법 모두 )
s1 & s2
s1.intersection(s2)

# (5) s1과 s2의 합집합을 구하시오.(2가지 방법 모두)
s1.union(s2)
s1 | s2

# (6) s1과 s2의 차집합을 구하시오.(기호)
s1 - s2

# (7) s2와 s1의 차집합을 구하시오.(함수)
s1.difference(s2)

# (8) s2에서 1을 빼보세요.
s2.remove(1)
print(s2)

# (9) s1과 s2의 대칭 차집합을 구하시오.
s1.symmetric_difference(s2)




# =======연습문제_112p============================

# 문제 1번
# 홍길동 씨의 평균 점수는?
k = 80
e = 75
m = 55

mean = (k+e+m)/3
print(mean)


# 문제 2번
# 자연수 13이 홀수인지 짝수인지 판별하라

a = 13

if a % 2 == 0:
    print('짝수')
else : print('홀수')

# 강사님 답안
num = 13
even_odd = ['짝수', '홀수']
print("%d : %s"%(num,even_odd[num%2]))          # 인덱스로 출력
print("%d 은 %s 입니다."%(num,even_odd[num%2]))

# 문제 3번
# 홍길동 주민 881120-1068234 나누어 보자

pin = '8811201068234'
yyyymmdd=pin[:6]
print(yyyymmdd)
num = pin[6:]
print(num)


# 문제 4번
# 주민 성별을 나타내는 숫자 출력
g = ['남자','여자']
pin = '8811202068234'
print(g[int(pin[6])-1])


# 문제 5번
# replace를 써서 바꿔보자!
a = "a:b:c:d"
b = a.replace(':','#')
print(b)


# 문제 6번
# 리스트 변환!
a = [1,3,5,4,2]
a.sort()
print(a)
a.reverse()
print(a)


# 문제 7번
# 문자열로 출력!
a = ['life', 'is', 'too', 'short']
result = ' '.join(a)
print(result)


# 문제 8번
#  튜플 값을 추가하자!
a = (1,2,3)
a1 = (4,)
a3 = a.__add__(a1)
print(a3)

# 문제 9번
# 오류 발생 이유를 찾자
a = dict()
a
type(a)
a[[1]] = 'python'


# 문제 10번
# B값 추출
a = {'A':90, 'B':80, 'C':70}
result = a.pop('B')
print(a)
print(result)


# 문제 11번
# 중복 제거
a = [1,1,1,2,2,3,3,3,4,4,5]
x = set(a)
b = list(x)
print(b)


# 문제 12번
# 결과 이유 설명
a = b = [1,2,3]
a[1] = 4
print(b)
# 참조 주소가 서로 같다. 키만 서로 복사
# 서로 가져다 쓴다



# ﻿[ 제어문 실습과제 ] ===========================================

# 1 번
# 1-1
for x in range(1,101):
    if x%10 != 0:
        print(x, end =',')
    else :
    print(x)

# 1-2_일단 패스
l = list(range(1,101))
x = [a for a in l if l%10 != 0]
print(x, end =',')


# 2번
n = input('n=')
a = range(1, int(n)+1)
list = []
for b in a:
    list.append(b)
print('총합=',sum(list))


# 3번
n = input('n=')
a = range(1,int(n)+1)
list1=[]                            # 무슨 변수를 만들지 먼저 생각, 제어문 안에 들어가면 삭제되니까 그 밖에 둔다.
list2=[]

for b in a:
    if b%2==0 :
        list1.append(b)
    else :
        b%2!=0
        list2.append(b)
print('짝수의 합=',sum(list1))
print('홀수의 합=',sum(list2))

# 재민's 답
n = input('n=')
a = range(1,int(n)+1)

n = input('n=')
a = range(1,int(n)+1)

ak = 0
bk = 0

count = 0

for b in a:
    if b%2==0 :
        ak = ak + b
        count = count +1
    else :
        b%2!=0
        list2.append(b)
print('짝수의 합=',ak)
print('홀수의 합=',sum(list2))




# 4번
n = input('n=')
a = range(1,int(n)+1)
list = []
for b in a:
    if b%3 != 0 and b%5 != 0:
        list.append(b)
print(sum(list))


# 5번
# 5-1
for x in range(2,10):
    for y in range(1,10):
        print('{}*{}={} '.format(x,y,x*y))


# 6번
n = 0
count1 = 0
count2 = 0
count3 = 0
while n != -999:
    n = int(input('n='))
    if n<0:
        count1 = count1 + 1
    else :
        n>0
            if n%2==0:
                count2 = count2 + 1
            else :
                n%2!=0
                count3 = count3 + 1
print('음수의 개수=',count1)
print('양수 홀수의 개수=',count2)
print('양수 짝수의 개수=',count3)


# 문제 7번
dict = {1:'+', 2:'-', 3:'*', 4:'/'}
type(dict)
a = input('a=')
b = input('b=')
c = input(dict)
result = eval(a+c+b)
print(result)


a = input('a:')
b = input('b:')
list = ['+', '-', '*', '/']
op_select = int( input( 'Input operator( 1:+, 2:-, 3:*, 4:/ ) : ' ) )
c = list[op_select-1]      #생각하지 못했던 개념
result = eval(a+c+b)
print(result)



# ==================================강사님 답안================================================
# 제어문실습.py

# 1번
print('{0:=^50}'.format('1-1'))
for x in range(1, 101):
    print('{:4}'.format(x), end='')
    if x % 10 == 0:
        print()

print('{0:=^50}'.format('1-2'))
l = [x for x in range(1, 101)]
for x in l:
    print('{:4}'.format(x), end='')
    if x % 10 == 0:
        print()

# 2번
print('{0:=^50}'.format('2'))
max_number = int(input('Input max number : '))

total = 0
for x in range(1, max_number + 1):
    total = total + x

print('1 ~ {0:^6} = {1:<8}'.format(max_number, total))

# 3번
print('{0:=^50}'.format('3'))
max_number = int(input('Input max number : '))

even_list = []
odd_list = []
for x in range(1, max_number + 1):
    if x % 2 == 0:
        even_list.append(x)
    else:
        odd_list.append(x)

print('even number : ', even_list)
print('1 ~ {0:^6} = {1:<8d}\n'.format(max_number, sum(even_list)))

print('odd number : ', odd_list)
print('1 ~ {0:^6} = {1:<8d}'.format(max_number, sum(odd_list)))

# 4번
print('{0:=^50}'.format('4'))
max_number = int(input('Input max number : '))

Excluding_Multiple_of_3_5 = []

for x in range(1, max_number + 1):
    if x % 3 != 0 and x % 5 != 0:
        Excluding_Multiple_of_3_5.append(x)

print('Excluding Multiple of 3 and 5 : ', Excluding_Multiple_of_3_5)
print('sum = {0:<6}'.format(sum(Excluding_Multiple_of_3_5)))

# 5번
print('{0:=^50}'.format('5-1'))
for x in range(2, 10):
    for y in range(1, 10):
        print('{:3}'.format(x * y), end='')
    print()

print('{0:=^50}'.format('5-2'))
multiple_table = [x * y for x in range(2, 10) for y in range(1, 10)]

count = 0
for x in range(8 * 9):
    count = count + 1
    print('{:3}'.format(multiple_table[x]), end='')
    if count % 9 == 0:
        print()
        count = 0

print('{0:=^50}'.format('5-3'))
multiple_table2 = [x * y for x in range(2, 10) for y in range(1, 10)]

start = 0
for x in range(9, 81, 9):
    print('{0[0]:3}{0[1]:3}{0[2]:3}{0[3]:3}{0[4]:3}{0[5]:3}{0[6]:3}{0[7]:3}{0[8]:3}' \
          .format(multiple_table2[start:x]))
    start = start + 9

# 6번
print('{0:=^50}'.format('6'))
total_list = [0, 0, 0, 0]
total_title = ('positive', 'negative', 'even', 'odd')

while True:
    number = int(input('Input number : '))
    if number == -999:
        break

    if number != 0:
        if number > 0:
            total_list[0] = total_list[0] + 1
            if number % 2 == 0:
                total_list[2] = total_list[2] + 1
            else:
                total_list[3] = total_list[3] + 1
        else:
            total_list[1] = total_list[1] + 1
    else:
        print('error : input not {}'.format(number))

print()
for x in range(4):
    print('{0:<10} : {1:<5}'.format(total_title[x], total_list[x]))

# 7번
print('{0:=^50}'.format('7'))
op = {1: '+', 2: '-', 3: '*', 4: '/'}

while True:
    number1 = input('Input number1 : ')
    number2 = input('Input number2 : ')
    op_select = int(input('Input operator( 1:+, 2:-, 3:*, 4:/, 0:end ) : '))

    if op_select == 0:
        break;

    result = eval(number1 + op[op_select] + number2)

    print('number1 : {0:^8.2}'.format(number1))
    print('number2 : {0:^8.2}'.format(number2))
    print('{0:^6} {2:^3} {1:^6} = {3:<.2f}\n'.format(number1, number2, op[op_select], result))

# 8번
print('{0:=^50}'.format('8'))
from collections import namedtuple

Student = namedtuple('Student', 'name, subject1, subject2, subject3, total, average, grade')

student_list = []
MAX = 10
SUBJECT = 3
count = 0

name = input('Input name : ')
while name != 'end' and count < MAX:

    count = count + 1
    subject = []
    for x in range(SUBJECT):
        input_subject = int(input('Input subject' + str(x) + ':'))
        subject.append(input_subject)
    total = sum(subject)
    average = total / SUBJECT
    if average >= 90:
        grade = 'Excellent'
    elif average <= 50:
        grade = 'Fail'
    else:
        grade = ' '

    student = Student(name, subject[0], subject[1], subject[2], total, average, grade)
    student_list.append(student)

    name = input('Input name : ')

print()
for x in student_list:
    print('{0:<10} {1:<3} {2:<3} {3:<3} {4:<5} {5:6.2f} {6:<10}'. \
          format(x.name, x.subject1, x.subject2, x.subject3, x.total, x.average, x.grade))



# [함수 실습과제]====================================================================================
# 1. 두 개의 정수를 입력 받아 평균을 반환하는 함수를 작성 (첫 번째 수가 -1 이면 종료)

a = int(input('a= '))
b = int(input('b= '))
def mean_0(a,b):
    if a != -1:
        c = (a + b)/2
    return c
print(mean_0(a,b))


# 2. 입력 받은 내용을 리스트에 저장 후 리스트를 전달받아 최대값과 최소값을 반환하는 함수 작성
# (-1이 입력될 때 까지 입력 받아 리스트에 저장)

def worhkd_1():
    l=[]
    while True:
        n = int(input('입력 = '))
        if n == -1:
            break
        l.append(n)
        l.sort()
        print(l)
    return('최솟값 = ', l[0]),('최대값 = ', l[-1])
print(worhkd_1())                                 # 순서를 잘 생각하자

# 함수 이용 방법
def worhkd1():
    l = []
    while True:
        n = int(input('입력='))
        l.append(n)
        if n == -1:
            break
    return ('최솟값=',min(l)),('최대값=',max(l))
print(worhkd1())


# 3. 함수의 인자로 시작과 끝 숫자를 받아 시작부터 끝까지의 모든 정수값의 합을
# 반환하는 함수를 작성(시작값과 끝값을 포함).

a = int(input('시작값='))
b = int(input('끝값='))
l = []
def worhkd2():
    for c in range(a, b+1):
        l.append(c)
    d = sum(l)
    return d
print(worhkd2())


# 4. 함수의 인자로 문자열을 포함하는 리스트가 입력될 때 각 문자열의 첫 세 글자로만
# 구성된 리스트를반환하는 함수를 작성.
# 예를 들어, 함수의 입력으로 ['Seoul', 'Daegu', 'Kwangju', 'Jeju']가 입력
#    될 때 함수의 반환값은 ['Seo', 'Dae', 'Kwa', 'Jej']('end' 입력시 입력 종료)


dic = {'Seoul':'Seo' , 'Daegu' : 'Dae' , 'Kwangju' : 'Kwa' , 'Jeju': 'Jej'}
def rhkd1(dic):
    while True:
        n = input('도시를 입력하세요')
        print(dic[n])
        if n == 'end' :
            return '끝이 났습니다'
            break
    return dic[n]
print(rhkd1(dic))

# 4가지 유형을 공부

# 5. range() 함수 기능을 하는 myrange()  함수를 작성
# (인자가 1,2,3개인 경우를 모두구현  return 값은 튜플 )
#  (range() 함수를 사용해도 무방, 단 인자 처리 코드는 반드시 구현)


def myrange(*worhkd):
        if len(worhkd) == 1:
            range(0,worhkd[0])
            a = tuple([a for a in range(worhkd[0])])
            return a
        elif len(worhkd) == 2:
            range(worhkd[0],worhkd[-1])
            b = tuple([b for b in range(worhkd[0],worhkd[-1])])
            return b
        else:
            len(worhkd) == 3
            range(worhkd[0],worhkd[1],worhkd[2])
            c = tuple([c for c in range(worhkd[0],worhkd[1],worhkd[2])])
            return c
print(myrange(1,8,2))


# <고급>
# 6. 화면에 다음과 같은 메뉴를 출력하여 선택된
# 메뉴의 기능(두수를 입력받아 연산)을 수행하는 프로그램
#     1.add
#     2.subtract
#     3.multiply
#     4.divide
#     0.end
#     select :

worhkd1 = {1:'add', 2:'subtract',3:'multiply',4:'divide',0:'end'}
print(worhkd1)

def worhkd0():
    worhkd1 = {1: 'add', 2: 'subtract', 3: 'multiply', 4: 'divide', 0: 'end'}
    while True:
        print(worhkd1)
        a = int(input('숫자 입력 = '))
        b = int(input('숫자 입력 = '))
        worhkd2 = {1: a + b, 2: a - b, 3: a * b, 4: a / b, 0: 'end'}
        x = worhkd2[int(input('연산자'))]
        print(x)

print(worhkd0())




# <기본>
# 1. 두 개의 정수를 입력 받아 평균을 반환하는 함수를 작성
# (첫 번째 수가 -1 이면 종료)

a = int(input('a = '))
b = int(input('b = '))

def worhkdTm(a,b):
    c = (a+b)/2
    return c
print(worhkdTm(a,b))

# 2. 입력 받은 내용을 리스트에 저장 후 리스트를 전달받아
# 최대값과 최소값을 반환하는 함수 작성
# (-1이 입력될 때 까지 입력 받아 리스트에 저장)

# 굳이 for 문이 필요없다면 쓰지 말자

def rhkdtm():
    l = []
    while True:
        n = int(input('숫자를 입력하세요'))
        if n == -1:
            break
        l.append(n)
        l.sort()
    return ('최소값 = ', l[0], '최대값 = ', l[-1])
print(rhkdtm())


# 3. 함수의 인자로 시작과 끝 숫자를 받아 시작부터 끝까지의
# 모든 정수값의 합을 반환하는 함수를
# 작성(시작값과 끝값을 포함).  (시작값이 끝값보다 클때 입력 종료)

a = int(input('시작 = '))
b = int(input('끝 = '))
l = []
def rhkddl():
    for c in range(a, b+1):
        l.append(c)
    d = sum(l)
    return d
print (rhkddl())


# 4. 함수의 인자로 문자열을 포함하는 리스트가 입력될 때
# 각 문자열의 첫 세 글자로만
# 구성된 리스트를반환하는 함수를 작성.
# 예를 들어, 함수의 입력으로 ['Seoul', 'Daegu', 'Kwangju', 'Jeju']가 입력
#    될 때 함수의 반환값은 ['Seo', 'Dae', 'Kwa', 'Jej']('end' 입력시 입력 종료)


dic = {'Seoul':'Seo' , 'Daegu' : 'Dae' , 'Kwangju' : 'Kwa' , 'Jeju': 'Jej'}
def goqhwk(dic):
    while True:
        n = input('도시를 입력하세요.')
        print(dic[n])
        if n == 'end':
            return('종료합니다.')
            break
    return dic[n]
print(goqhwk(dic))

# 1. 키와 몸무게를 입력받아 비만도를 구하고 결과를 출력하시요(함수를 만드시요)
# 표준체중(kg)=(신장(cm)-100)×0.85
# 비만도(%)=현재체중/표준체중(%)×100

# 비만도가90%이하 -->저체중
# 90초과~110%     --> 정상
# 110초과~120%   --> 과체중
# 120%초과         -->  비만

a = int(input('키 = '))
b = int(input('몸무게 = '))

def bimando(a,b):
    while True:
        a = int(input('키 = '))
        b = int(input('몸무게 = '))
        l = (a-100)*0.85
        x = b/l*100
        if x <= 90:
            print('당신은 저체중입니다.')
        elif 90 < x <=110:
            print('당신은 정상입니다.')
        elif 110 < x <= 120:
            print('당신은 과체중입니다.')
        else:
            x>120
            print('당신은 비만입니다.')
        continue
        return x
print(bimando(a,b))


# 2. 연도를 입력받아
# 1) 윤년여부를 출력하시요(함수를 만드시요)
# 윤년의 조건
# 1-1) 4로 나눠 떨어지지만 100으로 나눠 떨어지지 않아야 한다 또는
# 1-2) 400 으로 나눠 떨어지면 윤년임
# 2) 나이를 출력하시요(함수를 만드시요)
# 3) 띠(12지신)를 출력하시요(함수를 만드시요)
#    ("쥐","소","호랑이","토끼","용","뱀","말","양","원숭이","닭","개","돼지",);
#    (서기 4년은 쥐띠이다,2019년 돼지)


def dbssus():
    while True:
        n = int(input('연도를 입력하시오'))
        if n%4 == 0 and n%100 != 0:
            print('윤년입니다.')
        elif n%400 == 0 :
            print('윤년입니다.')
        else :
            print('윤년이 아닙니다.')
        continue


dbssus()


def El():
    while True:
        n = int(input('연도를 입력하시오'))
        x = 2019 - n + 1
        print('당신의 나이는 {}입니다.'.format(x))
El()



# 풀이 1번
def Elsms():
    while True:
        n = int(input('연도를 입력하시오'))
        if n%12 == 0:
            print('원숭이띠')
        elif n%12 == 1:
            print('닭띠')
        elif n%12 == 2:
            print('개띠')
        elif n%12 == 3:
            print('돼지띠')
        elif n%12 == 4:
            print('쥐띠')
        elif n%12 == 5:
            print('소띠')
        elif n%12 == 6:
            print('호랑이띠')
        elif n%12 == 7:
            print('토끼띠')
        elif n%12 == 8:
            print('용띠')
        elif n%12 == 9:
            print('뱀띠')
        elif n%12 == 10:
            print('말띠')
        else:
            n%12 == 11
            print('양띠')
        continue

Elsms()

# 풀이 2번
def ektlgoqha():
    a = ["원숭이", "닭", "개", "돼지", "쥐", "소", "호랑이", "토끼", "용", "뱀", "말", "양"]
    while True:
        n = int(input('연도를 입력하세요'))
        x = a[n%12]
        print(x)
    return x
print(ektlgoqha())

# 3. 점수를 입력받아
# 90~100 'A'
# 80~89 'B'
# 70~79 'C'
# 60~69 'D'
# 나머지 'F'
# 딕셔너리를 이용하여 구하시요(함수를 만드시요)


def wjatn():
    dic = {range(90, 101): 'A', range(80, 90): 'B', range(70, 80): 'C',
           range(60, 70): 'D', range(0, 60): 'F'}
    while True:
        x = int(input('점수를 입력하세요 = '))
        if 90<= x <= 100:
            print([k for x,k in dic.items() if x == range(90,101)])
        elif 80<= x <= 89:
            print([k for x,k in dic.items() if x == range(80, 90)])
        elif 70 <= x <= 79:
            print([k for x,k in dic.items() if x == range(70, 80)])
        elif 60 <= x <= 69:
            print([k for x,k in dic.items() if x == range(60, 70)])
        else :
            x < 60
            print([k for x, k in dic.items() if x == range(0, 60)])

wjatn()


# srp={'가위':'보','바위':'가위','보':'바위'}
# print([k for k,v in srp.items() if v == '바위'].pop())



# 4. m(미터) 를 입력받아 마일로 변환하시요(함수를 만드시요)
#    (1 mile =  1.609 meter)

def qusghks():
    while True :
        n = int(input('m를 입력하세요'))
        x = n/1.609
        print('{} mile 입니다.'.format(x))

print(qusghks())


# 5. 화씨 를 입력받아 섭씨로 변환하시요(함수를 만드시요)
#    (celsius = ( fahrenheit - 32 ) / 1.8)

def ghkTl():
    while True:
        n = int(input('화씨를 입력하세요'))
        x = (n-32)/1.8
        print('{} 섭씨 입니다.'.format(x))

print(ghkTl())


# 6. 하나의 정수를 입력받아 약수를 구하는 함수를 만드시요.
#    (어떤 정수 n을 자연수 k로 나누어 나머지가 0 일경우 k는 정수 n의 약수이다)

def wjdtn():
    n = int(input('정수를 입력하세요'))
    l = []
    for k in range(1,n+1):
        if n%k == 0:
            l.append(k)
    print('{}는 {}의 약수입니다'.format(l,n))

print(wjdtn())


# 7. 2개의 정수를 입력받아 절대값의 합을 구하는 함수를 만드시요
#    ( abs()합수를 사용하지 않고 구현한다)

def wjfeorkqt():
    a = int(input('a ='))
    b = int(input('b ='))
    if a < 0 :
        a = -a
    if  b < 0 :
        b = -b
    x = a+b
    return(x)

print(wjfeorkqt())


# 8. map 함수와 동일한 기능을 하는 mymap 함수를 구현하시요
#
# <map()함수의 기능>
# def multi_two(x):
#     return x*2
# result = map(multi_two,[1,2,3,4,5])
# print(list(result))
# 출력 결과 :[2, 4, 6, 8, 10]

# ===================강사 답안==========================================
# 함수추가과제문제.py

# 1번
print('{0:=^50}'.format(' 1번 '))
def calc_fat_ratio(height,weight):
    std_weight = (height  - 100) * 0.85
    fat_ratio = (weight/std_weight)*100
    if fat_ratio <= 90:
        fat_grade = '저체중'
    elif fat_ratio > 90 and fat_ratio <= 110:
        fat_grade = '정상'
    elif fat_ratio > 110 and fat_ratio <= 120:
        fat_grade = '과체중'
    else:
        fat_grade = '비만'
    return fat_ratio,fat_grade

def get_health_info():
    height = int(input('Your Height(cm):'))
    weight = int(input('Your Weight(kg):'))
    fat_ratio,fat_grade = calc_fat_ratio(height,weight)
    print('키 : {0:3}cm  몸무게: {1:3}kg'.format(height,weight))
    print('비만도 : {0:>5.1f}% ({1})'.format(fat_ratio,fat_grade))

# get_health_info()

# 2번
def get_leap_year(year):

    if (year % 4 == 0 and year % 100 !=0) \
            or (year % 400 == 0):
        return '윤년'
    return '평년'

def get_age(year,current_year):
    age = current_year - year + 1
    return age

# 12지
# 子(자/쥐) 丑(축/소) 寅(인/호랑이) 卯(묘/토끼)
# 辰(진/용) 巳(사/뱀) 午(오/말) 未(미/양)
# 申(신/원숭이) 酉(유/닭) 戌(술/개) 亥(해/돼지).
# 서기 4년  : 子(자/쥐)

def get_12_animals(year):
    animals = ['子(자/쥐)',  '丑(축/소)', '寅(인/호랑이)',
               '卯(묘/토끼)','辰(진/용)', '巳(사/뱀)',
               '午(오/말)',  '未(미/양)', '申(신/원숭이)',
               '酉(유/닭)',  '戌(술/개)', '亥(해/돼지)']
    idx = (year - 4)%12
    return animals[idx]

def get_year_info():
    while True:
        print('-'*30)
        current_year = 2019
        year = int(input('year(0 to quit):'))
        if year == 0 : break
        if year < 0 : continue
        print(year,'year :',get_leap_year(year))
        print('age :', get_age(year,current_year))
        print('animal :', get_12_animals(year))

 # get_year_info()


# 3번
def get_grade(score):
    d = {'90~100':'A','80~89':'B','70~79':'C',
         '60~69':'D','0~59':'F'}
    if score >=90 and score <= 100:
        grade = d['90~100']
    elif score >=80 and score < 90:
        grade = d['80~89']
    elif score >=70 and score < 80:
        grade = d['70~79']
    elif score >=60 and score < 70:
        grade =  d['60~69']
    else:
        grade = d['0~59']
    return grade

def get_score_grade():
    while True:
        score = int(input('score(-1 to quit)='))
        if score < 0 : break
        print(score,':',get_grade(score))
# get_score_grade()

# 4번
def get_mile(meter):
    if meter < 0 :
        return
    mile = meter / 1.609
    return mile
def input_meter_to_mile():
    while True:
        meter = float(input('meter(-1 to quit)='))
        if meter < 0 : break
        print(meter,'meter:{:6.2f}'.format(get_mile(meter)),'miles')

# input_meter_to_mile()

# 5번
def get_celsius(fahrenheit):
    celsius = ( fahrenheit - 32 ) / 1.8
    return celsius

def input_fahrenheit_to_celsius():
    fahrenheit = float( input( 'Input fahrenheit : ' ) )
    celsius = get_celsius(fahrenheit)
    print( 'fahrenheit : {0:<6.2f} -> celsius : {1:<6.2f}'.format( fahrenheit, celsius ) )

# input_fahrenheit_to_celsius()

# 6번
def get_divisor(number):
    result = []
    for k in range(1,number + 1):
        remain = number % k
        if remain == 0:
            result.append(k)
    return result

def input_number_for_divisor():
    while True:
        number = int(input('number(0 to quit)='))
        if number == 0 : break
        print(number,':',get_divisor(number),
              len(get_divisor(number)),'개')

# input_number_for_divisor()

# 7번
def sum_abs(a,b):
    if a < 0 :
        a = a * -1
    if b < 0 :
        b = b * -1
    return a + b

def input_number_to_sum_abs():
    while True:
        number1 = int(input('number1(0 to quit)='))
        if number1 == 0 : break
        number2 = int(input('number20 to quit)='))
        if number2 == 0 : break
        print(number1,'and',number2,'==>',sum_abs(number1,number2))

# input_number_to_sum_abs()

# 8번
def mymap(func,var_list):
    result_list = []
    for k in var_list:
        result_list.append(func(k))
    return result_list

def multi_two(x):
    return x*2

# print(mymap(multi_two,[1,2,3,4,5,6]))



# ===============================================================================
# 1. Car class를 만들고 다음 멤버와 메서드를 구현하고
#    호출하는 코드를 구현해보세요
#    클래스의 인스턴스 객체  sonata 를 만든다
#    클래스의 모든 메서드를 호출해서 동작을 확인해본다

class Car:
    def __init__(self,name,drv,speed,direction,fuel,state):
        self.car_name = name
        self.car_drv = drv
        self.car_speed = speed
        self.car_direction = direction
        self.car_fuel = fuel
        self.car_state = state

    def set_car_name(self,name):
        self.name = name
        print('차종이 [{}]로 변경 되었습니다'.format(name))
    def get_car_name(self):
        return car_name
    def set_car_drv(self):
        self.drv = drv
        print('차의 구동방식이 {}으로 바뀌었습니다.'.format(drv))
    def get_car_drv(self):
        return drv
    def set_car_fuel(self,fuel):
        self.fuel = fuel
        print("차의 연료 방식이 [ 전기 ]로 변경 되었습니다")
    def get_car_fuel(self):
        return fuel



# ===========================================================================
class Car:

    def __init__(self,
                 name = 'sonata',
                 drv = '전륜',
                 speed = 0,
                 direction = '앞쪽',
                 fuel = '휘발유',
                 state = '정상'):

        print('생성자 호출')
        self.car_name = name
        self.car_drv = drv
        self.car_speed = speed
        self.car_direction = direction
        self.car_fuel = fuel
        self.car_state = state

    def set_car_name(self, name):
        self.car_name = name
        print('{0} [{1}]{2}'.format("차종이", name, "으(로) 변경 되었습니다"))

    def get_car_name(self):
        return self.car_name

    def set_car_drv(self, drv):
        self.car_drv = drv
        print('{0} [{1}]{2}'.format("차의 구동 방식이", drv, "으(로) 변경 되었습니다"))

    def get_car_drv(self):
        return self.car_drv

    def set_car_fuel(self, fuel):
        self.car_fuel = fuel
        print('{0} [{1}]{2}'.format("차의 연료 방식이", fuel, "으(로) 변경 되었습니다"))

    def get_car_fuel(self):
        return self.car_fuel

    def set_car_state(self, state):
        self.car_state = state
        print('{0} [{1}]{2}'.format("차의 상태가", state, "으(로) 변경 되었습니다"))

    def get_car_state(self):
        return self.car_state

    def set_car_speed(self, speed):
        self.car_speed = speed
        print('{0} [{1}]{2}'.format("차의 속력이 시속", speed, "km 로 변경 되었습니다"))

    def get_car_speed(self):
        return self.car_speed

    def turn(self, direction):
        self.car_direction = direction
        print("자동차의 방향이 ", direction, "으로 변경되었습니다.")

    def stop(self):
        self.car_direction = '[ 정지 ]'
        print("자동차가 정지 하였습니다.")
        return self.car_direction

    def start(self):
        self.car_direction = '[ 시동 ]'
        print("자동차가 시동이 걸렸습니다.")

    def move_forward(self, speed):
        self.car_speed = speed
        self.car_direction = '앞쪽'
        direction = '전진'
        print('{0} [{1}]{2} [{3}]{4}'.format("자동차가", direction, "합니다. 속도는", speed, "km 입니다."))

    def move_backward(self, speed):
        self.car_speed = speed
        self.car_direction = '뒤쪽'
        direction = '후진'
        print('{0} [{1}]{2} [{3}]{4}'.format("자동차가", direction, "합니다. 속도는", speed, "km 입니다."))

    def __del__(self):
        print(self.car_name, '자동차가 제거되었습니다.')



# =============================================================================
class Car:

    def __init__(self,
                 name='소나타',
                 drv='전륜',
                 speed=0,
                 direction='앞쪽',
                 fuel='휘발유',
                 state='정상'):
        print('생성자 호출')
        self.car_name = name
        self.car_drv = drv
        self.car_speed = speed
        self.car_direction = direction
        self.car_fuel = fuel
        self.car_state = state

    def set_car_name(self,name):         # set 입력
        self.car_name = name
        print('차종이 [{}]으로 변경되었습니다.'.format(name))


    def get_car_name(self,name):         # 값을 받을 때는 인자를 넣을 필요가 없다
        return self.car_name

    def set_car_drv(self,drv):
        self.car_drv = drv
        print('차의 구동방식이 {}으로 변경되었습니다.'.format(drv))

    def get_car_drv(self):
        return self.car_drv

    def set_car_fuel(self,fuel):
        self.car_fuel = fuel
        print("차의 연료 방식이 [ {} ]로 변경 되었습니다".format(fuel))

    def get_car_fuel(self,fuel):
        return self.car_fuel

    def set_car_state(self,state):
        self.car_state = state
        print("차의 상태가 [ {} ]으로 변경 되었습니다".format(state))

    def get_car_state(self,state):
        return self.car_state

    def set_speed(self,speed):
        self.car_speed = speed
        print(" 자동차의 속력이 시속 [{}] km 로 변경되었습니다".format(speed))

    def get_speed(self,speed):
        return self.car_speed

    def turn(self,direction):
        self.car_direction = direction
        print(" 자동차의 방향이  [ {} ]으로 변경되었습니다".format(direction))

    def stop(self):
        self.car_direction = '[ 정지 ]'
        print("자동차가 정지하였습니다.")
        return self.car_direction

    def start(self):
        self.car_direction = '[시동]'
        print('자동차가 시동이 걸렸습니다.')
        return self.car_direction

    def move_forward(self,direction,speed):
        self.car_direction = '앞쪽'
        self.car_speed = speed
        print('자동차가 {}합니다. 속도는 {}입니다.'.format(direction,speed))

    def move_backward(self,direction, speed):
        self.car_direction = '뒤쪽'
        self.car_speed = speed
        print('자동차가 {}합니다. 속도는 {}입니다.'.format(direction,speed))

    def  __del__(self,name):
        self.car_name = name
        print(self.car_name, '{} 자동차가 제거되었습니다.'.format(name))




# ====================================2번<강사님 답안>======================================

# 2번

class CarCenter:
    price = {'정상': 10, '브레이크고장': 1000, '전조등고장': 2000, '후미등고장': 3000, '연료부족': 4000,
            '타이어펑크': 5000, '엔진오일부족': 6000, '냉각수부족': 7000, '폐차처리': 9000}

    def __init__(self):
        self.fix_cost = 0
        self.fixed_list = {}
        self.accent = Car()          # 이번 과제 핵심

    def fix_car(self,car):

        self.fix_cost = CarCenter.price[car.car_state]
        self.fixed_list[car.car_name] = car.car_state
        print('[',car.car_name,']의 [',car.car_state,
              '] 수리 완료, 비용은 [',self.fix_cost,'] 원 입니다')

    def set_car_drv(self,car, drv):
        car.car_drv = drv
        self.accent.car_drv = drv
        print("차의 구동 방식이 [", car.car_drv ,"]으로 변경 되었습니다")

    def get_car_drv(self,car):
        return car.car_drv

    def set_car_fuel(self,car,fuel):
        car.car_fuel = fuel
        print("차의 연료 방식이 [", car.car_fuel,"]로 변경 되었습니다")

    def get_car_fuel(self,car):
        return car.car_fuel

    def get_fixed_list(self,car):
        fixed_item = self.fixed_list[car.car_name]
        cost = CarCenter.price[fixed_item]
        return '[' + fixed_item + '] : [' + str(cost) + ']원'

    def __del__(self):
        pass

def test_carcenter(car):
    sonata = car

    ct1 = CarCenter()

    ct1.fix_car(sonata)

    ct1.set_car_drv(sonata,'후륜')
    print(ct1.get_car_drv(sonata))

    ct1.set_car_fuel(sonata, '전기')
    print(ct1.get_car_fuel(sonata))

    print(ct1.get_fixed_list(sonata))


# test_carcenter(sonata)

# 별도의 파일로 작성한다
import 클래스기초실습문제 as car

avante = car.Car()
avante.set_car_name('아반테')
print(avante.get_car_name())
avante.set_car_state('전조등고장')
print(avante.car_state)

print('-'*30)
ct1 = car.CarCenter()
ct1.fix_car(avante)
ct1.set_car_drv(avante, '후륜')
print(ct1.get_car_drv(avante))

ct1.set_car_fuel(avante, '수소')
print(ct1.get_car_fuel(avante))
print(ct1.get_fixed_list(avante))


sorento = car.Car()
sorento.set_car_name('소렌토')
sorento.set_car_state('타이어펑크')
ct1.fix_car(sorento)
print(ct1.get_fixed_list(sorento))

pride = car.Car()
pride.set_car_name('프라이드')
pride.set_car_state('엔진오일부족')
ct1.fix_car(pride)
print(ct1.get_fixed_list(pride))

pride.set_car_state('타이어펑크')
ct1.fix_car(pride)
print(ct1.get_fixed_list(pride))

print(ct1.fixed_list)
# {'아반테': '전조등고장', '소렌토': '타이어펑크',
# '프라이드': '엔진오일부족'}



# ====================1. 계산기 프로그램===============================

class Calculator():
    def calculate(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
        x = { 1:'+', 2:'-', 3:'*', 4:'/' }
        result = eval(str(a) + x[c] + str(b))
        return result

a = Calculator()
a.calculate(1,2,3)

#  eval 문자열을 정수형으로, int는 str로 바꾸고 시작
# () 쓰기


class ControlCalculator:
    def __init__(self):
        self.rhkd = Calculator()
# 몰랐던 개념, 변수는 할당 x 클래스 인스턴트 객체를 멤버로 가져온다는 것은
# 그 클래스를 가져와서 내가 새로 만든 변수에 입력한다는 뜻.
# 그럼 나는 각각의 다른 객체의 성격을 갖을 수 있게 된다.
    def calculate(self):
        Calculator.calculate()


class ViewCalculator:
    def __init__(self):
        self.rhkdTm = ControlCalculator()
    def DisplayCalculator(self):
        while True :
            n = int(input('숫자를 입력해용'))
            continue
        ControlCalculator.calculate(n)
    print(DisplayCalculator(n))

# ====================1. 강사님 답안===============================

# ====================2. 학생 클래스===============================

class Student:
    def calculate(self,q,w,e):
        self.q = q
        self.w = w
        self.e = e
        sum_hong = q+w+e
        sum_kim = q+w+e
        sum_nam = q+w+e
        return sum_hong, sum_kim, sum_nam


# ====================================================================
class Airplane(Car):
    def __init__(self,
                 name ='KAL147',
                 height = 0,
                 speed = 0,
                 direction = '정지',
                 state = '정상'):
        self.air_name = name
        self.air_height = height
        self.air_speed = speed
        self.air_direction = direction
        self.air_state = state

    def set_air_name(self,name):
        self.air_name = name
        print('기종이 {} 으로 변경되었습니다.'.format(self.air_name))

    def get_air_name(self):
        return self.air_name

    def set_air_height(self,height):
        self.air_height = height
        print('비행 고도를 {} km 로 설정하였습니다'.format(self.air_height))

    def get_air_height(self):
        return self.air_height

    def land_to_ground(self,direction):
        self.air_direction = direction
        print('비행기가 {}하였습니다'.format(self.air_direction))

    def set_speed(self,speed):
        self.air_speed = speed
        print (" 비행기의 속력이 시속 {} km 로 변경되었습니다".format(self.air_speed))

    def get_speed(self):
        return self.air_speed

    def move_forward(self,direction,speed):
        self.air_speed = speed
        self.air_direction = direction
        print("비행기가 {}으로 전진합니다 속도는 {}km입니다".format(self.air_direction, self.air_speed))

    def move_backward(self,direction,speed):
        self.air_speed = speed
        self.air_direction = direction
        print("비행기가 {}으로 후진합니다 속도는 {} km 입니다".format(self.air_direction, self.air_speed ))



a = Airplane()
print(a.get_speed())
print(a.set_speed(200))
print(a.get_speed())
print(a.land_to_ground('착륙'))
print(a.move_forward('앞쪽',151))

# 초기값을 설정한 변수명으로 self.변수명을 동일하게 설정
# print 값에도 변수명을 입력할 것.


# def move_backward(self, direction, speed):
# #     self.air_speed = speed

# 위의 self.air_speed = speed 에서
# 앞의 변수명은 초기값 설정과 같이 해줘야 하고,
# 뒤의 speed는 어떠한 값이 아니라 입력될 인자를 말해준다. 인자값을 받겠다
# 맨 위의 speed는 인자로 입력될 값의 위치를 말한다.

    # def 메소드 변수 (self, 인자값1, 인자값2):
    #     self. <설정된 변수1> = 인자값1
    #     self. <설정된 변수2> = 인자값2

#  메타몽은 어떤 설정된 변수에서 입력될 인자값을 받겠습니다.


# =============================강사님 답안=============================================================

# 클래스상속실습과제.py

class Car:

    def __init__(self):
        print('Car 생성자')
        self.car_name = '소나타'
        self.car_drv = '전륜'
        self.car_speed = 0
        self.car_direction = '앞쪽'
        self.car_fuel = '휘발유'
        self.car_state = '정상'

    def set_car_name(self, name):
        self.car_name = name
        print("차종이 [",self.car_name,"]으로 변경 되었습니다")

    def get_car_name(self):
        return self.car_name

    def set_car_drv(self, drv):
        self.car_drv = drv
        print("차의 구동 방식이 [", self.car_drv ,"]으로 변경 되었습니다")

    def get_car_drv(self):
        return self.car_drv

    def set_car_fuel(self, fuel):
        self.car_fuel = fuel
        print("차의 연료 방식이 [", self.car_fuel,"]로 변경 되었습니다")

    def get_car_fuel(self):
        return self.car_fuel

    def set_car_state(self,state):
        self.car_state = state
        print("차의 상태가 [",self.car_state, "]으로 변경 되었습니다")

    def get_car_state(self):
        return self.car_state

    def set_speed(self,speed):
        self.car_speed = speed
        print("자동차의 속력이 시속 [",self.car_speed,"]km 로 변경되었습니다")

    def get_speed(self):
        return self.car_speed

    def turn(self,direction):
        self.car_direction = direction
        print("자동차의 방향이 [",self.car_direction ,"]으로 변경되었습니다")

    def stop(self):
        self.car_direction = '정지'
        print("자동차가 정지 하였습니다")

    def start(self):
        print("자동차가 시동이 걸렸습니다")

    def move_forward(self):
        self.car_direction = '앞쪽'
        print("자동차가 전진합니다 속도는 ",self.car_speed,"km입니다")

    def move_backward(self):
        self.car_direction = '뒤쪽'
        print("자동차가 후진합니다 속도는 ",self.car_speed,"km입니다")

    def __del__(self):
        print('[', self.car_name, "] 자동차가 제거되었습니다")


class Airplane(Car):

    def __init__(self):
        print('Airplane 생성자')

        # < 추가 인스턴스 멤버 >
        self.air_name = 'KAL147'
        self.air_height =  0
        self.air_speed = 0
        self.air_direction = '정지'
        self.air_state = '정상'

        self.car = Car()

    # < 추가 메서드 >
    def set_air_name(self, name):
        self.air_name = name
        print('비행기 기종이 [', self.air_name, ']로 변경 되었습니다')

    def get_air_name(self):
        return self.air_name

    def set_height(self,height):
        self.air_height = height
        print('비행 고도를 [',self.air_height,'] km 로 설정하였습니다')

    def get_height(self):
        return self.air_height

    def land_to_ground(self):
        self.air_direction = '정지'
        print('비행기가 착륙하였습니다')

    # < 메서드 오버라이딩구현 >
    def set_speed(self,speed):
        self.air_speed = speed
        print('비행기의 속력이 시속 [',self.air_speed,'] km 로 변경되었습니다')

    def get_speed(self):
        return self.air_speed

    def move_forward(self):
        self.air_direction = '앞쪽'
        print('비행기가 전진합니다 속도는 [',self.air_speed,
        ']km입니다')

    def move_backward(self):
        self.air_direction = '뒤쪽'
        print('비행기가 후진합니다 속도는 [',self.air_speed,
        '] km 입니다')

    def __del__(self):
        print('[', self.air_name, "] 비행기가 제거되었습니다")


if __name__ == '__main__':
    kal = Airplane()

    kal.set_air_name('아시아나104')
    print(kal.get_air_name())

    kal.set_height(1000)
    print(kal.get_height())

    kal.land_to_ground()

    kal.set_speed(100)
    print(kal.get_speed())

    kal.move_forward()
    kal.move_backward()

    print(kal.car.car_name)







# =======================파일 실습 문제========================================

f = open('items.txt','w')
f.write('품목명,단가')
f.close()

f = open('items.txt','r')
f.close()

