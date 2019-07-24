# 04-1 함수

# 모듈하고 함수
'''
def 함수명 (매개변수):
    <수행할 문장1>
    <수행할 문장2>
    ....
    return 반환할 값
'''
# =인자, 아규먼트
# 함수 객체를 만들 뿐, 함수를 호출해야 한다

# 1회 덧셈
a = 10
b = 10
c= a + b
print('a+b = %d'%c)

# 10회 덧셈을 반복
a = [1,2,3,4,5,6,7,8,9,10]
b = [10,20,30,40,50,60,70,80,90,100]
for k in range(len(a)):
    print(k, ':', a[k] + b[k])

# 함수 구현

# add() 함수
def add(a,b):            # 함수 정의
    c = a + b
    print('add called')
    return c

result = add(10,20)               # 함수 호출
print(result)
print('종료')

'''
# A() -> B()
# stepout 은 다시 A()로 돌아간다
# stepover 는 건너띄기
# stepinto 는 실행하기
'''


# subtract() 함수 정의
def subtract(x,y):
    c = x-y
    print('subtract')
    return c
result = subtract(20,10)
print(result)

# multiply() 함수 정의
def multiply(x,y):
    c = x * y
    print('multiply')
    return c
result = multiply(20,10)
print(result)

# divide() 함수 정의
def divide(x,y):
    c = x/y
    print('divide')
    return c


# 함수 호출
result = add(10,20)
print(result)

result = subtract(20,10)
print(result)

result = multiply(20,10)
print(result)

result = divide(20,10)
print(result)

# 함수내에서 함수 호출하는 함수 구현

# myfunc(a,b,c,d) = ((a+b)*(c-d)/e


# =================================================================
# 1. 함수내에서 직접 연산

def myfunc1(a,b,c,d,e):
    if e == 0 :
        return "error"
    f = ((a + b)*(c - d))/e
    return f
result = myfunc1(1,1,1,1,1)
print(result)

# 2. add/subtract/multiply/divide 함수 사용, 오픈 소스, 다른 사람의 함수를 가져와 내 함수로 만들 수 있다.

def myfunc2(a,b,c,d,e):
    if e == 0 :
        return "error"
    # f = ((a + b) * (c - d)) / e
    f = divide(multiply(add(a,b),subtract(c,d)),e)
    return f
result = myfunc2(1,1,1,1,1)
print(result)

# ====================================================================

# def not used(): , 주석대신에 쓸 수 있다. 함수를 부르지 않는 이상 실행되지 않으니까
# 실행하고 싶으면 마지막에 not used() 써주기
# 어떤 코드든지 def 쓰고 이름 쓰고 들여쓰기

# 함수 구현 순서
# (1) def 를 사용하고 함수이름을 결정
# (2) 매개변수(인자)를 결정
# (3) 인자를 사용하여 처리하는 코드를 구현
# (4) return을 사용하여 결과값을 반환
# (5) 인자를 설정하여 함수를 호출하여 결과를 확인

# 함수 유형 4 가지
#  [1] 반환값이 없고  매개변수(인자,인수) 없고
def f_1():
    print('f_1 is called')
f_1()

#  [2] 반환값이 없고  매개변수(인자,인수) 있고
def f_2(a,b):
    print('f_2 :" ,a,b,+b')
f_2(1,1)

def f_22(a1,a2,a3 = 'None'): # 'none', 기본 인자
    print('f_22', a1,a2,a3)
f_22(10,20)
f_22(10,00,30)
f_22('hello',(1,2,3,4), [5,6,7])

#  [3] 반환값이 있고  매개변수(인자,인수) 없고
def f_3():
    print('f_3 is called')
    return 'bread','butter'     #괄호가 생략된 튜플
print(f_3())

#  [4] 반환값이 있고  매개변수(인자,인수) 있고
def f_4(a,b):
    print('f_4 is called')
    return a + b, a - b, a * b, a/b
print(f_4(3,5))

# 문제 : 2개의 정수에 대해 큰 수, 작은 수의 순서로 반환하는
#       함수를 만드세요 <함수 이름은 order()로>
def order(x,y):
    if x < y:
        print(y,x)
    return x,y

print(order(6,10))
print(order(10,9))

# ===========교수 답안=================
def order(a,b):
    if a < b :
        a,b = b,a
    return a,b

def order(a,b):
#===========문제풀기=================
# max(a,b) 함수
def max(a,b):
    if a < b:
        print(b)
    else:
        a>b
        print(a)
    return a,b
print(max(12,10))
# ============ 강 사 ===================
def max(a,b):
    if a > b :
        return a
    return b

# min(a,b) 함수
def min(a,b):
    if a > b:
        print(b)
    else:
        a<b
        print(a)
    return a,b
print(min(16,10))
# ============= 강 사 ===============
def min(a,b):
    if a < b :
        return a
    return b

# sum(리스트) 함수
def sum(a,b,c,d):
    c = (a+b+c+d)
    return c
print(sum(1,2,3,4))
# ============== 강 사 ===============
def sum(l):
    total = 0
    for k in l :
        total = total + k
    return total
print(sum([1,2,3,4,5,6,7,8,9,10]))
print(sum(list(range(100))))


# mean(리스트) 함수
def mean(a,b,c,d):
    c = (a+b+c+d)/4
    return c
print(mean(1,2,3,4))
# ============== 강 사 ===============
def mean(l):
    total = 0
    for k in l :
        total = total + k
    return total / len(l)

def mean(l):
    return sum(l)/len(l)

print(mean([1,2,3,4,5,6,7,8,9,10]))
print(mean(list(range(100))))

# 인수의 갯수가 고정되지 않은 인수 처리 방법
def add_many(a,b):
    c = a + b
    return c
print(add_many(10,20))

def add_many(*args):                  # * => '나머지 전체 다' 라는 뜻
    total = 0
    for k in args:
        total = total + k
    return total
print(add_many(10,20,32,41,3,123,12))

# 키워드 인수(사전 사용)

#  리스트 출력
def printList(title,numberPerLine,prnList):
    count = 0
    print(title)
    for k in prnList:
        count = count + 1
        print('{0:<10}'.format(k), end='')
        if count % numberPerLine == 0:
            print()

title = '========성적테이블=========='
l = [0,1,2,3,4,5,6,7,8,9,10,11]
number = 4
printList(title,number,l)



def printList_1(title,numberPerLine,prnList):
    count = 0
    print(title)
    for k in prnList:
        count = count + 1
        print('{0:<10}'.format(k), end='')
        if count % numberPerLine == 0:
            print()
title = '========성적테이블=========='
mylist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
printList('mynumber', 4, mylist)


def add_many(*args):  # * => '나머지 전체 다' 라는 뜻
total = 0
for k in args:
total = total + k
return total

print(add_many(10, 20, 32, 41, 3, 123, 12))


print( '{0:=^50}'.format( "함수구현 4유형"))
# 함수 유형 4가지
print( '{0:-^50}'.format( "[1] 반환값   X      매개변수(인자/인수)   X"))
# [1] 반환값   X      매개변수(인자/인수)   X
def f_1():
    print('f_1 is called')
f_1()


print( '{0:-^50}'.format( "[2] 반환값   X      매개변수(인자/인수)   O"))
# [2] 반환값   X      매개변수(인자/인수)   O
def f_2(a,b):
    print('f_2 - 인자 a: {0} , 인자 b : {1}'.format(a,b))

f_2(12,34)

#f_2 - 인자 a: 12 , 인자 b : 34
#f_2()
#TypeError: f_2() missing 2 required positional arguments: 'a' and 'b'


def f_22(a1,a2,a3):
    print('f_22:',a1,a2,a3)
#f_22(10,20)
#TypeError: f_22() missing 1 required positional argument: 'a3'

#기본 인자 a3= None
def f_222(a1,a2,a3=None):
    print('f_222:',a1,a2,a3)
f_222(10,20)

#기본 인자 a3= None
def f_2222(a1=0,a2=0,a3=None):
    print('f_2222:',a1,a2,a3)
f_2222()
f_2222(10,20)

# f_2222: 0 0 None
# f_2222: 10 20 None

print( '{0:-^50}'.format( "[3] 반환값   O      매개변수(인자/인수)   X"))
# [3] 반환값   O      매개변수(인자/인수)   X
def f_3():
    print('f_3 is called')
    return 'Bread', 'Butter',100

print(f_3())

print( '{0:-^50}'.format( "[4] 반환값   O      매개변수(인자/인수)   O"))
# [4] 반환값   O      매개변수(인자/인수)   O
def f_4(a,b):
    print('f_4 is called')
    return a + b, a * b, a * b, a / b
print(f_4(3,5))

print( '{0:=^50}'.format( "문제"))


# 문제 : 2개의 정수에 대해 큰수, 작은수의 순서로 반환하는 함수를 만드시오
print( '{0:-^50}'.format( "order()"))
# 함수 이름은 order()
def order(n1,n2):
    if   n1 > n2 :  return n1, n2
    elif n1 < n2 :  return n2, n1
    else:           return n1, n2 # 같을 경우 그대로

def order2(n1,n2):
    if   n1 < n2 :  return n2, n1
    else:           return n1, n2 # 같을 경우도 그대로

#강사님
def order3(n1,n2):
    if   n1 < n2 :
        a,b = b,a
        return a,b

print(order(3,10))
print(order(10,10))
print(order(10,3))

print( '{0:-^50}'.format( "max()"))
# max
def max(a,b):
    if a>b:
        return a
    elif a<b:
        return b
    else: # 같을때
        return a,b
print("max(3,10) : {0} ".format(max(3,10)))
print("max(10,10): {0} ".format(max(10,10)))
print("max(10,3): {0} ".format(max(10,3)))


print( '{0:-^50}'.format( "min()"))
# min
def min(a,b):
    if a>b:
        return b
    elif a<b:
        return a
    else:
        return a,b

print("min(3,10) : {0} ".format(min(3,10)))
print("min(10,10) : {0} ".format(min(10,10)))
print("min(10,3) : {0} ".format(min(10,3)))

print( '{0:-^50}'.format( "sum()"))
# sum(list) for 사용
def sum(a):
    result=0
    for i in a:
        result = result + i
    return result

test_list = [1,2,3,4,5,6,7,8,9,10]

print("[1,2,3,4,5,6,7,8,9,10] sum : {0} ".format(sum(test_list)))


print( '{0:-^50}'.format( "mean()"))
#mean(list)
def mean(a):
    sum = 0
    for i in a:
        sum = sum + 1

    mean = sum / len(a)
    return mean


print("[1,2,3,4,5,6,7,8,9,10] mean : {0} ".format(mean(test_list)))


# ==================================================================

# 인수의 갯수가 고정되지 않은 인수 처리 방법
# (1) *변수 : tuple 형식
def add_many(*args): # 갯수 제한 없음
    total = 0
    for k in args:
        total = total + k
    return total

print(add_many(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
print(tuple([ k for k in range(10,101,10)]))


# ==================================================================

# 키워드 인수(사전사용)
# (1) **변수 : dict 형식
def rhkd(width, heigh, **args):   # **는 사전 데이터를 받겠다.
    print(width, heigh)
    print(args)
rhkd(10,20,depth=5, dimension = 7) # 사전 데이터로 만듬
# 인자에 값을 주지만 사실은 키워드로 dict 함수로 된거임
# dict 타입이다

# ==================================================================

# 함수 사용시 변수의 유효범위 규칙(scope rule)
# LEGB 규칙 : Local > EnclosingFunction Local
#                 > Global > Built-in

# 로컬이 가장 작은 개념
# 글로벌은 함수 밖에 만든 것
x = 10      # g : 전역변수(global variable)
y = 20      # g : 전역변수(global variable)

def foo():
    x = 20    # l : 지역변수(local),(함수 내에 있으니까 같아도 다르게 사용)
    pass
    print('foo : ',x)   # x = 20

foo()

def foo():
    x2 = 20
    pass
    print('foo : ',x)   # x = 10

foo()          # 지역 변수가 가장 우선이다. 없으면 그 다음 없으면 그 다음

def foo():
    x = 20
    foo_list = [0]
    print('foo : ',x)
    def bar():
        a = 30   # 지역 변수임
        print('bar:',a,x,y)     # a : 지역변수, x : Enclosing(함수 사이에)
                                # y : 글로벌
    bar()
foo()

def foo2():
    foo_list = [0]              # 다른 함수는 다른 변수임(같은 이름이라도)
# ==================================================================

# 일급 함수(first class function)

# (1) 함수객체를 다른 함수의 인수로 전달할 수 있다.
# (2) 함수객체를 반환 값으로 전달할 수 있다.
# (3) 함수객체를 다른 자료구조에 저장해서 사용가능.

def add_1(a,b):
    c = a+b
    print("%d"%c)



k = add_1(1,2)
print(k)

3

result = func_1(add_1,10,20)            # 함수를 인자형으로 변환
print(result)

def foo2():
    print('foo2 is called')
    def bar2():
        print('bar2 is called')
    return bar2()                # 함수 객체를 반환
result = foo2()
result()                         # class, function

# 함수를 자료구조에 저장
function_list = [add, subtract, multiply, divide]
function_list[0](10,50)          # add(10,50)
print(result)

a = add
a(1,2)
b = a          # a가 b가 된다
b(3,4)

# ==================================================================

# 람다 함수_한 줄짜리 함수식
# 식을 정의하는 순간 바로 함수 객첵로 사용
# 바로 인수로 전달이 가능하다

# 일반 함수
# def add_lam(a,b):
#     return a + b
# 함수명 = lambda <인수>1, <인수2>,.....> : 반환할 식
# 모든 함수에 적용 x 일부분만
add_lam = lambda a,b: a+b
result = add_lam(1,8)
print(result)

def f1(x):
    return x*x + 1

def g(func):
    return [func(x) for x in range(1,5)]

print([f1(x) for x in range(1,5)])
print([f1m, f(1),f(2),f(3),f(4)])
print(g(f1))
print(g(lambda x: x*x+1))             # 람바함수 사용

# ==================================================================

# 파이썬 내장(built_in) 함수

# map 함수
# 여러번 해야되는 것을 한번에 한다.
def mymap(func,var_list):
    result_list = []
    for k in var_list:
        result_list.append(func(k))
    return result_list

def multi_two(x):
    return x*2

# (abs) : 절대값을 구해주는 함수
print(abs(-123123))           # 절대값으로 바뀐다
print(-301)

# chr(숫자) : 아스키 코드 문자.
print(chr(97))  =>   'a'가 나온다

1 ~ 255까지 아스키 코드값과 해당 문자가 나오도록
97 : a
98 : b
for k in range(1,255):
    print(k,':', chr(k),end=',')
    if k%5 == 0:
        print
print

# ard() : 문자의 아스키 값을 반환
print(ord('a'))      # 97
# max()/min()/sum()/round()/pow()
# id() : 개체의 참조 주소 값을 반환
print(list(zip([1,2,3],[5,6,7])))
print(list(zip([1,2,3],['a','b','c'],[5,6,7])))
# [(1, 'a', 5), (2, 'b', 6), (3, 'c', 7)]

# globals()/locals() : 전역/지역 심볼테이블을 얻는다.
print(globals())
print(locals())

# ==================================================================
