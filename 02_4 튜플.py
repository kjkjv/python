# 02-4 튜플.py
# tuple : immutable 자료형, 시퀀스(연속, 순서형), () : 소괄호 사용

# [] : 대괄호, {} : 중괄호, () : 소괄호

a = (1,2,3,4,5,6,7)
print(a,type(a))     # <class 'tuple'>

# indexing
print(a[0],a[1],a[2])
print(a[-1],a[-2],a[-3])

# a[0] = 10            TypeError / 요소값 변경이 불가능

b = list(a)
b[0] = 10
b = tuple(b)
print(b)

# 패킹 / 언패킹

t = 1,2,3
print(t,type(t))      # <class 'tuple'>

a, b, c = 10, 20, 30  # a, b, c는 모두 int형
print(type(a),type(b),type(c))  # <class 'int'>

t = 1,2, 'hello'       # <class 'tuple'>                                        #패킹(튜플)
x,y,z = t              # 튜플안의 요소를 넣으면 int int str 으로 타입이 바뀐다     #언패킹(튜플)

l = ['food','bar',4,5]  # 패킹(리스트)
x,y,z,w = l             # 언패킹(리스트)

# 확장된 언팩킹
t = (1,2,3,4,5)
a,b = t                 # ValueError: too many values to unpack (나머지를 꺼낼 곳이 없어용 ㅠㅠ)
print(a,b)

a, *b = t               # a에 하나 놓고 나머지는 b에다 넣는다.
print(a)                # 1
print(b)                # [2, 3, 4, 5] 단, 리스트로 넣는다.

*a,b = t
print(a)                # [1, 2, 3, 4]
print(b)                # 5

a,b,*c = t
print(a)                # 1
print(b)                # 2
print(c)                # [3, 4, 5]

a,*b,*c = t             # SyntaxError


t = ()     # 빈 튜플
t = 1,2,3
t = 1      # int 형
t = 1,     # tuple 형, 데이터가 1개라도 쉼표를 사용하면 튜플
t = (1,)   # tuple 형, 데이터가 1개라도 쉼표를 사용하면 튜플

# 연산자
t1 = (1,2,3)
t2 = ('apple','banana')
t3 = t1 + t2            # 합연산  t3 = t1.__add__(t2)와 동일결과
print(t3)               # (1, 2, 3, 'apple', 'banana')
t4 = t1*2               # 반복 연산
print(t4)               # (1, 2, 3, 1, 2, 3)
print(len(t1))          # 3, 요소의 갯수
print(1 in t1)          # True
print(sum(t1))          # 6, 시퀀스형 데이터의 총합

t = (1,2,3,2,2,3)
print(t.count(2))       # 3
print(t.index(2))       # 1
print(t.index(2,4))     # 4, 앞에 2는 찾을 인덱스고 뒤에 4는 찾기 시작할 위치

# 튜플 중첩
t = (12345, 54321, 'hello')
u = t, (1,2,3,4,5)
print(u)                # ((12345, 54321, 'hello'), (1, 2, 3, 4, 5))
print(u[0][2])          # hello

# 두개의 값을 서로 변경
x,y = 1,2
x,y = y,x     # 간단하게 값을 변경
print(x,y)

(x1,y1),(x2,y2) = (1,2),(3,4)   # 복수개의 변수값을 할당 가능
print(x1,y1,x2,y2)      # 1 2 3 4

# named tuple   : 이름있는 튜플
# 튜플을 숫자 인덱스가 아닌 이름으로 접근 가능하다
from collections import namedtuple
point = namedtuple('ppoint', 'x,y')       # named tuple
                                          # 클래스 객체를 생성
print(point.__name__)                     # 이름을 출력
pt1 = point(1.0, 5.0)
pt2 = point(2,3)
pt3 = point('hello', 'goodbye')
print(pt1,pt2,pt3)
# ppoint(x=1.0, y=5.0) ppoint(x=2, y=3)
# ppoint(x='hello', y='goodbye')
# x라는 변수명을 사용할 수 있다.
number = pt1.x + pt1.y                            # 요소값이라 int형이고 인덱스 필요없이 편하게 불러온다
string = pt3.x + pt3.y
print(string)                                     # hellogoodbye


