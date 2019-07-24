# 05_3 클래스 연산자 중복.py

# 연산자 중복 (operator overloading)
# 파이썬에 내장되어 있는 메소드(연산자, 함수)를
# 사용자 클래스내에서 재정의해서 사용
# 연산을 얼마든지 재정의해서 쓸 수 있다.
# 파이썬의 확장성
# 상속과는 살짝 다른 개념
# 무한대로 확장할 수 있어진다.
# __함수__ 이런 것들은 전부 다 중복이 가능

class Calculator:

    def __init__(self):
        self.a = 0
        self.b = 0
        self.info_str = 'Calculator class v1.0'

    def add(self,a,b):
        return a + b

    def subtract(self,a,b):
        return a - b

    def __add__(self):
        print('연산자 중복입니다')
        return self.a + self.b

    # __str__는 메소드를 대신할 수 있다.
    def __str__(self):
        print('연산자 중복:__str__')
        return self.info_str

    # __repr__은 str 메소드를 대신할 수 있다.
    def __repr__(self):
        print('연산자 중복:__repr__')
        return self.info_str
#   이것이 활용도가 높다. 전부 출력 가능하기 때문에
#   class 안의 속성을 한번에 표현하는 기능

cal = Calculator()
cal.a = 300
cal.b = 500
print(cal.add(10,20))
result = cal.a + cal.b
print(result)



# <class int>
a = 10
b = 20
c = a + b
c = a.__add__(b)     # +   ==>   __add__()
print(c)


# <class str>
a = '하이'
b = '바이'
c = a + b
c = a.__add__(b)

print(str(cal))
print(repr(cal))

