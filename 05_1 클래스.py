# 05-1 클래스.py

# 파이썬 클래스 관련 용어

# 클래스(class) :
# class 문으로 정의 하며, 멤버와 메서드를 가지는 객체이다.
# 새로운 이름 공간을 갖는 단위.
# 명사나 명사구가 클래스
# 클래스 객체(Class Object) : 클래스와 같은 의미

# 클래스 인스턴스(Class Instance)_가장 중요*
# : 클래스를 호출하여 생성된 객체
# 클래스 인스턴스 객체(Class Instance Object)
# : 클래스 인스턴스와 같은 의미, 인스턴스 객체라고 부르기도 한다.

# 멤버(Member) : 클래스 혹은 클래스 인스턴스 공간에 정의된 변수
# 메서드(Method) : 클래스 공간에 정의된 함수, def 로 정의

# 속성(Attribute) : 멤버와 메서드 전체를 가리킴

# 상속(Inheritance)
# : 상위 클래스의 속성과 행동을 그대로 받아들이고
#   추가로 필요한 기능을 덧붙이는 것

# class 와 class instance 객체의 생성
# class 클래스 이름(대문자로 하는 습관) :
#         속성....
class Simple :  # 클래스를 선언
    pass

print(Simple)      # <class '__main__.Simple'>

s1 = Simple()      # ()를 하면 클래스를 함수처럼 호출
                   # 클래스의 인스턴스 객체의 생성
print(s1)          # <__main__.Simple object at 0x0000021F4BDAF588>

s2 = Simple()
print(s2)          # <__main__.Simple object at 0x0000021F4BDAF978>_다른 객체

# 멤버 접근  : 클래스 멤버와 인스턴스 멤버
# (1) 클래스 멤버의 구현과 접근 방법
class MyClass :
    class_member = 100     # 클래스 멤버
    class_list = [1,2,3]   # 클래스 멤버
    a = 'hi'

# 클래스 객체를 통해서 접근
print(MyClass.class_list)
print(MyClass.class_member)
print(MyClass.a)

# 클래스 인스턴스로도 접근할 수 있다
# 인스턴스 객체를 통해 변경하면 인스턴스의 값만 변경되고
# 원래 클래스 멤버값은 변경되지 않는다.

MC = MyClass()            # 클래스의 인스턴스 객체
MC.class_member
print(MC.class_member)    # 100


# 틀은 원본을 유지하고
# 복사본은 바뀔 수 있다

# 멤버는 변수
# 메소드는 함수


# (2) 인스턴스 멤버의 구현과 접근 방법*
# 클래스의 메소드 내에 구현한 멤버
# 인스턴스 멤버 : self.변수명 = <값>*
# 클래스 멤버와 인스턴스 멤버는 다른 것이다

class MyClass2():

    def __init__(self) :   # 생성자, 인스턴스 객체를 생성할 때 자동 호출
        print('MyClass2의 생성자가 호출되었어요')
        self.in_mem = 0
        self.a = 0
        self.in_list = [0]

    def set(self,var):     # 메서드
        print('set is called!!!!!!!!!!')
        self.in_mem = var  # 인스턴스 멤버
        self.in_list = [1,2,3]
        self.a = 100
    def get(self):         # 메서드
        return self.in_mem,self.in_list,self.a

    def __del__(self):     # 소멸자
        print("MyClass2의 생성자가 호출되었어요")


m2 = MyClass2()            # 클래스의 인스턴스 객체
m2.set(30)
print(m2.get())
m2.set(50)

m3 = MyClass2()


# * 쉬운 예
# a = 10
# b = 20
# 같은 int라는 class 안에 있는 인스턴스 값이다.
# 같은 class지만 인스턴스는 다른 객체가 된다.


# # __name__main
# main 파일과 module 파일이 있다.
# module에서 메인 파일도
# main에서는 그냥 name 파일이 된다.
# main 파일에서 module 파일을 열고 싶은데 다 실행시키면 곤란하니까
# if 문을 걸어서 프린트를 정지한다.
# 그리고 __name__main을 하면 if 문에 걸려서 안나온다.
#
#
# self는 method 안에서 밖의 값과 안의 값이 같고 싶을 때 사용.




# ===========================클래스 상속==================================

# 05-2 클래스.py

# 상위 클래스 = 부모 클래스 = 기반(base) 클래스
class Calculator:
    def __init__(self):
        print('계산시작하즈아!!')
        self.a_range = 20000
        self.b_range = 1000


    def add(self,a,b):
        if a > self.a_range or b > self.b_range:
            return 'over range!'
        return a + b

    def sub(self,a,b):
        return a - b

    def mul(self,a,b):
        return a*b

    def div(self,a,b):
        return a/b

cal = Calculator()
# print(cal.add(10,20))
# print(cal.div(10,20))
# print(cal.div(10,0))          # ZeroDivisionError: division by zero

# 하위 클래스 = 자식 클래스 = 파생 클래스
# 상속 : class 자식 클래스(부모 클래스1, 부모 클래스2....._다중 상속 가능)
class Calculator_2(Calculator):

    def __init__(self):
        print('계산시작하즈아!!')
        Calculator.__init__(self)       # 부모 클래스의 생성자를 호출
        super                        # 부모에 입력했으니 생성자를 호출
#   추가 메소드 구현
    def pow(self,a,b):
        return a**b
    def add_3(self,a,b,c):
        return a + b + c

#    메소드 오버라이딩 : 상속
#    (부모 메소드를 그대로 쓴다_재정의 개념)
#    부모 함수만 따라하고 인스턴스 멤버는 내 마음대로 해도 가능

    def div(self,a,b):
        if b == 0:
            return 'ZeroDivisionError!!'
        return a/b


mcal = Calculator_2()
mcal.pow(2,3)
mcal.add_3(2,3,4)
# 부모 클래스를 그대로 사용 가능. 상속받기만 한다면
mcal.div(20,0)

print(mcal.add(20202020,110105015))    # over range!


# 부모클래스 인스턴스를 마음대로 바꿀 수도 있다. 범위 재설정
# 메소드와 멤버는 속성, 속성도 바꾼다.
# 오버라이딩이 아닌 그냥 값을 바꿈

mcal.a_range = 32030202320320
mcal.b_range = 1391312313939
print(mcal.add(20202020,110105015))


# def move_backward(self, direction, speed):
# #     self.air_speed = speed

# 위의 self.air_speed = speed 에서
# 앞의 변수명은 초기값 설정과 같이 해줘야 하고,
# 뒤의 speed는 어떠한 값이 아니라 자리의 위치를 말해준다.
# 맨 위의 speed는 인자로 입력될 값의 위치를 말한다.