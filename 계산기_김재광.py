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
