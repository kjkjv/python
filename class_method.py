# Calculator 클래스 구현

# 1. class 를 정의
# 2. 클래스 멤버를 정의
# 3. 생성자를 구현
# 4. 인스턴스 멤버를 정의하라
# 5. 메소드를 구현
# 6. 소멸자는 생략해도 된다
# 7. 클래스의 인스턴스 객체를 만들고 메소드를 호출해본다
# 8. 모듈로 제공하기 위한 처리 구현
# if __name__ == '__main__':


class Calculator:
    a = 10
    b = 20
    def __init__(self,a,b):     #생성자임
        print('생성자 호출')
        self.a = a              # 위의 a,b와는 다르다 로컬이 먼저임 위에는 클래스고
        self.b = b

    def add(self, a,b):
        self.a = a
        self.b = b
        return self.a + self.b

    def subtract(self, a,b):
        self.a = a
        self.b = b
        return self.a - self.b

    def multiply(self, a,b):
        self.a = a
        self.b = b
        return self.a * self.b

    def divide(self, a,b):
        self.a = a
        self.b = b
        return self.a / self.b

    def __del__(self):
        print('소멸자')

if __name__ == '__main__':         #
    c1 = Calculator(1,1)
    print(c1.a,c1.b)
    print(c1.add(20,30))
    print(c1.subtract(20,30))
    print(c1.multiply(20,30))
    print(c1.divide(20,30))