# class_human

class Human:
    life = True
    eyes = 2
    def __init__(self,name,gender):
        print('생성자')
        self.name = name
        self.age = 0
        self.gender = gender
        self.height = 0
        Human.life = True
        self.phone_number = '000-0000-0000'

    def set_name(self,name):                # 이름을 변경
        self.name = name

    def get_name(self):                     # 이름을 반환
        return self.name

    def set_age(self,age):                  # 나이를 변경
        self.age = age + 1

    def get_age(self):                      # 나이를 반환
        return self.age

    def set_age_one_year(self,age):         # 나이를 한 살 증가
        self.age = self.age + 1

    def set_height(self, height):           # 키를 변경
        self.height = height

    def get_height(self):                   # 키를 반환
        return self.height

    def set_phone_number(self,phone_number):
        self.phone_number = '010-3039-3940'

    def __del__(self):                      # 인스턴스 객체 소멸시 자동으로 호출
        print('소멸자')                      # 파이썬에서는 소멸자를 생략하는게 일반적

h1 = Human('홍길동','남성')                  # 클래스의 인스턴스 객체 생성

print(h1.name)                              # 인스턴스 멤버를 접근
h1.get_name()                               # 메소드를 호출




