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
