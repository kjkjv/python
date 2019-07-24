# ====================0702==========================
# 02-2 문자열.py
'''
파이썬 문자열 자료
'''

a = "python is great"
print(a)
b = 'life is short, you need python'
print(b)
c = "he said 'i love you'"
print(c)
d = 'he said "i love you"'
print(d)

# d = "he said "i love you""   ==>  SyntaxError가 난다.
# print(d)

f ='''
HERE!!
hello
python
'''
print(f)

3   # 변수가 없어도 오류는 생기지 않는다. 다만 주석처럼 쓸 수는 없다.

# 자료형에 {변경 가능, 변경 불가능(숫자형,int, float, tuple 등등)은 읽어올 수 있다.} 이 존재한다.

# 문자열의 indexing / slicing
# 값을 가져오거나 나눌 수 있으나 변경은 불가능하다.

#=====인덱싱======
a = 'life is too short, you need python'

print(a[0], a[1], a[2], a[10])
#a[0] = 'L'                     # TypeError: 'str' object does not support item assignment, 불변형 자료라 수정 X
print(a[0], a[1], a[2], a[-1])
a[-3]                          # 끝에서 부터 인덱싱, 거꾸로 세주는 기능
# 머신러닝에서 마지막 값은 답이 나온다. 그래서 [-값]으로 편하게 추출이 가능해짐.

print(a[0], a[1], a[True], a[False])
# True가 1값으로 False가 0값으로 처리된다.

#=====슬라이싱======
# 슬라이싱 : a[시작옵셋 : (끝옵셋 +1) ]
print(a[0:3])    # FROM 0 TO (3-1)
print(a[12:-1])  # 끝자리를 제외하고 까지
print(a[:])      # 전체가 다 나옴

print(a[-1:-8]) #?


# a[시작옵셋 : (끝옵셋 +1) : 간격(step) ]
print(a[1:30:3])
# 앞에 두 개는 범위 뒤에는 표시되는 간격(3간격으로 표시), 원래는 1이 생략된 것이다.
print(a[::2])  # 전체 범위에서 2간격으로 표시
print(a[::-1]) # 전체 범위에서 거꾸로 표시, nohtyp deen uoy ,trohs oot si efil
print(a[::-2]) # nhy enuy,rh o iei, (-) 값은 거꾸로 인덱싱

a = "20010303Rainy"
year = (a[0:4])
date = (a[4:8])
weather = (a[8:])     # 처음 시작 0을 제외해도 되고 끝까지면 역시 제외해도 괜찮다.

print(year)
print(date)
print(weather)


# 객체를 수정하는 방법 (불변형 자료)
a = "pithon"
b = a[:1] + "y" + a[2:]
print(b)

# 문자열 연산 : <class 'str'>의 method
s1 = 'first'
s2 = 'second'
s3 = s1 + s2
print(s3)

len(s3)      # 문자열의 사이즈

a = "i like programming. i like swimming."
type(a)      #<class 'str'>

print(a.upper())  # 대문자로 변환하는 메소드(스트링 안에서의 함수, 메소드)
# a라는 큰 범주를 먼저 정하고 함수나 메소드를 사용한다. (.)을 사용하여서. ex) a.메소드
print(a)          # 원래값은 유지한다. 불변 자료는 원형값을 건들지 않는다.

b = a.upper()     # b 라는 새로운 값을 만들었음으로  b는 변형 가능.
print(b)
print(a)
# 사람 class 홍길동은 인스턴스(구체화하여 사용할 수 있게 만든 것.)

c = b.lower()     # upper 와 반대되는 함수
print(c)
d = c.title()     # 단어의 처음을 대문자로 강조
print(d)

num = a.count('like')     # count = like 라는 단어를 몇 번 사용?
print(num)


# ===find===
num = a.find('like')      # like 라는 글자가 어디에 위치하는지. 많이 사용(find). 인덱스 값을 반환
print(num)

num = a.find('hello')     # 찾는 것이 없으면 -1
print(num)

num = a.find('like', 3)   # 22. 검색할 위치를 검색. (,3)은 인덱스 3부터 검색해봐라.
print(num)

num = a.rfind('like')     # 역순으로 검색, 뒤에서 부터 검색
print(num)

num = a.index('like')     # like의 인덱스를 검색 find와 비슷.
print(num)

num = a.index('hello')    # ValueError가 뜬다. find와 다른 점. find는 없으면 -1인데 index는 에러가 뜬다.
print(num)

b = a.startswith('i like')    # 'i like'로 시작하는 문자열인가?
print(b)                      # True


# ===strip===
a = '  spam and ham  '
s = a.strip()                 # 좌우 공백 제거
print(s)

s = a.lstrip()                # 왼쪽 공백 제거
print(s)

s = a.rstrip()                # 오른쪽 공백 제거
print(s)


# ===split===
s1 = a.split()
print(s)                      # ['spam', 'and', 'ham'] 공백을 기준으로 분류하고 리스트로 반환

s = a.split('and')
print(s)                      # split (x) x를 기준으로 나누어 분류하고 리스트로 반환

# ===join=== 문자를 삽입하여 합친다.
s2 = ':'.join(s1)
print(s2)                     # spam:and:ham ==> split이랑 반대? 되는 개념

a = 'asdfq'
s2 = ':'.join(a)
print(s2)                     # a:s:d:f:q  ==> 문자열도 가능

#===replace===       =>  변환해주는 함수
a = 'life is too short'
b = a.replace('life', 'your leg')
print(b)

# ===eval()===
# : 문자열로 표현된 식(expression)을 파이썬 인터프리터가 번역하여 실행시킨다.
#   문자열을 숫자열로 변환할 필요가 없다.

a = 10
b = 20
# c = a + b   => 이것은 문장이다. 어디엔가 할당하는 것은 문장
c= eval('a+b')           # 문자열을 숫자열로 알아서 계산해줌
print(c)

# ===byte=== : 0~255(0xFF)
b = b"python reles"           # 앞에 b라고 나와있으면 bite 데이터이다. <class 'bytes'>
print(b, type(b))             # <class 'bytes'>

s = b.decode('utf-8')         # <class 'str'>으로 변형가능. byte 문자를.
print(s, type(s))

b2 = s.encode('utf-8')         # <class 'bytes'>로 다시 변형가능.
print(b2, type(b2))


# maketrans() / translate() 함수 :
# 문자열을 mapping하여 변환한 결과를 얻을 수 있다

a = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'
b = 'ㅏㅔㅣㅗㅜ'
c = 'gndrmbsojcktph'
d = 'aeiou'

trancetable = ''.maketrans(a+b,c+d)
result = 'ㄱㅏㄴㅣㅂㅏㅂㅗ'.translate(trancetable)
type(trancetable)            # <class 'dict'>

print(result)
print(len(a),len(b))