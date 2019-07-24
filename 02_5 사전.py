#  02-5 사전.py

# Dictionary , <class 'dick>
# {key:values,....}
# primary key는 중복되지 않는 고유의 값, 기준의 값
# foreign key는 그에 해당되는 값
# key는 데이터가 아님
# 순서가 없는 데이터, 그래서 인덱싱이 불가능해서 더 편하다
# int, list 등도 파이썬 안에서는 key,value로 저장
# key값은 반드시 immutable 값을 입력
# 연관형, mutable(요소 추가, 변경이 가능)
# 시퀀스형(순서형)은 아님 : 순서가 없음
# value는 immutable, mutable 모두 가능
# key 가 사라지면 객체에 접근 x 그러나 사라진 것은 x


a = {'name':'홍길동', 'age':30, 'phone':'010-4451-3209','birthday':'20021525'}
print(a)
# {'name': '홍길동', 'age': 30, 'phone': '010-4451-3209', 'birthday': '20021525'}

# 요소 접근 : a[key]
print(a['name'],a['age'],a['phone'],a['birthday'])  # 홍길동 30 010-4451-3209 20021525

# 요소 변경, 추가, 삭제
a['name'] = '김재광'
print(a['name'])    #김재광, 변경 가능, value값이니까!
# key 값은 변경이 불가능해서 아예 삭제를 하고 다시 만들어야 한다
a['age'] = 26
a['age']

a['adress'] = '대전시'
a['직업'] = '대학생'
a[(1,2,3)] = 10      # tuple 은 가능하다
a[[1,2,3]] = 10      # TypeError: unhashable type: 'list'
print(a)

del a['직업']
del a['adress']

# 연산자
print(len(a))        # 5 (key의 갯수)

print('birth' in a)     # True
print(a.keys())         # key를 dict_keys로 반환 / dict_keys(['name', 'age', 'phone', 'birthday', (1, 2, 3)])
print(list(a.keys()))   # 리스트로 반환 / ['name', 'age', 'phone', 'birthday', (1, 2, 3)]
print(a.values())       # value를 dict_values로 반환 / dict_values(['김재광', 26, '010-4451-3209', '20021525', 10])
print(list(a.values())) # 리스트로 반환 / ['김재광', 26, '010-4451-3209', '20021525', 10]
print(a.items())        # dict_items([('name', '김재광'), ('age', 26), ('phone', '010-4451-3209'), ('birthday', '20021525'), ((1, 2, 3), 10)])
print(list(a.items()))  # [('name', '김재광'), ('age', 26), ('phone', '010-4451-3209'), ('birthday', '20021525'), ((1, 2, 3), 10)]

a.get('name')           # 함수에는 인자를 넣을 수 있다. 더 많은 어드벤스를 얻을 수 있다
print(a.get('ㅋㅋㅋ'))   # None
print(a('ㅋㅋㅋ'))       # 오류
print(a.get('ㅋㅋㅋ','ㅎㅎㅎㅎ'))    # ㅎㅎㅎㅎ, key가 없을 때 기본값을 설정한다.

d = {'gender':'남성','ten':10}
a.update(d)              # 새로운 dict의 요소를 모두 추가
print(a)                 # {'name': '김재광', 'age': 26, 'phone': '010-4451-3209', 'birthday': '20021525', (1, 2, 3): 10, 'gender': '남성', 'ten': 10}

d = {'name':'박재광','gender':'남성','ten':10}     # key는 중복이 안되고 value 값은 mutable이라 수정이 가능하다. 즉, 중복된 key값은 value값이 바뀐다.

a.pop('gender')          # 순서의 개념이 사라지니 마지막이 없어지는 것이 아니다

# dict comprehension(사전 내장)
a = {key:1 for key in 'abcdefg' }
print(a, type(a))
# {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1} <class 'dict'>

b = ['name', 'age', 'phone', 'birth']
a = {key:1 for key in b }
# {'name': 1, 'age': 1, 'phone': 1, 'birth': 1} <class 'dict'>

b = ['name', 'age', 'phone', 'birth']
a = {key:1 for key in b }

a1 = ('name', 'age')
a2 = ('김재광','20')
b = {a1[0]:a2[0],a1[1]:a2[1]}   # {} 키와 밸류는 중괄호
print(b)                        # {'name': '김재광', 'age': '20'}

a = { x:y  for  x,y  in (a1,a2) }
# 두 개의 다른 리스트를 사용하여 한 개의 사전을 생성
b = { x:y  for  x,y  in zip(a1,a2) }
print(a)                       # [('name', '김재광'), ('age', '20')]

a = list(zip(a1,a2))           # [('name', '김재광'), ('age', '20')]

