# 02-6 집합.py
# set, {} <- dict와 비슷해서 혼돈, ㅡmutable 자료형
# 중복허용하지 않는다

s = {1,2,3}         # dict과 다르게 key값이 없이 value 값만 있다.
print(s, type(s))   # {1, 2, 3} <class 'set'>

s = {3,2,1}
print(s, type(s))   # {1, 2, 3} <class 'set'>, 순서 차이가 없다.

l = [1,2,2,3,3,4,4,5]   # 중복허용
print(len(l))           # 8

s = {1,2,2,3,3,4,4,5}
print(s, type(s))       # {1, 2, 3, 4, 5} <class 'set'>, 중복허용 x
print(len(s))           # 5

s.add(6)                # 집합에 하나의 요소만 추가
print(s)

s.update({7,8,9})       # 집합에 여러 요소를 추가

del s.(1)

s.remove(9)             # 특정 요소를 제거
print(s)

# 집합의 연산 : 교집합, 합집합, 차집합, 대칭 차집합
s1 = {1,2,3,4,5,6}
s2 = {6,7,8,9,10}

# 합집합 : |, union
s = s1 | s2
print(s)            # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

s = s1.union(s2)    # 이것도 합집합, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# 교집합 : intersection(), &
s = s1 & s2
print(s)            # {6}

s = s1.intersection(s2)
print(s)            # 이것도 교집합, {6}

# 차집합 : '-', difference()
s = s1 - s2
print(s)               # {1, 2, 3, 4, 5}
s = s1.difference(s2)  # 이것도 차집합, {1, 2, 3, 4, 5}

# 대칭 차집합
s = s1.symmetric_difference(s2)   # (A-B) union (B-A)
print(s)                          # {1, 2, 3, 4, 5, 7, 8, 9, 10}

s = (s1-s2) | (s2-s1)             # 이것도 대칭 차집합
print(s)



# 조건식과 조건문


