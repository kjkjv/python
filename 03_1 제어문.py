# 03-1 제어문.py

# if, for, while 문

# 할당 연산자, 프린트 등은 문장


# if============================================================================
a = 8
if a > 5:
    print('big')
else :
    print('small')

a = 15
if a == 10:
    print('a=10')
else:
    pass                    # pass는 아무것도 안함 자리를 차지하기 위해, 문장이 필요하나 아무 일도 하면 안될 때

order = 'ham'
if order == 'spagetti':
    price = 5000
elif order == 'spam':
    price = 6000
elif order == 'egg':
    price = 7000
elif order == 'ham':
    price = 8000
else :
    price = 0

print('order price : %d'%price)


if a > 5:
    x = a*2
else:
    x = a/2
print(x)

x = a*2 if a > 5 else a/2
print(x)


# 파이썬 스타일
x = (a/2, a*2)[a>5]
print(x)
# 위에를 풀어쓰면
t = (5, 20)
index = a>5
x = t[index]


# 파이썬 스타일(분석가들은 이렇게 해야 한다.)
a = 13
x = {True : '홀수',False : '짝수'}[a%2]      # 딕셔너리 표현으로 접근 키 값을 조건식으로 만든다.
print(x)

x = {True : '홀수',False : '짝수'}[a%2][1]   # [1]앞을 커다란 덩어리로 보고 나온 값을 인덱싱이 가능
x = {True : '홀수',False : '짝수'}[a%2].strip('수')  # 아직 실행이 안되니 함수는 안나오지만 실행이되면 가능
print(x)





# for==============================================================================================

for k in range(10):      # 일반적으로 range 함수 사용
    print(k)

l = [1,2,3,4,5,6]        # 응용하여 다른 것도 사용 가능
for k in l:
    print(k)

l = ['cat','dog','pig']  # enumerate(열거하다), 리스트를 인덱싱하여서 표시해줌 /i=인덱스, k=리스트/ 인덱스값도 반환
for i,k in enumerate(l):
    print(i,k)
    if i == 2 :
        print('[2=%s]'%k) # 조금 더 강조하고 싶다면

# continue, break 문
for x in range(100):
     if x > 3 :
         break             # for 문 loop를 탈출
     print(x)

for x in range(100):
     if x < 3 :
         continue          # 사실 skip에 더 가깝다, 다시 for문의 다음 문장을 skip 하고
     print(x)              # for 문의 시작부분으로 이동

l = ['cat','dog','pig']
for x in l:
     if x == 'pig' :
         continue          # 사실 skip에 더 가깝다, 다시 for문의 다음 문장을 skip 하고
     print(x)
     4
# 중첩된(nested) for 문
for x in range(2,4):
    for y in range(1,10):
        print(x,'*',y,'=',x*y)

# while 문
a = 0
while a < 10 :
    print(a)
    a = a + 1

while True :         # 무한 루프, true 를 넣을 수도 있다.
    print(a)
    a = a + 1

while 3 > 0 :         # 무한 루프
    print(a)
    a = a + 1
    if a > 1000: break    # while 에는 break 를 걸어 둔다. / 조건이 걸리도록 설정



# while 문 : 특정 조건이 참일 때 반복
# for 문 : 지정한 횟수만큼 반복


a = [1,2,3,4]
a.insert(0,a)
print(a)           #[[...], 1, 2, 3, 4], 순환 참조 리스트
print(a[0])
print(a[0][0])


