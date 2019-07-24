# ﻿[ 제어문 실습과제 ] ===========================================

# 1 번
# 1-1
for x in range(1,101):
    if x%10 != 0:
        print(x, end =',')
    else :
    print(x)

# 1-2_일단 패스
l = list(range(1,101))
x = [a for a in l if l%10 != 0]
print(x, end =',')


# 2번
n = input('n=')
a = range(1, int(n)+1)
list = []
for b in a:
    list.append(b)
print('총합=',sum(list))


# 3번
n = input('n=')
a = range(1,int(n)+1)
list1=[]
list2=[]

for b in a:
    if b%2==0 :
        list1.append(b)
    else :
        b%2!=0
        list2.append(b)
print('짝수의 합=',sum(list1))
print('홀수의 합=',sum(list2))


# 4번
n = input('n=')
a = range(1,int(n)+1)
list = []
for b in a:
    if b%3 != 0 and b%5 != 0:
        list.append(b)
print(sum(list))


# 5번
# 5-1
for x in range(2,10):
    for y in range(1,10):
        print('{}*{}={} '.format(x,y,x*y))


# 6번
n = 0
count1 = 0
count2 = 0
count3 = 0
while n != -999:
    n = int(input('n='))
    if n<0:
        count1 = count1 + 1
    else :
        n>0
            if n%2==0:
                count2 = count2 + 1
            else :
                n%2!=0
                count3 = count3 + 1
print('음수의 개수=',count1)
print('양수 홀수의 개수=',count2)
print('양수 짝수의 개수=',count3)

# 문제 7번
dict = {1:'+', 2:'-', 3:'*', 4:'/'}
type(dict)
a = (input('a='))
b = (input('b='))