# calculator.py

def add(a,b):            # 함수 정의
    c = a + b
    print('add called')
    return c

result = add(10,20)               # 함수 호출
print(result)
print('종료')

def subtract(x,y):
    c = x-y
    print('subtract')
    return c
result = subtract(20,10)
print(result)


def multiply(x,y):
    c = x * y
    print('multiply')
    return c
result = multiply(20,10)
print(result)



def divide(x,y):
    c = x/y
    print('divide')
    return c


print(add(10,20))
print(multiply(10,20))