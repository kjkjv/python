# 05-2 모듈

# import '소스'
# '소스'.'함수'
#
# 모듈 만들기
# 모듈 불러오기
# if__name__=='__main__': 의 의미
# 클래스나 변수 등을 포함한 모듈
# 다른 파일에서 모듈 불러오기

main 모듈
import calculator


print(__name__)

if__name__== '__main__':
    result = calculator.add(100,200)
    print(result)


import calculator as cal          # 불러오는 파일 이름 바꾸기

from calculator import add          # add라는 함수 하나만 쓰겠습니당
from calculator import add,subtract # 두 개가 필요하다면
from calculator import *            # 다 필요하다면

