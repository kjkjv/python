

print('안녕하세요 배송봇입니다.')
print('배송을 위해 개인정보를 수집하겠습니다.')
print('아래의 답변을 적어주세요!')



while True :
    agree = input('개인정보 수집에 동의합니까? yes/no')
    if agree == 'no':
        print('동의하셔야만 이용이 가능합니다.')
        break
    name = input ('성함이 어떻게 되나요?')
    phone = input('{}님, 연락처가 어떻게 되나요?'.format(name))
    agree_phone = input('고객님 {} 이 번호가 맞습니까? yes/no'.format(phone))

    elif agree_phone == 'no'





    elif agree_phone == 'no':
        print('다시 입력해주세요.')
        break
    else :
        print('주문이 완료되었습니다. 감사합니다')
        break







coffee = 10
while True:
    money = int(input("돈을 넣어 주세요: "))
    if money == 300:
        print("커피를 줍니다.")
        coffee = coffee -1
    elif money > 300:
        print("거스름돈 %d를 주고 커피를 줍니다." % (money -300))
        coffee = coffee -1
    else:
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")
        print("남은 커피의 양은 %d개 입니다." % coffee)
    if coffee == 0:
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
        break

