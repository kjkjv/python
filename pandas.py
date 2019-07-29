# pandas
# 실습 기본 문제==============================================================
import numpy as np
# 1번
import pandas as pd
df = pd.read_csv('boston_train.csv')

df.shape
df.index
df.values
df.columns

df[:]
df[10:20]

df['CRIM']
df[df.columns[2]]
df[df.columns[-1]]

r = df['TAX']
print(r)

r2 = df[df.columns[7]]
print(r2)

df2 = df[['ZN','INDUS','NOX','RM']]       # 괄호를 두 번 쓸 것
print(df2)

df3 = df[[df.columns[3],df.columns[5],df.columns[7],df.columns[0]]]
print(df3)


df_real = df2 + df3
df_real.to_csv('연습 한번 해봤다.csv',index=False)

df.iloc[3,5] = 17
df.iloc[3,5]

print(df.loc[78,'INDUS'])

# =========================실습 문제===================================================

# 1번
print(df)
df.index
df.describe()
df.columns
CRIM=df['CRIM']
ZN=df['ZN']
INDUS=df['INDUS']
NOX=df['NOX']
RM=df['RM']
AGE=df['AGE']
DIS=df['DIS']
TAX=df['TAX']
PTRATIO=df['PTRATIO']
MEDV=df['MEDV']
CRIM.describe()
CRIM.mad()
CRIM.min()

# 2번
df[CRIM>CRIM.mean()]
df['CRIM'>'CRIM'.mean()]
df[df['CRIM'] > df['CRIM'].mean()]
CRIM.mean()
df[df['CRIM'] > df['CRIM'].mean()]
df[df > df['CRIM'].mean()]

df[df['AGE'] < df['AGE'].mean()]
df[df < df['AGE'].mean()]
AGE.mean()

df[df < df['MEDV'].median()]
df[df > df['MEDV'].median()]
MEDV.median()


# 3번
df_test = pd.read_csv('boston_test.csv')
df_test.to_csv('dsfa.csv',index=False)

a=df[:11]
b=df_test[:11]
rufrhk = pd.concat([a,b],axis = 0, ignore_index=True)
print(rufrhk)
result = df[:11].append(df_test[:11])
result.to_csv('저장해봤다.csv',index=False)     # numpy 파일은 csv로 저장할 수 없음.


# 4번

CRIM=df['CRIM']
ZN=df['ZN']
INDUS=df['INDUS']
NOX=df['NOX']
RM=df['RM']
AGE=df['AGE']
DIS=df['DIS']
TAX=df['TAX']
PTRATIO=df['PTRATIO']
MEDV=df['MEDV']

sum,mean,median,min,max

print('CRIM''sum=',CRIM.sum(),'mean=',CRIM.mean(),'median=',CRIM.median(),'min=',CRIM.min(),'max=',CRIM.max())
print('ZN''sum=',ZN.sum(),'mean=',ZN.mean(),'median=',ZN.median(),'min=',ZN.min(),'max=',ZN.max())
print('INDUS''sum=',INDUS.sum(),'mean=',INDUS.mean(),'median=',INDUS.median(),'min=',INDUS.min(),'max=',INDUS.max())
print('NOX''sum=',NOX.sum(),'mean=',NOX.mean(),'median=',NOX.median(),'min=',NOX.min(),'max=',NOX.max())
print('RM''sum=',RM.sum(),'mean=',RM.mean(),'median=',RM.median(),'min=',RM.min(),'max=',RM.max())
print('AGE''sum=',AGE.sum(),'mean=',AGE.mean(),'median=',AGE.median(),'min=',AGE.min(),'max=',AGE.max())
print('DIS''sum=',DIS.sum(),'mean=',DIS.mean(),'median=',DIS.median(),'min=',DIS.min(),'max=',DIS.max())
print('TAX''sum=',TAX.sum(),'mean=',TAX.mean(),'median=',TAX.median(),'min=',TAX.min(),'max=',TAX.max())


# 반복은 무조건 for 문을 생각해서 하자 재광아!!!!

for col in df.columns:
    print(df[col].sum())
    print(df[col].mean())
    print(df[col].median())
    print(df[col].min())
    print(df[col].max())


# ===================================================================================
# *데이터 분류하기
#
# train : 학습 (70%)
# test : 검증, 정확도 측정 (30%)
# predict : 예측용, 답이 없는 데이터 (거의 안씀)

# 단위를 통일시켜라 (스케일링)
# 해당 열을 똑같이 올려주거나 내려줘도 무방
# 데이터를 이상적으로 만들어서 그래프를 알맞게 만들어줘야 한다.
# 분석은 텐선플로 보다 전처리가 중요하다
# ===================================================================================


# 5번
import quandl
sunspots = quandl.get('SIDC/SUNSPOTS_A')
sunspots.columns
sunspots['Yearly Mean Total Sunspot Number'].isna().sum()
sunspots['Yearly Mean Standard Deviation'].isna().sum()
sunspots['Number of Observations'].isna().sum()
sunspots['Definitive/Provisional Indicator'].isna().sum()
sunspots.fillna(0)
sunspots.isna().sum()
sunspots.to_csv('sunspots.csv')
# df.fillna() nan 갯수변형
# df.isnull().values.any()


# 6번
sunspots[Date]

zlzl=pd.read_csv('sunspots.csv')
zlzl.columns
zlzl['Date']

result=pd.to_datetime(zlzl['Date'])
print(result)
pd.DataFrame.dtypes(result)
type(result)
result.mean()
result[result>result.mean()]

zlzl['Date'] = result[result>result.mean()]
zlzl.to_csv('sunspot_new2.csv')


# 7번
data = {'판매월':['1월','2월','3월','4월'],
        '제품A':[100,150,200,130],
        '제품B': [90,110,140,170]}
print(data)
il = pd.DataFrame(data)
print(il)

data_2 = {'판매월':['1월','2월','3월','4월'],
        '제품c':[110,130,120,320],
        '제품d': [130,190,140,130]}

il_2 = pd.DataFrame(data_2)

pd.merge(il,il_2, how='inner', on='판매월')


# 8번

gkwk = {'key':['a','b','c'],
        'left':[1,2,3]}
gkwk2 = {'key':['a','b','d'],
        'left':[4,5,6]}

rhkd=pd.DataFrame(gkwk)
rhkd2=pd.DataFrame(gkwk2)

pd.merge(rhkd,rhkd2,how='inner',on='key')
pd.merge(rhkd,rhkd2,how='outer')
pd.merge(rhkd,rhkd2,how='left', on='key')
pd.merge(rhkd,rhkd2,how='right', on='key')


# 9번
df=pd.read_csv('WHO_first9cols.csv')

print(df[df['Country'].str.contains('Albania')])
print(df[df['Country'].str.contains('Ethiopia')])
gg = df['Country'].str.contains('Al')

print(df[gg])                 # 데이터 프레임으로 출력
print(df['Country'][gg])      # 요소값들로 출력



# ==================Series 사용 실습 과제===================================

# 문제 1번)
# Pandas의 Series 데이터구조를 이용하여 다음과 같은 기능을 구현하세요.
# append()와 contains()메소드 사용

import pandas as pd
index_col= ['제품','테레비','냉장고','노트북']
product = pd.Series(['수량',[],[],[]],index=index_col)
print(product)

print('='*60)
print('제품수량관리')
print('1. 입력 2. 출력 3. 검색 9. 종료 중에 메뉴를 선택하세요')
print('='*60)
while True:
    n = int(input('입력하세요'))
    if n == 1 :
        while True:
            p = input('제품명')
            print(product[p])
            num = int(input('수량'))
            product[p].append(num)
            print(sum(product[p]))
            a = input('계속 입력? y/n')      # print를 안써도 input은 문구가 뜬다
            if a == 'y' :                    #''를 안쓰면 변수값이 된다.
                continue
            else:
                break
    if n == 2 :
        print(product)
    elif n == 3 :
        p = input('제품명')
        print(product[product.index.str.contains(p)])   # 인덱스값으로 구하는 방법
    else:
        n == 9
        print('종료합니다.')
        break

# ===============================================================================
import pandas as pd
import seaborn as sb
titanic = sb.load_dataset('titanic')

titanic.columns

titanic.to_csv('titanic_new.csv')

titanic.corr()



print(product)
print(input('계속 입력하시겠습니까? Y/N'))


