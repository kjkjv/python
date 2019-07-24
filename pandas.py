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


# 3번_??
df_test = pd.read_csv('boston_test.csv')
df_test.to_csv('dsfa.csv',index=False)


a=df[:11]
b=df_test[:11]

result=a.append(b)
print(result)
result.to_csv('연습이다2222.csv',index=False)     # numpy 파일은 csv로 저장할 수 없음.


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
#



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


