# matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================================[ matplotlib  실습]==================================

# 1. Boston 데이터 시각화 실습 문제
#
# 1.1 boston_train.csv 파일을 읽어와서 log plot을 출력하세요
# 'CRIM'과 'MEDV' 컬럼의 데이터를 사용한다

# 한글패치
import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')
df = df.groupby('MEDV').agg(np.mean)
# plt.hist(df)
MEDV = df.index.values
CRIM = df['CRIM'].values
poly = np.polyfit(MEDV,np.log(CRIM),deg=1)
plt.semilogy(MEDV,CRIM,'o')
# plt.semilogy(MEDV,CRIM)
plt.semilogy(MEDV,np.exp(np.polyval(poly,MEDV)))
plt.show()


# 1.2 boston_train.csv 파일을 읽어와서 scatter plot을 출력하세요
# # 'CRIM'과 'MEDV','ZN' 컬럼의 데이터를 사용한다

df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')
df = df.groupby('MEDV').agg(np.mean)
gpu = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')
gpu = gpu.groupby('MEDV').agg(np.mean)
gpu
df2 = pd.merge(df,gpu,how='outer',left_index=True,right_index=True)
df.mean()

MEDV = df.index.values
CRIM = df['CRIM'].values
ZN = df['ZN'].values
zz = np.log(CRIM)
zz
plt.scatter(MEDV,zz)
plt.show()

plt.scatter(MEDV,cnt_s = ZN)
plt.show()


#  1.4 boston_train.csv파일을 읽어와서 3차원 plot을 출력하세요
df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')
from mpl_toolkits.mplot3d.axes3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
df.columns

fig = plt.figure()
ax = Axes3D(fig)

x = df['CRIM'].values
y = np.where(df['MEDV'].values>0,
             np.log(df['MEDV'].values),0)
z = np.where(df['ZN'].values>0,
             np.log(df['ZN'].values),0)

x,y = np.meshgrid(x,y)
z,_=np.meshgrid(z,0)

ax.plot_surface(x,y,z)
ax.set_xlabel('CRIM')
ax.set_ylabel('MEDV')
ax.set_zlabel('ZN')
ax.set_title("재광's")
plt.show()


# 1.5 boston_train.csv파일을 읽어와서 지연 plot을 출력하세요
df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')

# 1) 풀이
from pandas.plotting import lag_plot
lag_plot(np.log(df['MEDV']))
lag_plot(np.log(df['CRIM']))
plt.show()

# 2) 풀이
lag_plot(np.log(df['MEDV']))
plt.show()


# 1.6 boston_train.csv 파일을 읽어와서 자기 상관 plot을 출력하세요
df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(np.log(df['MEDV']))
plt.show()


#   1.7 boston_train.csv 파일을 읽어와서 pandas를 사용해서
#   전체 컬럼의 box plot과 'TAX' 컬럼의 box plot을 출력하세요

import numpy as np

df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')

df = np.log(df.values)
df.plot.box()
plt.show()


plt.boxplot(df)



# =======================강사 답안===========================================================

# matplotlib_실습문제.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('C:/Users/CPB06GameN/PycharmProjects/GitHub/bigdata/bigdata/파이썬빅데이터분석/boston_train.csv')



# 1.1번 로그플롯
crim = df['CRIM'].values
medv = df['MEDV'].values

poly = np.polyfit(crim,np.log(medv),deg=1) # 학습
print(type(poly))
print('Poly',poly[0]) # W, 기울기
print('Poly',poly[1]) # b, y절편
# plt.plot(crim,np.log(medv),'o')
# plt.show()
plt.semilogy(crim,medv,'o')
plt.semilogy(crim,np.exp(np.polyval(poly,crim)))
plt.title('1.1 Boston crim/zn medv scatter plot')

plt.show()
print(df.corr())




# 1.2번 분산 플롯
crim = df['CRIM'].values
medv = df['MEDV'].values
zn = df['ZN'].values

# c: color, s:size, apha:투명도
plt.scatter(crim,medv,c = 200*crim,
            s =20 + 200*zn/zn.max(),
            alpha = 0.5)  # 버블차트

plt.grid(True)
plt.xlabel('crim')
plt.ylabel('medv')
plt.title('1.2 Boston crim/zn medv scatter plot')
plt.show()



# 1.3 번
crim = df['CRIM'].values
medv = df['MEDV'].values

poly = np.polyfit(crim,np.log(medv),deg=1) # 학습
plt.plot(crim, np.polyval(poly, crim), label='Fit')

medv_start = int(medv.mean())
print(medv_start )
y_ann = np.log(df.at[medv_start, 'MEDV']) - 0.1
print(y_ann)
ann_str = "Medv Crime\n %d" % medv_start
plt.annotate(ann_str, xy=(medv_start, y_ann),
             arrowprops=dict(arrowstyle="->"),
             xytext=(-30, +70), textcoords='offset points')

cnt_log = np.log(medv)
plt.scatter(crim, cnt_log, c= 200 * crim,
            s=20 + 200 * zn/zn.max(),
            alpha=0.5, label="Scatter Plot")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel("Crime")
plt.ylabel("Medv", fontsize=16)
plt.title("1.3 Boston Housing : Crime Medv")
plt.show()


# ==============================================================

# titanic실습과제.py

# https://kaggle-kr.tistory.com/17

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

# 한글 출력을 위한 설정
import matplotlib
matplotlib.rcParams['font.family']="Malgun Gothic"
matplotlib.rcParams['axes.unicode_minus'] = False

titanic = sb.load_dataset('titanic')


# 한글 출력을 위한 설정
import matplotlib
matplotlib.rcParams['font.family']="Malgun Gothic"
matplotlib.rcParams['axes.unicode_minus'] = False

titanic = sb.load_dataset('titanic')

# 2.1 생존자와 사망자의 수를 pie 차트로 그리시오(matplotlib)

survived = titanic['survived'][titanic['survived'] == 1].count()

not_survived = titanic['survived'][titanic['survived'] ==0].count()
data = [survived,not_survived]
pie_label = ['생존자','사망자']
exp = [0.05,0.05]
plt.figure(figsize=(5,5))
plt.pie(data,labels = pie_label,explode = exp,
        autopct ='%.1f%%',shadow = True)
plt.title('2.1 Titanic Survived - Pie Chart')
plt.show()


# 2.2 등급별 티켓 비용(fare)의 평균을  barplot으로 그리시오(seaborn)
sb.barplot('pclass','fare', data=titanic)
plt.title('2.2 Titanic pclass/fare - barplot')
plt.show()

# 2.3 성(Sex)별 생존자와 사망자의 수를 countplot 으로 그리시오(seaborn)
sb.countplot(data=titanic,x='sex',hue='survived')
plt.title('2.3 Titanic Survived/sex - countplot')
plt.show()


# 2.4  상관 관계 heatmap을 출력하시요 (seaborn)
plt.figure(figsize=(10, 10))
sb.heatmap(titanic.corr(), linewidths=0.01, square=True,
            annot=True, cmap=plt.cm.viridis, linecolor="white")
plt.title('2.4 Titanic Correlation between features')
plt.show()

# 2.5 나이(age) 분포도(distplot)를 그리시오 (seaborn)
#       결측치는 평균값으로 수정
titanic = titanic.fillna(titanic.mean())
sb.distplot(titanic['age'])
plt.title('2.5 Titanic age - distplot')
plt.show()


# 2.6 객실의 등급(pclass)별 'age'의 분포를 boxplot으로 그리시오(seaborn)
titanic = sb.load_dataset('titanic')

sb.boxplot(data=titanic, x='pclass',y='age')

df = titanic.groupby('pclass')
med = df.agg([np.median])
r = med['age']
t0 = r['median'].values[0]
t1 = r['median'].values[1]
t2 = r['median'].values[2]
plt.text(0,t0, round(t0,2))
plt.text(1,t1, round(t1,2))
plt.text(2,t2, round(t2,2))

plt.title('2.6 Titanic pclass/age - boxplot')
plt.show()