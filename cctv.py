import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 불러오기
df_dong = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_동구_CCTV_20190321.csv')
df_su = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_서구_CCTV_20190705.csv')
df_dae = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_대덕구_CCTV_20190331.csv')
df_u = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_유성구_CCTV_20190520.csv')
df_joong = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_중구_CCTV_20190527.csv')

# 대전으로 합치기
x=df_dong.append(df_su,ignore_index=True)
y =df_dae.append(df_u,ignore_index=True)
z=x.append(y,ignore_index=True)
df_cctv=z.append(df_joong,ignore_index=True)

print(df_cctv)
df_cctv.columns


# ============================================================================================

# cctv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =======================================cctv=================================================================

df_dong = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_동구_CCTV_20190321.csv')
df_su = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_서구_CCTV_20190705.csv')
df_dae = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_대덕구_CCTV_20190331.csv')
df_u = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_유성구_CCTV_20190520.csv')
df_joong = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시_중구_CCTV_20190527.csv')

# df_cctv1 = np.row_stack((df_dong,df_su))
# df_cctv2 = np.row_stack((df_dae,df_yoo))
# df_cctv3 = np.row_stack((df_cctv2,df_joong))
# df_cctv = np.row_stack((df_cctv1,df_cctv3))

x=df_dong.append(df_su,ignore_index=True)
y =df_dae.append(df_u,ignore_index=True)
z=x.append(y,ignore_index=True)
df_cctv=z.append(df_joong,ignore_index=True)

print(df_cctv)
df_cctv.columns
df_cctv.rename(columns={df_cctv.columns[0]:'대전지역별'},inplace=True)
print(df_cctv.head())
print(df_cctv.tail())

df_cctv = df_cctv.drop(columns=['설치목적구분','카메라화소수','촬영방면정보','보관일수','설치년월','관리기관전화번호'])
df_cctv = df_cctv.columns
df_cctv.sort_values(['카메라대수'],ascending=True)
print(df_cctv)


# 구별 cctv 현황을 보자
df_cctv_count=df_cctv.drop(columns=['소재지도로명주소','소재지지번주소','위도', '경도', '데이터기준일자'])
print(df_cctv_count)
df_cctv_count = df_cctv_count.groupby('대전지역별').sum()
print(df_cctv_count)
df_cctv_count=df_cctv_count.rename(index={'대전광역시 대덕구청':'대덕구','대전광역시 서구청':'서구',
                            '대전광역시 동구':'동구','대전광역시 중구청':'중구',
                            '유성구청': '유성구'})
print(df_cctv_count)


# =================================================================================================================



# =============================================범죄율============================================================
df_crim = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/대전광역시지방경찰청_자치구별 5대범죄 발생 현황(2018).csv')
df_crim.columns
df_crim_dong = df_crim[df_crim.columns[2]][0:10]
print(df_crim_dong)
df_crim_dong=sum(df_crim_dong)
print(df_crim_dong)

df_crim_joong = df_crim[df_crim.columns[2]][0:5]
print(df_crim_joong)
df_crim_joong=sum(df_crim_joong)

df_crim_su = df_crim[df_crim.columns[2]][5:15]
print(df_crim_su)
df_crim_su=sum(df_crim_su)

df_crim_dae = df_crim[df_crim.columns[2]][15:20]
print(df_crim_dae)
df_crim_dae=sum(df_crim_dae)

df_crim_yoo = df_crim[df_crim.columns[2]][20:]
print(df_crim_yoo)
df_crim_u=sum(df_crim_yoo)
df_crim_u = df_crim_yoo
# ===============================================================================================================
# import folium as fm
# map_osm = fm.Map(location=[35.194012, 128.101959], zoom_start = 17)
# fm.Marker([35.194012, 128.101959], popup='대전').add_to(map_osm)


# ============================================인구율=============================================================
df_pop = pd.read_csv('C:/Users/CPB06GameN/Desktop/cctv/2019년 6월말 주민등록현황(구별-동별-연령별).csv')
df_pop.columns
dong_pop=df_pop['Unnamed: 1'][4]
joong_pop=df_pop['Unnamed: 1'][21]
su_pop=df_pop['Unnamed: 1'][39]
u_pop=df_pop['Unnamed: 1'][63]
dae_pop=df_pop['Unnamed: 1'][75]

dong_pop=dong_pop.replace(',','')
joong_pop=joong_pop.replace(',','')
su_pop=su_pop.replace(',','')
u_pop=u_pop.replace(',','')
dae_pop=dae_pop.replace(',','')

# ======================================데이터 합치기================================================
df_cctv_count['지역별 인구수'] = dae_pop, dong_pop,su_pop,joong_pop,u_pop
print(df_cctv_count)
df_cctv_count['지역별 범죄건수'] = df_crim_dae,df_crim_dong,df_crim_su,df_crim_joong,df_crim_u
print(df_cctv_count)
df_cctv_count=df_cctv_count.drop(columns='지역별 범죄율')
print(df_cctv_count)

df_cctv_count

df_cctv_count['지역별 인구수'] = df_cctv_count['지역별 인구수'].astype(np.int64)  # object 을 int로 바꾸기

df_cctv_count.dtypes

# ======================================파생 컬럼 추가=============================================
df_cctv_count['인구 대비 카메라'] = df_cctv_count['카메라대수']/df_cctv_count['지역별 인구수']
df_cctv_count
df_cctv_count['인구 대비 범죄'] = df_cctv_count['지역별 범죄건수']/df_cctv_count['지역별 인구수']
df_cctv_count
# 인구대비 카메라와 인구대비 범죄
df_cctv_count['cctv와 범죄'] = df_cctv_count['인구 대비 카메라']/df_cctv_count['인구 대비 범죄']
print(df_cctv_count)
# =======================================시각화====================================================
import seaborn as sns

import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# 지역별 인구대비 범죄
plt.figure(figsize=(8,6))
sns.barplot(x=df_cctv_count.index.values, y="인구 대비 범죄", data=df_cctv_count)
plt.grid
plt.title('지역별 인구대비 범죄')
plt.show()


# 지역별 인구대비 cctv
plt.figure(figsize=(8,6))
sns.barplot(x=df_cctv_count.index.values, y="인구 대비 카메라", data=df_cctv_count)
plt.grid
plt.title('지역별 인구대비 cctv')
plt.show()


# 상관관계 파악
df_cctv_count.corr()

print(df_cctv_count)


# ======================================지도 위에 뿌려보기======================================
import folium
folium.Map(location=[35,127])


