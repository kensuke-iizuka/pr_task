# -*- coding: utf-8 -*-

# EMアルゴリズム（混合正規分布(二変量)）

import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# 正規分布の個数
K = 4

# 特徴数（このプログラムはD=2のみ）
D = 2

# 正規分布一つあたりのデータ数
P = 50

# 全データ数
N = K*P

# 正規分布（多変量）
def mnd(x, ave, cov):
    a = np.sqrt( np.linalg.det(cov)*(2*np.pi)**cov.ndim)
    b = np.reshape( x-ave , (D,1) )
    bT = np.reshape( x-ave , (1,D) )
    c = -0.5 * np.dot( np.dot( bT , np.linalg.inv(cov) ) , b )
    return np.exp(c)/a

#平均，分散共分散行列
average = np.zeros( (K,D) , dtype=np.float32 )
cov = np.zeros( (K,D,D) , dtype=np.float32 )

# 平均，標準偏差の設定
for k in range(K):
    for d in range(D):
        a = (np.random.rand() - 0.5) * 20.0
        average[k][d] = a

    for i in range(D):
        for j in range(D):
            s = np.random.randint( 1 , 5 )
            if i == j:
                cov[k][i][j] = s

# データ（全データ数，特徴数）
data = np.zeros( (N,D) , dtype=np.float32 )

# データの生成
for k in range(K):
    for p in range(P):
        data[k*P+p][0],data[k*P+p][1] = np.random.multivariate_normal(average[k],cov[k]).T

# グラフの描画
plt.figure()

# 散布図をプロットする
for n in range(N):
    plt.scatter(data[n][0],data[n][1],color="red")

# ラベル
plt.xlabel("x",size=10)
plt.ylabel("y",size=10)
plt.title( "MND" )

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)

# グラフの表示
plt.show()
plt.clf()

# 初期化(Q値，平均，分散共分散，重み）
Q = np.zeros((N, K),dtype=np.float64)
e_average = np.zeros((K,D),dtype=np.float64)
e_cov = np.zeros((K,D,D),dtype=np.float64)
e_lamda = np.zeros(K,dtype=np.float64)

for k in range(K):
    e_average[k] = random.choice(data)
    e_cov[k] = (np.cov(data , rowvar=0 , bias=1) )
    e_lamda[k] = 1/K

print( "平均 -> " , e_average )
print( "分散 -> " , e_cov )
print( "重み -> " , e_lamda )

# EMアルゴリズムにより，パラメータQ値（Q），平均ベクトル
#（e_average），分散共分散行列（e_cov），重み（e_lamda）
# を推定しなさい
# EMアルゴリズム
for loop in range(LOOP):

    # E-step
    for i in range(N):
        temp = np.zeros(K,dtype=np.float64)
        sum_t = 0
        for k in range(K):
            temp[k] = st.norm.pdf( data[i] , e_average[k] , e_sigma[k] )
            sum_t += ( temp[k] * e_lamda[k] )

        for k in range(K):
            Q[i][k] = ( temp[k] * e_lamda[k] ) / sum_t

    # M-step
    # 重みの更新
    for k in range(K):
            sum_Q = 0
            for i in range(N):
                sum_Q += Q[i][k]
            e_lamda[k] = sum_Q / N
    print( loop , ":lamda -> " , e_lamda )

    # 平均値の更新
    new_average = np.zeros(K,dtype=np.float32)
    for k in range(K):
        sum_q = 0
        sum_q1 = 0
        for i in range(N):
            sum_q += Q[i][k] * data[i]
            sum_q1 += Q[i][k]
        new_average[k] = sum_q / sum_q1
    print( loop , ":average -> " , new_average )

    # 標準偏差の更新
    new_sigma = np.zeros(K,dtype=np.float32)
    for k in range(K):
        sum_q = 0
        sum_q1 = 0
        for i in range(N):
            sum_q += Q[i][k] * (data[i]-e_average[k])**2
            sum_q1 += Q[i][k]
        new_sigma[k] = np.sqrt( sum_q / sum_q1 )
    print( loop , ":sigma -> " , new_sigma )
    print( " ----- " )

    # 平均値，標準偏差を更新
    e_average = new_average.copy()
    e_sigma = new_sigma.copy()




# Q値が最大のクラスターを表示
for i in range(N):
    ans = np.argmax( Q[i,:] )
    print( i , ans )

# グラフの描画
plt.figure()

plt.subplot(2,1,1)

# 散布図をプロットする
plot_c = [ "red" , "orange" , "blue" , "green" , "pink" ]
plot_l = [ "class-1" , "class-2" , "class-3" , "class-4" , "clss-5" ]
for k in range(K):
    plt.scatter(data[k*P:(k+1)*P,0],y=data[k*P:(k+1)*P,1],color=plot_c[k],label=plot_l[k])

for k in range(K):
    plt.figtext((average[k][0]+20)/40,(average[k][1]+20)/40/2+0.5,plot_l[k],size=10)

# ラベル
plt.xlabel('x',size=10)
plt.ylabel('y',size=10)

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)

# 散布図をプロットする
for i in range(N):
    ans = np.argmax( Q[i,:] )
    plt.scatter(data[i][0],data[i][1],color=plot_c[ans])

for k in range(K):
    plt.figtext((e_average[k][0]+20)/40,(e_average[k][1]+20)/40/2,plot_l[k],size=10)

# ラベル
plt.xlabel('x',size=10)
plt.ylabel('y',size=10)

# 軸
plt.axis([-20.0,20.0,-20.0,20.0])
plt.grid(True)

# 保存
plt.savefig("EM-result.png")
plt.show()
plt.clf()
