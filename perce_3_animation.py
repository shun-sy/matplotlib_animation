#coding:utf-8
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
from matplotlib import animation as ani
from moviepy.editor import *

#学習係数
epsilon = 0.1
#学習回数
Num_Learn = 2000
#各層のユニット数
Num_input = 1
Num_hidden = 5 
Num_output = 1

#今回回帰が目的なので出力層は恒等写像
#活性化関数にはロジスティックシグモイド関数を使用
def logistic(u):
    return 1.0/(1.0 + np.exp(-1*u))
def d_logistic(u):
    return logistic(u)*(1-logistic(u))

def tanh(u):
    return np.tanh(u)
def d_tanh(u):
    return 1 - np.tanh(u)*np.tanh(u)

def out_put(x,w1,w2,b1,b2):
    z = np.zeros(Num_hidden)
    y = np.zeros(Num_output)
    z = np.dot(w1,x) + b1
    z = tanh(z)
    y = np.dot(w2,z) + b2
    return y

#描画アニメーション準備
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(-1,1)
ax.set_xlim(-3,3)
ax.grid()

d_func, = ax.plot([],[],'bo',lw=2)
p_func, = ax.plot([],[],'r-',lw=2)


#サンプル数
N = 50
#学習用データの作成
xlist = np.linspace(-3,3,N).reshape(N,1)
#dlist = xlist * xlist
dlist = np.sin(xlist)

#重み初期化
w1 = randn(Num_hidden,Num_input)/(np.sqrt(Num_input))
w2 = randn(Num_output,Num_hidden)/(np.sqrt(Num_input))
#バイアス初期化
b1 = np.array([randn(Num_hidden)]).T
b2 = np.array([randn(Num_output)]).T

def animate(nframe):
    global w1,w2,b1,b2
    for i in range(len(xlist)):

        x = np.array([xlist[i]]).T
 
        z = np.zeros(Num_hidden)
        y = np.zeros(Num_output)

        dy = np.zeros(Num_output)
        dz = np.zeros(Num_hidden)

        #順伝搬計算
        #中間層の計算
        #print(b1)
        z = np.dot(w1,x)+b1
        z = tanh(z)
        #出力層の計算
        y = np.dot(w2,z)+b2
        #逆誤差伝搬で重み修正
        dy = y - dlist[i]
        #print(dy)
        #print(np.dot(w2.T,dy))
        dz = d_tanh(z) * (np.dot(w2.T,dy))
        #print(dz)
        #重み,バイアス更新
        w2 = w2 - epsilon*(np.dot(dy,z.T))
        b2 = b2 - epsilon*(dy)
        w1 = w1 - epsilon*(np.dot(dz,x.T))
        b1 = b1 - epsilon*(dz)

    #描画データをいれる
    ylist = np.zeros((N,Num_output))
    for n in range(N):
        ylist[n] = out_put(np.array([xlist[n]]).T,w1,w2,b1,b2)
    d_func.set_data(xlist,dlist)
    p_func.set_data(xlist,ylist)

print(w1)
print(w2)
print(b1)
print(b2)
#animationさせたい奴 update関数　データ引き渡し関数
anim = ani.FuncAnimation(fig,animate,frames = Num_Learn,blit=False,interval = 100,repeat = False)
#anim.save('multi_perceptron.mp4',fps=13)
plt.show()







