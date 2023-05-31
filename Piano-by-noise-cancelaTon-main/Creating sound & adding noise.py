import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
from scipy import pi

t = np.linspace(0,3,12*1024)
Fi=[0,0,0,0,0,0,0]
fi=[261.63,0,293.66,349.23,392,440,0]
ti=[0,0.4,0.9,1.4,2,2.5,3]
Ti=[0.4,0.5,0.5,0.6,0.5,0.5,0.4]
i=0
s=0
def u(t):
 x=np.reshape([t>0], np.shape(t))
 return x
while i<len(fi):
    a=np.multiply(np.sin(2*np.pi*fi[i]*t),((u(t-ti[i]))^(u(t-ti[i]-Ti[i]))))
    i+=1
    s+=a
plt.plot(t, s)   
#sd.play(s, 3*1024)

𝑁 = 3*1024
𝑓 = np.linspace(0 , 512 , int(𝑁/2))

x_f = fft(s)
x_f = 2/N * np.abs(x_f[0:np.int(N/2)])

#plt.plot(f,x_f)

fn1=np.random.randint(0, 512, 1)
fn2=np.random.randint(0, 512, 1)

n=np.sin(2*(fn1)*(np.pi)*t)+np.sin(2*(fn2)*(np.pi)*t)

sn=s+n

xn_f = fft(sn)
xn_f = 2/N * np.abs(xn_f[0:np.int(N/2)])

#plt.plot(f,xn_f)
#plt.show()

i=0
max1=0
maximum=max(x_f)
max_number1=maximum
for i in range(len(xn_f)) :
    if xn_f[i]>maximum :
       max_number1=xn_f[i]
       max1=i
       xn_f[i]=0
       break
       
       
       
j=0
max2=0

max_number2=maximum
for j in range(len(xn_f)) :
    if xn_f[j]>maximum :
       max_number2=xn_f[j]
       max2=j
       break
       
xn_f[max1]=max_number1
xn_f[max2]=max_number2    

fn1_r =xn_f[np.round(max1)]
fn2_r =xn_f[np.round(max2)]   
    
    
xfiltered=sn-np.sin(2*(fn1)*(np.pi)*t)-np.sin(2*(fn2)*(np.pi)*t)
sd.play(xfiltered, 3*1024)



xFiltered = fft(xfiltered)
xFiltered = 2/N * np.abs(xFiltered[0:np.int(N/2)])




plt.subplot(3,2,1)
plt.plot(t,s)

plt.subplot(3,2,2)
plt.plot(f,x_f)

plt.subplot(3,2,3)
plt.plot(t,sn)


plt.subplot(3,2,4)
plt.plot(f,xn_f)

plt.subplot(3,2,5)
plt.plot(t,xfiltered)

plt.subplot(3,2,6)
plt.plot(f,xFiltered)

sd.play(xfiltered,3*1024)



















