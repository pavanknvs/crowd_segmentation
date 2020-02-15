import cv2 as cv
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#for dimensions of video
file_path = 'E:\crowd.scene.analasys\h264\crowd001.avi'
vid = cv.VideoCapture(file_path)
h =int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))

p= 1
h1=int(h/16)
w1=int(w/16)
br=0
se  = np.arange(w1*h1).reshape(h1,w1)
for i in range(h1):
    for j in range(w1):
        se[i][j] =0

z1 = list()
z2 = list()
oreo = 0
t = 0
t1 = 0
#for    finding    threshold   value

fhand = open('E:\crowd.scene.analasys\h264\crowd001.txt')
for line in fhand:

             ui = 0
             words = (line.replace(',',' ')).split()
             k1 =  int(words[6]) - int(words[4])
             k2 =  int(words[7]) - int(words[5])
             if words[1]=='-1' and  (k1!= 0 or k2 != 0) :
                 j = int(words[4])
                 i = int(words[5])
                 j=int(j/16)
                 i=int(i/16)
                 if i  < h1 and j  < w1:
                    se[i][j] =  se[i][j]  + 1




for i in range(h1):
    for j in range(w1):
        t = se[i][j]+ t



t = int(t/(h1*w1))
print(br)
print('threshold    value   is  ',t)
t =int(input('enter    threshold    value   if u   want    to  change   '))
print('after    changing    threshold    value   is  ',t)
#print(se)
#####################################################################################################


#for    finding  mask   before  morphological   operation

for i in range(h1):
    for j in range(w1):
        if se[i][j] >= t :
            for r3 in range(16):
                for r4 in range(16):
                    z1.insert(oreo, 16* i + r3)
                    z2.insert(oreo, 16 * j + r4)
                    oreo = oreo + 1


fig=plt.rcParams['figure.figsize'] = (w/100,h/100)
plt.axis([0,w-1,h-1,0])
plt.scatter(z2,z1)
plt.subplots_adjust(0,0,1,1,0.2,0.2)
plt.savefig('E:\crowd.scene.analasys\h264\pew12\pro'+str(p)+'mv.png')
plt.show()
#############################################################################


#applying   morphological   operation   on  mask

img = cv.imread('E:\crowd.scene.analasys\h264\pew12\pro'+str(p)+'mv.png', 0)
kernel = np.ones((6,6),np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
erosion = cv.erode(closing,kernel,iterations = 1)
opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel)
cv.imshow('image',opening)
cv.imwrite('E:\crowd.scene.analasys\h264\pew12\pro'+str(p)+'mvmask.png',opening)
ref1 = cv.imread('E:\crowd.scene.analasys\h264\pew12\pro'+str(p)+'mvmask.png',-1)
cv.waitKey(0)&0xFF
cv.destroyAllWindows()
#####################################################################################


#for    extracting  motion  vectors in  the mask(dynamic    region)
fhand1 = open('E:\crowd.scene.analasys\h264\crowd001.txt','r')
f = open("E:\crowd.scene.analasys\h264\pew12\pro"+str(p)+"mvtp.txt", "w")
for line1 in fhand1:

             ui = 0
             words2 = (line1.replace(',',' ')).split()
             k1 = int(words2[6]) - int(words2[4])
             k2 = int(words2[5]) - int(words2[7])
             j = int(words2[4])
             i = int(words2[5])
             if words2[1] == '-1' and (k1 != 0 or k2 != 0) :
                if i < h and j < w:
                 if ref1[i-1][j-1] < 255 :
                     f.write(line1)

######################################################################################
print(br)
#finding    average motion  vector  field
ser  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        ser[i][j] =0

w8  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        w8[i][j] =0
w2  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        w2[i][j] =0
bx  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        bx[i][j] =0

by  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        by[i][j] =0


z6 = list()
z7 = list()
l1 = list()
l2 = list()



c12 = list()
c13= list()
c14 = list()
c15 = list()
c16 = list()
c17 = list()


d12 = list()
d13 = list()
d14 = list()
d15 = list()
d16 = list()
d17 = list()

c19 = list()
d19 = list()
c20 = list()
d20 = list()
c21 = list()
d21 = list()


fmax= 0
fhand = open('E:\crowd.scene.analasys\h264\pew12\pro'+str(p)+'mvtp.txt')
for line in fhand:

             words = (line.replace(',',' ')).split()
             k1 =  int(words[6]) - int(words[4])
             k2 =  int(words[5]) - int(words[7])
             if words[1]=='-1' and  (k1!= 0 or k2 != 0) :
                 j = int(words[4])
                 i = int(words[5])
                 if i < h and j < w:
                                ser[i-1][j-1] =  ser[i-1][j-1]  + 1
                                w8[i-1][j-1] = w8[i-1][j-1] + k1
                                w2[i-1][j-1] = w2[i-1][j-1] + k2
                                fmax= int(words[0])


for i in range(h):
    for j in range(w):
        if ser[i][j]==0:
            ser[i][j]=1




t1=input('enter    the no  of  frames  to  keep    it  as  threshold:')

ft=int(t1)
print(ft)
for i in range(h):
    for j in range(w):
        if ser[i][j]>ft:
             w8[i][j]=int(100*(w8[i][j]/ser[i][j]))
             w2[i][j]=int(100*(w2[i][j]/ser[i][j]))
        else:
            w8[i][j] = 0
            w2[i][j] = 0
        bx[i][j]=int(w8[i][j]/ser[i][j])
        bx[i][j] = int(w2[i][j] / ser[i][j])

oreow=0

for i in range(h):
    for j in range(w):
        if w2[i][j]!=0 or w8[i][j]!=0 :

                    l1.insert(oreow,j)
                    l2.insert(oreow,i)
                    z6.insert(oreow,w8[i][j])
                    z7.insert(oreow,w2[i][j])
                    oreow = oreow + 1


fig, ax = plt.subplots()
ax.quiver(l1, l2, z6, z7, scale=30,color='y')
ax.axis([0, w-1,h-1,0])
fig.subplots_adjust(0,0,1,1,0.2,0.2)
plt.savefig('E:\crowd.scene.analasys\h264\pew12\image'+str(p)+'.png')
plt.show()
#####################################################################

#clustering     motion  vectors in  average motion  vector  field   using   dbscan
print(oreow)

w891= np.arange((oreow)*4).reshape((oreow,4))
w892= np.arange((oreow)*2).reshape((oreow,2))
for i in range(oreow):

        w891[i][0]=l1[i]
        w891[i][1] = l2[i]
        w891[i][2] = z6[i]
        w891[i][3] = z7[i]
        w892[i][0] = z6[i]
        w892[i][1] = z7[i]


k= -999
gen=pairwise_distances(w892,metric='cosine')
db = DBSCAN(eps=1.5, min_samples=10).fit(gen)
labels = db.labels_
for i in range(oreow):
   if   labels[i]>k:
       k= labels[i]


print('no   of  clusters    before  merging similar clusters',k+1)


o12= 0
o13= 0
o14= 0
o15= 0
o16= 0
o17= 0
o18= 0
o19= 0
o20= 0


###########################################################################################3

#merging    similar clusters
w8923= np.arange((k+1)*3).reshape((k+1,3))
for i in range(k+1):
    w8923[i][0]=0
    w8923[i][1] = 0
    w8923[i][2] = 0


lab= np.arange((k+1)*1).reshape((k+1,1))
for i in range(k+1):
    lab[i]=i

#x1=0



for i in range(oreow):
    if  labels[i]>-1:
       w8923[labels[i]][0] = w8923[labels[i]][0] +  w891[i][2]
       w8923[labels[i]][1] = w8923[labels[i]][1] +  w891[i][3]
       w8923[labels[i]][2] = w8923[labels[i]][2] + 1


for i in range(k+1):
    w8923[i][0] = int(w8923[i][0] / w8923[i][2])
    w8923[i][1] = int(w8923[i][1] / w8923[i][2])


print(w8923)


w89231= np.arange((k+1)*3).reshape((k+1,3))
for i in range(k+1):
    w89231[i][0]=0
    w89231[i][1] = 0
    w89231[i][2] = 0

for i in range(oreow):
 if labels[i] > -1:
    w89231[labels[i]][0] = w89231[labels[i]][0] + w891[i][0]
    w89231[labels[i]][1] = w89231[labels[i]][1] + w891[i][1]
    w89231[labels[i]][2] = w89231[labels[i]][2] + 1



for i in range(k + 1):
    w89231[i][0] = int(w89231[i][0] / w89231[i][2])
    w89231[i][1] = int(w89231[i][1] / w89231[i][2])



print(w89231)

for i in range(k+1):
    for j in range(i+1,k + 1):
        if  abs(w89231[i][0]-w89231[j][0])<=w/4   and abs(w89231[i][1]-w89231[j][1])<=h/4 :
            v1 = w8923[i][0] * w8923[i][0] + w8923[i][1] * w8923[i][1]
            v2 = w8923[j][0] * w8923[j][0] + w8923[j][1] * w8923[j][1]
            v3 = math.sqrt(v1)
            v4 = math.sqrt(v2)
            v5 = v3 * v4
            v6 = w8923[i][0] * w8923[j][0] + w8923[i][1] * w8923[j][1]
            v7 = v6 / v5

            if  v7  >=0:
                #print(w89231[i][0],w89231[i][1],w89231[j][0],w89231[j][1])
                if  lab[j] == j:
                    lab[j]=lab[i]




for i in range(k+1):
    for j in range(oreow):
        if  labels[j]==i:
         labels[j] =lab[i]



for i in range(oreow):
   if   labels[i]==-1:
       c12.insert(o12,w891[i][0])
       d12.insert(o12,w891[i][1])
       o12=o12+1
   elif labels[i] == 0:
       c13.insert(o13, w891[i][0])
       d13.insert(o13, w891[i][1])
       o13 = o13 + 1
   elif labels[i] == 1:
       c14.insert(o14, w891[i][0])
       d14.insert(o14, w891[i][1])
       o14 = o14 + 1
   elif labels[i] == 2:
       c15.insert(o15, w891[i][0])
       d15.insert(o15, w891[i][1])
       o15 = o15 + 1
   elif labels[i] == 3:
       c16.insert(o16, w891[i][0])
       d16.insert(o16, w891[i][1])
       o16 = o16 + 1
   elif labels[i] == 4:
       c17.insert(o17, w891[i][0])
       d17.insert(o17, w891[i][1])
       o17 = o17 + 1
   elif labels[i] == 5:
       c19.insert(o18, w891[i][0])
       d19.insert(o18, w891[i][1])
       o18 = o18 + 1
   elif labels[i] == 6:
       c20.insert(o19, w891[i][0])
       d20.insert(o19, w891[i][1])
       o19 = o19 + 1
   elif labels[i] == 7:
       c21.insert(o20, w891[i][0])
       d21.insert(o20, w891[i][1])
       o20 = o20 + 1



fig=plt.rcParams["figure.figsize"] = (w/100,h/100)
plt.axis([0,w,h,0])
plt.subplots_adjust(0,0,1,1,0.2,0.2)
#plt.scatter(c12,d12,color='black',s=5)
plt.scatter(c13,d13,color='yellow')
plt.scatter(c14,d14,color='red')
plt.scatter(c15,d15,color='green',s=25)
plt.scatter(c16,d16,color='tan',s=25)
plt.scatter(c17,d17,color='blue',s=25)
plt.scatter(c19,d19,color='gold',s=25)
plt.scatter(c20,d20,color='orange',s=25)
plt.scatter(c21,d21,color='grey',s=25)
plt.savefig('E:\crowd.scene.analasys\h264\pew12\merge'+str(p)+'.png')
plt.show()

#############################################################################
k=-999
k8=0
for i in range(oreow):
   if   labels[i]>k:
        k= labels[i]
        print(k)
        k8=k8+1
print('no   of  clusters    after   merging',k8)



sew  = np.arange(w*h).reshape(h,w)
for i in range(h):
    for j in range(w):
        sew[i][j] =0
for i in range(oreow):

    if labels[i] == 1   and sew[w891[i][1]][w891[i][0]]==0:
        r1 = 0
        n1=w891[i][1]
        n2=w891[i][0]
        while r1 < 16:
                r2 = 0
                while r2 <  16:
                    if (n1 + r1 < h and n2 + r2 < w):
                        if sew[w891[i][1] + r1][w891[i][0] + r2] == 0:
                            sew[w891[i][1]+ r1][w891[i][0]+ r2]=30
                    r2 = r2 + 1
                r1 = r1 + 1
for i in range(oreow):

    if labels[i] == 0 and sew[w891[i][1]][w891[i][0]]==0:
        r1 = 0
        n1 = w891[i][1]
        n2 = w891[i][0]
        while r1 < 16:
            r2 = 0
            while r2 < 16:
                if ( n1+ r1<h   and  n2+ r2<w):
                    if  sew[w891[i][1]+ r1][w891[i][0]+ r2]==0 :
                        sew[w891[i][1] + r1][w891[i][0] + r2] = 130
                r2 = r2 + 1
            r1 = r1 + 1

    elif labels[i] == -1 and sew[w891[i][1]][w891[i][0]]==0:
        r1 = 0
        n1 = w891[i][1]
        n2 = w891[i][0]
        while r1 < 16:
            r2 = 0
            while r2 < 16:
                print(n1+ r1,n2+ r2)
                if (n1 + r1 < h and n2 + r2 < w):
                    if sew[w891[i][1] + r1][w891[i][0] + r2] == 0:
                        sew[w891[i][1] + r1][w891[i][0] + r2] = 255
                r2 = r2 + 1
            r1 = r1 + 1

for i in range(h):
    for j in range(w):
        if sew[i][j]  == 0:
            sew[i][j]=255


fig=plt.rcParams['figure.figsize'] = (w/100,h/100)
plt.axis([0,w-1,h-1,0])
plt.subplots_adjust(0,0,1,1,0.2,0.2)
plt.imshow(sew)
plt.savefig('E:\crowd.scene.analasys\h264\pew12\yfin'+str(p)+'.png')
plt.show()
img5 = cv.imread('E:\crowd.scene.analasys\h264\pew12\merge'+str(p)+'.png')
kernel = np.ones((12,12),np.uint8)
closing1 = cv.morphologyEx(img5, cv.MORPH_OPEN, kernel)
cv.imwrite('E:\crowd.scene.analasys\h264\pew12\yfinal1'+str(p)+'.png',closing1)
cv.imshow('image',closing1)
plt.show()