# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
import random as rand
import math
h=1
m=0
salpha=0.4
sbeta=0.6
alpha=0.4
beta=0.6
sigma=-100
velocity=40
rand.seed(3)
matrix=np.zeros([269,269],dtype=float)
#print(matrix)
ph=math.ceil(rand.uniform(1,1000))
ph=math.ceil(10*math.log10(ph))
print("ph value",ph)
m=0
pf=707 #considered constant velocity of 10 m/s
pf=math.ceil(10*math.log10(pf))+30
print("flight energy is",pf)
#pmax=200*pow(10,-6)
pmax=math.ceil(rand.uniform(1,100))
#pmax=200
#pmax=math.ceil(10*math.log10(pmax))
print("pmax",pmax)
v={}
l=[]
GTs=math.ceil(rand.uniform(1,50))
print("NUmber of GTs",GTs)
Buav=math.ceil(rand.uniform(1000,900000000000000000000000000))
#Buav=1
#print("BUAV",Buav)
#Buav=3000*pow(10,3)
Buav=math.ceil(10*math.log10(Buav))
print("BUAV is",Buav)
actions=[]
Plevels=math.ceil(rand.uniform(2,10))
print("plevels",Plevels)
grid=100
Pag=np.empty(GTs)
xcord = np.zeros(GTs+1)
for i in range(xcord.size):
  xcord[i]= rand.uniform(0,grid)
print("\n")

ycord = np.zeros(GTs+1)
for i in range(ycord.size):
  ycord[i]= rand.uniform(0,grid)
dist=[[0 for i in range(GTs+1)] for j in range(GTs+1)]

for i in range(GTs+1):
  for j in range(GTs+1):
    dist[i][j]=math.ceil(math.sqrt((xcord[i]-xcord[j])**2 + (ycord[i]-ycord[j])**2))

d=dist
"""
maxdiffdist=[]
for i in dist[0]:
    maxdiffdist.append(max(i-40,0))
#print("dist=",d)
print("Distance of each GT from home is",maxdiffdist)
"""
print("distance",dist[0])
for i in range(Buav,-1,-1):
  for j in range(GTs+1):
    for k in range(2):
      l.append((i,j,k))


##### Creating a dictionary  #######################


for i in l:
  if(i[0]==0):
    v[i]=0
  else:
    v[i]=0


########### Finding probability of LOS for all areas ################


areas=[(0.1,750,8),(0.3,500,15),(0.5,300,20),(0.5,300,50)]
dangle=90
rangle=1.5708
problos=[]
t=0
for i in areas:
  avalue=i[0]*i[1]
  bvalue=i[2]
  problos.append(1/(1+(avalue*np.exp(-bvalue*(dangle-avalue)))))
  t=t+1




#########  Finding Expected fading values for all AREAS ##########



freq=2000
diag=[]
height=[1400,2100,2100,1750]
radius=[3300,2200,1700,600]
for i in range( len(height)):
	diag.append(math.sqrt(height[i]*height[i]+radius[i]*radius[i]))
eta=[(0.1,21),(1.0,20),(1.6,23),(2.3,34)]
c=3*10^8
Egamma=[]
t=0
for i in eta:
  Plos=20*math.log10(diag[t])+20*math.log10(freq)+20*math.log10((4*math.pi)/c)+i[0]
  PNlos=20*math.log10(diag[t])+20*math.log10(freq)+20*math.log10((4*math.pi)/c)+i[1]
  Egamma.append(problos[t]*Plos+(1-problos[t])*PNlos)
  t=t+1
print("fading of different areas",Egamma)
f = np.empty(GTs)
for i in range(f.size):
    f[i]=rand.choice(Egamma)
print("fading:",f)


print("----------------fading-----------------")

k=0
for i in f:
    print(i,k+1)
    k=k+1


######## Finding various power levels for each action ###########

power=[0]
pow=[]
pow=np.empty(Plevels+1)
power[0]=ph
pow[pow.size-1]=pmax
#step=pmax/Plevels
"""
#newly added
for i in range(0,pmax-1,3):
    pow.append(i)
pow.append(pmax)
"""
for i in range(0,pow.size-1):
  temppow=[]
  temppow=rand.sample(range(1,pmax),1)
  pow[i]=temppow[0]
pow.sort()
print("Power is",pow)
for i in range(0,pow.size):
	#print(power[i])
	pow[i]=math.ceil(10*math.log10(pow[i]))
#print("power before ordereddict:",pow)
pow=list(OrderedDict.fromkeys(pow))
#print("power after dbm",pow)
power.extend(pow)
print("final power is",power)
size=len(power)

totalactions=GTs+size+1
for i in range(totalactions):
  actions.append(i)
print("actions size",len(actions))
##################### Computing Reward ##########################


temp={}
def compute(x,a,m):
  retreward=funreward(x,a)

  retvalue=findexp(x,a)
  temp[x]=retreward+retvalue
  return temp[x]



def funreward(x,a):
  if(x[1]==0):
  	return 0
  else:
    if(a>GTs):
      if((x[0]-(power[a-(GTs+1)]+ph))>=math.ceil((d[x[1]][0]/velocity)*pf)):
        if(x[2]==0):
          #print(x,a,(x[0]-(power[a-(GTs+1)]+ph)),math.ceil((d[x[1]][0]/velocity)*pf))
          if((a-GTs)>1):
            return(calcreward(x,a))
          else:
            return 0
        else:
          return 0
      else:
        return 0
    else:
    	return 0

######### Finding reward value using R(t) in base paper ###################


def calcreward(x,a):
  Pag[x[1]-1]=power[a-(GTs+1)]*f[x[1]-1]
  #return(1+(Pag[x[1]-1]/(sigma*sigma)))
  return(math.log2((1+(Pag[x[1]-1]/(sigma*sigma)))))




#############    find expected J value  ########################


def findexp(x,a):
  if(x[1]==0):
    if(a<=GTs):
      if(a==0):
        return 0
      else:
        if(x[2]==0):
           return(salpha*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2])]+(1-salpha)*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2]+1)])
        else:
          return(sbeta*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2])]+(1-sbeta)*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2]-1)])
    else:
    	return 0
  else:
    if(a<=GTs):
      if(a==0):
        if(x[2]==0):
          return(salpha*v[(0,0,x[2])]+(1-salpha)*v[(0,0,x[2]+1)])
        else:
          return(salpha*v[(0,0,x[2])]+(1-salpha)*v[(0,0,x[2]-1)])
      elif(a==x[1]):
        return 0
      else:
        if(x[2]==0):
          return(salpha*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2])]+(1-salpha)*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2]+1)])
        else:
          return(sbeta*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2])]+(1-sbeta)*v[(max(x[0]-math.ceil((d[x[1]][a]/velocity)*pf),0),a,x[2]-1)])
    else:
      ft1=max((x[0]-ph),0)
      ft2=max((x[0]-(power[a-(GTs+1)]+ph)),0)

      if(a==29 and m==14 and x[1]==10 and x[2]==0):
        #print(ft2)
        print("from ",x, " ---> ", (ft2,x[1],x[2]), " with ",alpha," and ",(ft2,x[1],x[2]+1), "with",1-alpha)
        #matrix[254][int(ft2)]=0.7
      if(a==29 and m==14 and x[1]==10 and x[2]==1):
              #print(ft2)
              print("from ",x, " ---> ", (ft1,x[1],x[2]), " with ",beta ,"and ",(ft1,x[1],x[2]-1), "with",1-beta)
         #     matrix[254][int(ft2)]=0.7



      """
      if(x[0]==197 and a==29):
          print("for",x[0],"and a=",power[a-(GTs+1)],"joules",ft2)
      if(x[0]==155 and a==30):
          print("for",x[0],"and a=",power[a-(GTs+1)],"joules",ft2)
      if(x[0]==155 and a==29):
          print("for",x[0],"and a=",power[a-(GTs+1)],"joules",ft2)
      """
      if(a==GTs+1):
       if(x[2]==1):
         return(beta*v[(ft1,x[1],x[2])]+(1-beta)*v[(ft1,x[1],x[2]-1)])
       else:
         return(alpha*v[(ft1,x[1],x[2])]+(1-alpha)*v[(ft1,x[1],x[2]+1)])
      else:
        if(x[2]==0):
          return(alpha*v[(ft2,x[1],x[2])]+(1-alpha)*v[(ft2,x[1],x[2]+1)])
        else:
          return(beta*v[(ft1,x[1],x[2])]+(1-beta)*v[(ft1,x[1],x[2]-1)])


vold={}
tempn={}
policy={}
x=l[0]
print(x)
while(1):
  #print("iteration",m)
  #print("actions",actions)
  policy={}
  vold=v.copy()
  for x in v:
    k=[]
    for a in actions:
      computedvalue=compute(x,a,m)
      k.append(computedvalue)

      if(x==l[0] and m==14 and a==17):
            print("value and action",computedvalue,a)

    v[x]=max(k)
    policy[x]=np.argmax(k)     # Here I should subtract GT accordingly I guess
  val1=np.empty(len(v))
  val2=np.empty(len(v))
  val1=list(v.values())
  val2=list(vold.values())
  diff=np.subtract(val1,val2)
  diff=abs(diff)
  if((diff<0.0005).all()):
    #print(v)
    break
  m=m+1
print("last iteration",m)
x=l[0]
t1=0

#print("last iteration is",m)
#print(d)
while(x[0]>0):
	t=()
	print("state",x)
    ### THIS IS THE CODE TO PRINT PATH OF  GIVEN STATE AND ACTION
	if(x[0]==l[0][0]):
	    y=policy.get(x)
	    #y=   ### HERE I AM SETTING THE ACTION FOR START STATE
	else:
	    y=policy.get(x)
	#y=policy.get(x)
	print("policy is",y)
	if(y<=GTs):
		print("Fly to ",y)
	else:
		print(" transfer",power[y-(GTs+1)])
	if(y==0):
		t=(x[0]-x[0],x[1],x[2])
	elif(y<=GTs):
		t=(x[0]-math.ceil((d[x[1]][y]/velocity)*pf),y,x[2])
		print("here",t)
	elif(y-GTs==1):
		t=(x[0]-ph,x[1],x[2])
	else:
		t=(max((x[0]-(power[y-(GTs+1)]+ph)),0) ,x[1],x[2])
	x=t
	t1=t1+1
print(t)

print("------------------------------------------------------------------------------------------------------------------------")
print("obtained throughput",v[l[0]])
print("starting state = ",l[0])

#finding number of Pmax slots
pmaxslots=[]
x=l[0]
for i in dist[0]:
    if(i!=0):
        #print(x[0])
        flightenergy=(i/velocity)*pf
        #print(flightenergy,"flightenergy")
        resbattery=max(x[0]-2*flightenergy,0)
        #print(valueis,"for dist",i)
        pslots=math.floor(resbattery/(power[size-1]+ph))
        #pslots=resbattery/(power[size-1]+ph)
        pmaxslots.append(pslots)

print("\n")


print("--------------distance------------------------")
k=0
#print(len(pmaxslots))
for i in d[0]:
    print(i,k)
    k=k+1
k=0

print(" ----------------------------pmaxslots---------------------")
#printing number of pmaxslots for each GT
for i in pmaxslots:
    print(i,k+1)
    k=k+1


print("\n")
#print(pmaxslots)

position=np.argmax(pmaxslots)
potentialGTs=[]
k=1
omicron=0.5

for i in pmaxslots:
    if(abs(pmaxslots[position]-i)<omicron):
        potentialGTs.append(k)
    k=k+1
#print("\n\n")
print("potentialGTs are ", potentialGTs)
fadeofpotentialGTs=[]
for i in  potentialGTs:
    fadeofpotentialGTs.append(f[i-1])
print(fadeofpotentialGTs)
bestGT=np.argmax(fadeofpotentialGTs)
#print(matrix[254])

print("Best GT is ",potentialGTs[bestGT])

# finding residual batteries####
#for i in potentialGTs
#omicron will decide the number of potentialGTs. If omicron is high we get more number of potentialGTs ###
