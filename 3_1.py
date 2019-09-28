#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


# In[114]:


def calc_angles(df):
    h_deg = df['DegreeEarthLocationHelioCentric'].values
    h_min = df['MinuteEarthLocationHelioCentric'].values
    g_deg = df['DegreeMarsLocationGeoCentric'].values
    g_min = df['MinuteMarsLocationGeoCentric'].values
    theta1 = []
    theta2 = []
    phi1 = []
    phi2 = []
    for i in range(0,df.shape[0],2):
        #print(i)
        theta1.append((h_deg[i] + h_min[i] / 60) * math.pi/180)
        theta2.append((h_deg[i+1] + h_min[i+1] / 60) * math.pi/180)
        phi1.append((g_deg[i] + g_min[i] / 60) * math.pi/180)
        phi2.append((g_deg[i+1] + g_min[i+1] / 60) * math.pi/180)
    return theta1,theta2,phi1,phi2


# In[116]:


def calc_coordinates_tri(m1,m2,t1,t2):
    x_coord = (m2 * math.cos(t2) - m1 * math.cos(t1) + math.sin(t1) - math.sin(t2)) / (m2 - m1)
    y_coord = (m2 * m1 *(math.cos(t2) - math.cos(t1)) +  m2 * math.sin(t1) - m1 * math.sin(t2)) / (m2 - m1)
    return x_coord, y_coord


# In[117]:


def coordinates_tri(t1,t2,p1,p2):
    x_coord = []
    y_coord = []
    for i in range(len(t1)):
        x, y = calc_coordinates_tri(math.tan(p1[i]),math.tan(p2[i]),t1[i],t2[i])
        x_coord.append(x)
        y_coord.append(y)
    return x_coord, y_coord


#    #### 2 part (ii)

# In[137]:


def calc_distance(r,x,y):
    loss = 0
    for i in range(len(x)):
        euc_distance = math.sqrt(x[i]**2 + y[i] ** 2)
        loss += abs(euc_distance - r)
    return loss


# In[138]:


def minimize_loss_tri(x,y):
    radius = 2
    parameter = minimize(calc_distance,radius,args = (x,y),method = 'L-BFGS-B')
    opt_param = parameter['x']
    loss = parameter['fun']
    return opt_param,loss


# In[123]:


def read_dataset(filename):
    df = pd.read_csv(filename)
    
    Zodiac_index = df['ZodiacIndex'].values
    Degree = df['Degree'].values
    Minute = df['Minute'].values
    Second = df['Second'].values
    sun = Zodiac_index, Degree, Minute, Second
    
    LatDegree = df['LatDegree'].values
    LatMinute = df['LatMinute'].values
    lat = LatDegree, LatMinute
    
    return sun, lat


# In[125]:


def calculate_heliocentric_angle(lat,r):
    deg, minute = lat
    d = r - 1
    helo_angle = []
    for i in range(len(deg)):
        theta = (deg[i] + minute[i]/60) * math.pi / 180
        phi = math.atan(d * math.tan(theta) / r)
        helo_angle.append(phi)
    return helo_angle


# #### 3 part(ii)

# In[127]:


def calc_longitude(z,d,m,s):
    return (z*30 + d + m/60 + s/3600 ) * math.pi/180


# In[128]:


def longitude(filename):
    sun,lat = read_dataset(filename)
    Zodiac_index, Degree, Minute, Second = sun
    alpha = []
    for i in range(len(Zodiac_index)):
        a = calc_longitude(Zodiac_index[i],Degree[i],Minute[i],Second[i])
        alpha.append(a)
    return alpha


# In[130]:


def coordinates_3d(theta,phi):
    coord = []
    for i in range(len(phi)):
        x = [math.cos(theta[i]) * math.cos(phi[i]), math.cos(theta[i]) * math.sin(phi[i]), math.sin(theta[i])]
        coord.append(x)
    return coord


# #### 3 part(iii)

# In[132]:


def calc_TotalDisFromPlane(params,coord):
    distance = 0
    for i in range(len(coord)):
        a,b,c = params
        distance += abs((a * coord[i][0] + b * coord[i][1] + c * coord[i][2]) / math.sqrt(a ** 2 + b ** 2 + c ** 2))
    return distance


# In[133]:


def minimize_loss_3(coord_3d):
    a = 1
    b = 1
    c = 1
    params = a, b, c
    parameter = minimize(calc_TotalDisFromPlane,params,args = coord_3d,method = 'L-BFGS-B')
    opt_param = parameter['x']
    loss = parameter['fun']
    return opt_param,loss


# In[150]:


def calc_inclination(param):
    angle = math.acos(param[2]/(math.sqrt(param[0] ** 2 + param[1] ** 2 + param[2] ** 2)))
    angle = angle * 180 / math.pi
    return angle


# #### 4 part(i)

# In[90]:


def calc_CoordOnMarsPlane(params,coord):
    coord_marsplane = []
    x, y = coord
    for i in range(len(x)):
        z = (params[0] * x[i] + params[1] * y[i]) * -1 / param[2]
        coord_marsplane.append([x[i],y[i],z])
    return coord_marsplane


# In[152]:


def find_dist_circle(coord):
    distance = 0
    d = []
    for i in range(np.array(coord).shape[0]):
        x = math.sqrt(coord[i][0] ** 2 + coord[i][1] ** 2 + coord[i][2] ** 2)
        distance += x
        d.append(x)
    loss = np.var(d)
    return distance/np.array(coord).shape[0] , loss


# #### 4 part(ii)

# In[205]:


def focus_ellipse(theta,phi,c):
    x = [ 2 * c * math.cos(theta) * math.cos(phi),  2 * c * math.cos(theta) * math.sin(phi), 2 * c * math.sin(phi)]
    #print(x)
    return x


# In[208]:


def calc_Dismarsplane(param,angle,coord):
    c, phi = param
    foc1 = [0,0,0]
    d = []
    foc2 = focus_ellipse(angle,phi,c)
    #print(foc2)
    for i in range(np.array(coord).shape[0]):
        d1 = math.sqrt(coord[i][0] ** 2 + coord[i][1] ** 2 + coord[i][2] ** 2)
        d2 = math.sqrt((foc2[0]-coord[i][0]) ** 2 + (foc2[1]-coord[i][1]) ** 2 + (foc2[2]-coord[i][2]) ** 2)
        d.append(d1+d2)
    loss = np.var(d)
    return loss


# In[209]:


def minimize_loss(angle,coord):
    c = 1
    phi = 1
    params = c, phi
    angle = angle * math.pi / 180
    parameter = minimize(calc_Dismarsplane,params,args = (angle,coord),method = 'L-BFGS-B')
    opt_param = parameter['x']
    loss = parameter['fun']
    return opt_param,loss


# In[214]:


if __name__ == '__main__':
    df = pd.read_csv('../data/01_data_mars_triangulation.csv')
    # 2 part(i)
    theta1,theta2,phi1,phi2 = calc_angles(df)
    x , y = coordinates_tri(theta1,theta2,phi1,phi2)
    # 2 part(ii)
    param = minimize_loss_tri(x,y)
    radius = param[0]
    print(radius)
    
    #3 part(i)
    data = pd.read_csv('../data/01_data_mars_opposition.csv')
    sun, lat = read_dataset('../data/01_data_mars_opposition.csv')
    helo_angle = calculate_heliocentric_angle(lat,radius)
    print("Heliocentric angle is : ",helo_angle)
    print
    
    #3 part(ii)
    filename = '../data/01_data_mars_opposition.csv'
    phi = longitude(filename)
    coord_3d = coordinates_3d(helo_angle,phi)
    
    #3 part(iii)
    param, loss = minimize_loss_3(coord_3d)
    print(param, loss)
    angle = calc_inclination(param)
    print("Inclination between the plane is :",angle)
    print

    #4 part(i)
    coord = x,y
    coord_marsplane = calc_CoordOnMarsPlane(param,coord)
    print("Coordinate of the Mars position : ", coord_marsplane)
    print
    
    #4 part(ii)
    radius_circle, closs = find_dist_circle(coord_marsplane)
    print("Radius of the circle is : ",radius_circle)
    print("Total Loss after fitting the circle is: ", closs)
    print
    
    #4 part(iii)
    opt_param, eloss = minimize_loss(angle,coord_marsplane)
    #print("Radius of the circle is : ",radius_circle)
    print("Total Loss after fitting the ellipse is: ", eloss)
    print
    print("Difference between losses after fitting circle and ellipse is :( var(R_circle) - var(R_ellipse)",closs-eloss)


# In[ ]:




