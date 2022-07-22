#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd


# In[2]:


df = pd.read_csv('rs.csv')


# In[3]:


Khoa_data= df[df['Name']=='khoa'].value_counts()
Khoa_data_str = str(Khoa_data)
Khoa_data_list = Khoa_data_str.split()

Huy_data= df[df['Name']=='huy'].value_counts()
Huy_data_str = str(Huy_data)
Huy_data_list = Huy_data_str.split()

# In[4]:


Hoi_data= df[df['Name']=='Hoi'].value_counts()
Hoi_data_str = str(Hoi_data)
Hoi_data_list = Hoi_data_str.split()


# In[5]:


long_data= df[df['Name']=='long'].value_counts()
long_data_str = str(long_data)
long_data_list = long_data_str.split()

trung_data= df[df['Name']=='trung'].value_counts()
trung_data_str = str(trung_data)
trung_data_list = trung_data_str.split()


# In[6]:


# def transform(list,Name,khongnghiemtuc,khochiu,nghiemtuc,vuive,dtype,int64,Class):
#     for i in range(len(list)):
#         if list[i] == Name:
#             list.remove(Name)
#         if list[i] == Class:
#             list.remove(Class)
#         if list[i] == khongnghiemtuc :
#             list.remove(khongnghiemtuc)
#         if list[i] == khochiu:
#             list.remove(khochiu)
#         if list[i] == nghiemtuc:
#             list.remove(nghiemtuc)
#         if list[i] == vuive:
#             list.remove(vuive)
#         if list[i] == dtype:
#             list.remove(dtype)
#         if list[i] == int64:
#             list.remove(int64)


# # In[7]:


# Name = 'Name'
# KhongNghiemTuc = 'KhongNghiemTuc'
# khochiu = 'Khochiu'
# nghiemtuc = 'NghiemTuc'
# vuive = 'VuiVe'
# dtype='dtype:'
# int64='int64'
# Class='Class'


# In[10]:

Hoi_data_list.remove('Name')
Hoi_data_list.remove('KhongNghiemTuc')
Hoi_data_list.remove('Khochiu')
Hoi_data_list.remove('NghiemTuc')
Hoi_data_list.remove('VuiVe')
Hoi_data_list.remove('Class')
Hoi_data_list.remove('dtype:')
Hoi_data_list.remove('int64')


# In[13]:
Khoa_data_list.remove('Name')
Khoa_data_list.remove('KhongNghiemTuc')
Khoa_data_list.remove('Khochiu')
Khoa_data_list.remove('NghiemTuc')
Khoa_data_list.remove('VuiVe')
Khoa_data_list.remove('Class')
Khoa_data_list.remove('dtype:')
Khoa_data_list.remove('int64')




# In[16]:
trung_data_list.remove('Name')
trung_data_list.remove('KhongNghiemTuc')
trung_data_list.remove('Khochiu')
trung_data_list.remove('NghiemTuc')
trung_data_list.remove('VuiVe')
trung_data_list.remove('Class')
trung_data_list.remove('dtype:')
trung_data_list.remove('int64')
#
long_data_list.remove('Name')
long_data_list.remove('KhongNghiemTuc')
long_data_list.remove('Khochiu')
long_data_list.remove('NghiemTuc')
long_data_list.remove('VuiVe')
long_data_list.remove('Class')
long_data_list.remove('dtype:')
long_data_list.remove('int64')


# In[24]:


data = ['name', 'KhongNghiemTuc','NghiemTuc','VuiVe','KhoChiu']


# In[25]:


with open('tt.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(data)
    csv_writer.writerow(trung_data_list)
    csv_writer.writerow(Hoi_data_list)
    csv_writer.writerow(Khoa_data_list)
    csv_writer.writerow(long_data_list)






