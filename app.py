

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data4.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT, sex TEXT, name TEXT, age TEXT, time TEXT)')


def add_userdata(username,password,sex,name,age,time):
	c.execute('INSERT INTO userstable(username,password,sex,name,age,time) VALUES (?,?,?,?,?,?)',(username,password,sex,name,age,time))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


"""Simple Login App"""

st.title("HOẠT ĐỘNG LÀM VIỆC")

menu = ["Home","Login","SignUp"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Home":
    image = Image.open('crashier.jpg')
    st.image(image)

    st.subheader('Báo cáo hoạt động làm việc hằng ngày của nhân viên')	
    image = Image.open('mag.jpg')
    st.image(image,width=200)

elif choice == "Login":
    st.subheader("Chào Mừng Trở Lại Ca Làm Việc")
    st.markdown("Vui Lòng Đăng Nhập")
    image1 = Image.open('signin.jpg')
    st.image(image1)


    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    if st.sidebar.checkbox("Login"):
        # if password == '12345':
        create_usertable()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:
            st.success("Logged In as {}".format(username))
            with open ('user.csv', mode = 'w', newline='') as f:
                        csv_writer=csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            task = st.selectbox("Task",["Dashboard","Profiles","Manage"])
            image = Image.open('male.jpg')
            # st.write('Tên nhân viên: ',new_name)
            # st.write('Giới tính: ',new_sex)
            # st.write('Ca làm việc: ',new_time)
            st.image(image)
            

            if username == "admin":	
                if task == "Dashboard":
                    st.subheader("Báo Cáo Hôm Nay")
                    st.title("Biểu Đồ Báo Cáo")
                    data = pd.read_csv("tt.csv")
                    chart_visual = st.sidebar.selectbox('Chọn Loại Biểu Đồ',
                                    ('Biểu đồ cột', 'Biểu đồ tròn'))
                    
                    selected_status = st.sidebar.selectbox('Chọn Trạng thái',
                                    options = ['KhongNghiemTuc',
                                                'NghiemTuc', 'VuiVe',
                                                'KhoChiu'])
                    fig = go.Figure()
                    if chart_visual == 'Biểu đồ cột':
                        if selected_status == 'KhongNghiemTuc':
                            fig.add_trace(go.Bar(x = data.name, y = data.KhongNghiemTuc,
                                name = 'KhongNghiemTuc'))
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'NghiemTuc':
                            fig.add_trace(go.Bar(x = data.name, y = data.NghiemTuc,
                                    name = 'NghiemTuc'))
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'VuiVe':
                            fig.add_trace(go.Bar(x = data.name, y = data.VuiVe,	
                                name = 'VuiVe'))
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'KhoChiu':
                            fig.add_trace(go.Bar(x=data.name, y=data.KhoChiu,
                                name="KhoChiu"))
                            st.plotly_chart(fig, use_container_width=True)
                    if chart_visual =='Biểu đồ tròn':
                        if selected_status == 'KhongNghiemTuc':
                            fig = px.pie(data, values='KhongNghiemTuc', names='name')
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'NghiemTuc':
                            fig = px.pie(data, values='NghiemTuc', names='name')
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'VuiVe':
                            fig = px.pie(data, values='VuiVe', names='name')
                            st.plotly_chart(fig, use_container_width=True)
                        if selected_status == 'KhoChiu':
                            fig = px.pie(data, values='KhoChiu', names='name')
                            st.plotly_chart(fig, use_container_width=True)
                        
                # elif task == "Manage":
                #     if st.button("Run Model"):
                #         st.info("Model is running")
                        
                elif task == "Profiles":
                    st.subheader("User Profiles")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["Username","Password","Sex","Name","Age","Time"])
                    st.dataframe(clean_db)
        else:
            st.warning("Incorrect Username/Password")

elif choice == "SignUp":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')
    new_sex = st.selectbox('Chọn giới tính', ('Không xác định','Nam','Nữ'))
    new_name = st.text_input("Tên đầy đủ:")
    new_age = st.number_input("Tuổi của bạn",min_value=16,max_value=35,value=16)
    new_time = st.selectbox('Chọn Ca làm việc',('Chọn','Sáng','Chiều'))

    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user,make_hashes(new_password),new_sex,new_name,new_age,new_time)
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")

elif choice == "Manange":
    st.title("Report Dashboard")
