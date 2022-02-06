#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:03:21 2022

@author: moni
"""
import re

from datetime import datetime
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from tkinter import Tk,Label,Button,Entry,messagebox,StringVar,OptionMenu
import sqlite3

returnval = 0

def lemmatize(word):
    wnl = WordNetLemmatizer()
    word = wnl.lemmatize(word, 'a')
    word = wnl.lemmatize(word, 'v')
    word = wnl.lemmatize(word, 'n')
    return word

def clean(text):
    str_punc = string.punctuation
    engstopwords = stopwords.words("english")
    engstopwordsV2 = re.sub('[' + re.escape(string.punctuation) + ']', '',' '.join(engstopwords)).split()
    engstopwords = set(engstopwords).union(set(engstopwordsV2))
    text = re.sub("http.*?([ ]|\|\|\||$)", "", text).lower()
    url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    text = re.sub(url_regex, "", text)
    text = re.sub(r'(:|;).', " ", text)
    text = re.sub('['+re.escape(str_punc)+']'," ",  text)
    text = re.sub('(\[|\()*\d+(\]|\))*', ' ', text)
    text = re.sub('[’‘“\.”…–]', '', text)
    text = re.sub('[^(\w|\s)]', '', text)
    text = re.sub('(gt|lt)', '', text)
    text = list(map(lemmatize, text.split()))
    text = [word for word in text if (word not in engstopwords)]
    text = " ".join(text)
    return text

def model_extractor():
    global model_1,model_2,model_3,model_4,tokenizer
    
    with open("tokenizer.pickle","rb") as token_file:
        tokenizer = pickle.load(token_file)
    model_1 = models.load_model("model_1.h5")
    model_2 = models.load_model("model_2.h5")
    model_3 = models.load_model("model_3.h5")
    model_4 = models.load_model("model_4.h5")


def close_cursor():
    conn.close()

def create_cursor():
    global conn,cursor
    conn = sqlite3.connect("employablity_test.db")
    cursor = conn.cursor()
    return 

def submit_test():
    inp1 = answer1.get()
    inp2 = answer2.get()
    inp3 = answer3.get()
    inp4 = answer4.get()
    if inp1=="" or inp2=="" or inp3=="" or inp4=="":
        messagebox.showwarning("","Please answer all the Questions")
    else:
        maxlen=922
        inp1 = clean(inp1)
        inp2 = clean(inp2)
        inp3 = clean(inp3)
        inp4 = clean(inp4)
        inp1 = tokenizer.texts_to_sequences([inp1])
        inp2 = tokenizer.texts_to_sequences([inp2])
        inp3 = tokenizer.texts_to_sequences([inp3])
        inp4 = tokenizer.texts_to_sequences([inp4])
        inp1 = pad_sequences(inp1, maxlen=maxlen)
        inp2 = pad_sequences(inp2, maxlen=maxlen)
        inp3 = pad_sequences(inp3, maxlen=maxlen)
        inp4 = pad_sequences(inp4, maxlen=maxlen)
        y1 = model_1.predict(inp1)
        y2 = model_2.predict(inp2)
        y3 = model_3.predict(inp3)
        y4 = model_4.predict(inp4)
        create_cursor()
        interaction_likelyhood=int(y1[0][0]*100)
        information_likelyhood=int(y2[0][0]*100)
        decision_likelyhood=int(y3[0][0]*100)
        structure_likelyhood=int(y4[0][0]*100)
        type_class = ""
        if interaction_likelyhood>25:
            interaction="Extraversion"
            type_class+="E"
        else:
            interaction="Introversion"
            type_class+="I"
        if information_likelyhood>25:
            information="Sensing"
            type_class+="S"
        else:
            information="Intuition"
            type_class+="N"
        if decision_likelyhood>=45:
            decision="Thinking"
            type_class+="T"
        else:
            decision="Feeling"
            type_class+="F"
        if structure_likelyhood>=45:
            structure="Judging"
            type_class+="J"
        else:
            structure="Perceving"
            type_class+="P"
        cursor.execute("Select FullName,Email,PhoneNo from User where Username = '"+username+"'")
        data = cursor.fetchall()
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        
        cursor.execute("Insert into Entry values (:ID,:Timestamp,:Job,:FullName,:Email,:PhoneNo,:Type,:Interaction,:Interaction_likelyhood,:Information,:Information_likelyhood,:Decision,:Decision_likelyhood,:Structure,:Structure_likelyhood)",
                           {
                               "ID":None,
                               "Timestamp":dt_string,
                               "Job":choice.get(),
                               "FullName":data[0][0],
                               "Email":data[0][1],
                               "PhoneNo":data[0][2],
                               "Type":type_class,
                               "Interaction":interaction,
                               "Interaction_likelyhood":interaction_likelyhood,
                               "Information":information,
                               "Information_likelyhood":information_likelyhood,
                               "Decision":decision,
                               "Decision_likelyhood":decision_likelyhood,
                               "Structure":structure,
                               "Structure_likelyhood":structure_likelyhood
                            }
                           )
        conn.commit()
        messagebox.showinfo("","Submitted Successfully")
        close_cursor()
        global returnval
        returnval=1
        Employee()
        


def take_test():
    employee_window.destroy()
    
    global test_window,answer1,answer2,answer3,answer4
    test_window = Tk(className="Test Window")
    test_window.configure(bg='#856ff8')
    
    create_cursor()
    cursor.execute("Select questions from JobsQuestions where Job='"+str(choice.get())+"'")
    lst = cursor.fetchall()
    questions = str(lst[0][0])
    questions = questions.split("|||")
    close_cursor()
    
    question1 = Label(test_window,text=questions[0],bg='#856ff8',font=("Times New Roman",20))
    question2 = Label(test_window,text=questions[1],bg='#856ff8',font=("Times New Roman",20))
    question3 = Label(test_window,text=questions[2],bg='#856ff8',font=("Times New Roman",20))
    question4 = Label(test_window,text=questions[3],bg='#856ff8',font=("Times New Roman",20))
    answer1 = Entry(test_window)
    answer2 = Entry(test_window)
    answer3 = Entry(test_window)
    answer4 = Entry(test_window)
    
    submit_btn = Button(test_window,text="Submit",bg='#856ff8',command=submit_test)
    question1.grid(row=1,column=1,padx=10,pady=10)
    answer1.grid(row=2,column=1,columnspan=400,padx=20,pady=10,ipadx=420,ipady=25)
    question2.grid(row=4,column=1,padx=10,pady=10)
    answer2.grid(row=5,column=1,columnspan=400,padx=20,pady=10,ipadx=420,ipady=25)
    question3.grid(row=7,column=1,padx=10,pady=10)
    answer3.grid(row=8,column=1,columnspan=400,padx=20,pady=10,ipadx=420,ipady=25)
    question4.grid(row=10,column=1,padx=10,pady=10)
    answer4.grid(row=11,column=1,columnspan=400,padx=20,pady=10,ipadx=420,ipady=25)
    submit_btn.grid(row=15,column=15,pady=20)
    
    test_window.geometry("1100x700")
    test_window.eval("tk::PlaceWindow . center")


def Employee():
    if returnval==0:
        login_window.destroy()
    else:
        test_window.destroy()
    
    global employee_window,choice
    
    employee_window = Tk(className="Options window")
    employee_window.configure(bg='#856ff8')
    employability_label = Label(employee_window,bg='#856ff8',text="Employability Pensonality Test", font=("Times New Roman",72))
    job_label = Label(employee_window,text="Select the job category")
    create_cursor()
    options = []
    cursor.execute("Select Job from JobsQuestions")
    lst = cursor.fetchall()
    for x in lst:
        options.append(x[0])
    close_cursor()
    take_test_button = Button(employee_window,text="Take Test",bg='#856ff8',command=take_test)
    
    choice = StringVar()
    choice.set(options[0])
    
    drop_box = OptionMenu(employee_window,choice,*options)
    
    employability_label.pack(pady=40)
    job_label.pack(pady=10)
    drop_box.pack(pady=15)
    take_test_button.pack(pady=30)
    employee_window.geometry("1100x700")
    employee_window.eval("tk::PlaceWindow . center")
    employee_window.mainloop()
    


def verify_register():
    fullname = register_window_fullname_text.get()
    UserName = register_window_username_text.get()
    PassWord = register_window_password_text.get()
    RePassWord = register_window_Re_password_text.get()
    EmailID = register_window_email_text.get()
    PhoneNo = register_window_phone_text.get()
    if fullname=="" or UserName=="" or PassWord=="" or RePassWord=="" or EmailID=="" or PhoneNo=="":
        messagebox.showerror("","PLEASE FILL ALL THE FIELDS")
    else:
        if len(PassWord)<8:
            messagebox.showerror("","PASSWORD SHOULD HAVE MINIMUM 8 CHARACTERS")
        elif PassWord!=RePassWord:
            messagebox.showerror("","PASSWORD DO NOT MATCH")
        else:
            create_cursor()
            try:
                cursor.execute("Insert into User values (:id,:fullname,:username,:Password,:EmailId,:PhoneNo)",
                           {
                               "id":None,
                               "fullname":fullname,
                               "username":UserName,
                               "Password":PassWord,
                               "EmailId":EmailID,
                               "PhoneNo":PhoneNo
                            }
                           )
                conn.commit()
                close_cursor()
                messagebox.showinfo("","Registeration Successful")
                register_window.destroy()
                main()
            except Exception:
                messagebox.showerror("","USERNAME/EMAIL-ID ALREADY EXISTS")
                close_cursor()


def Register():
    login_window.destroy()
    global register_window,register_window_btn,register_window_username_text,register_window_fullname_text,register_window_password_text,register_window_Re_password_text,register_window_email_text, register_window_phone_text
    register_window = Tk(className="Register Window")
    register_window.configure(bg='#856ff8')
    register_window_register_label = Label(register_window,bg='#856ff8',text="Register",font=("Times New Roman",28))
    register_window_fullname_label = Label(register_window,text="Full Name",bg='#856ff8')
    register_window_fullname_text = Entry(register_window)
    register_window_username_label = Label(register_window,text="Username",bg='#856ff8')
    register_window_username_text = Entry(register_window)
    register_window_password_label = Label(register_window,text="Password",bg='#856ff8')
    register_window_password_text = Entry(register_window)
    register_window_Re_password_label = Label(register_window,text="Re-enter Password",bg='#856ff8')
    register_window_Re_password_text = Entry(register_window)
    register_window_email_label = Label(register_window,text="Email ID",bg='#856ff8')
    register_window_email_text = Entry(register_window)
    register_window_phone_label = Label(register_window,text="Mobile Number",bg='#856ff8')
    register_window_phone_text = Entry(register_window)
    register_window_btn = Button(register_window,text="Register",command=verify_register,bg='#856ff8')
    
    register_window_register_label.grid(row=1,column=1,columnspan=2,padx=5,pady=5)
    register_window_fullname_label.grid(row=2,column=0,columnspan=2,padx=5,pady=5)
    register_window_fullname_text.grid(row=2,column=2,columnspan=2,padx=5,pady=5)
    register_window_username_label.grid(row=3,column=0,columnspan=2,padx=5,pady=5)
    register_window_username_text.grid(row=3,column=2,columnspan=2,padx=5,pady=5)
    register_window_password_label.grid(row=4,column=0,columnspan=2,padx=5,pady=5)
    register_window_password_text.grid(row=4,column=2,columnspan=2,padx=5,pady=5)
    register_window_Re_password_label.grid(row=5,column=0,columnspan=2,padx=5,pady=5)
    register_window_Re_password_text.grid(row=5,column=2,columnspan=2,padx=5,pady=5)
    register_window_email_label.grid(row=6,column=0,columnspan=2,padx=5,pady=5)
    register_window_email_text.grid(row=6,column=2,columnspan=2,padx=5,pady=5)
    register_window_phone_label.grid(row=7,column=0,columnspan=2,padx=5,pady=5)
    register_window_phone_text.grid(row=7,column=2,columnspan=2,padx=5,pady=5)
    register_window_btn.grid(row=8,column=1,columnspan=2,padx=5,pady=5)
    register_window.eval("tk::PlaceWindow . center")
    register_window.mainloop()


def verify_login():
    global username
    create_cursor()
    username = EmailUsername_text.get()
    password = Password_text.get()
    if password == "" or username == "":
        messagebox.showinfo("", "Please entry all the fields")
    else:
        cursor.execute("Select * From User where username='"+username+"' and Password='"+password+"'")
        if len(cursor.fetchall())!=1:
            messagebox.showerror("","Incorrect Username or Password")
            close_cursor()
        else:
            messagebox.showinfo("","Login Successful")
            close_cursor()
            Employee()


def main():
    global login_window,EmailUsername_text,Password_text
    login_window = Tk(className="Login Window")
    login_window.configure(bg='#856ff8')
    login_label = Label(login_window,text="Login",font=("Times New Roman",28),bg='#856ff8')
    EmailUsername_label = Label(login_window,text="Username",bg='#856ff8')
    Password_label = Label(login_window,text="Password",bg='#856ff8')
    EmailUsername_text = Entry(login_window)
    Password_text = Entry(login_window)
    login_btn = Button(login_window,text="Login",command=verify_login,bg='#856ff8')
    Register_btn = Button(login_window,text="Register",command=Register,bg='#856ff8')
    
    login_label.grid(row=1,column=2,columnspan=2)
    EmailUsername_label.grid(row=2,column=1,columnspan=2)
    EmailUsername_text.grid(row=2,column=3,columnspan=2)
    Password_label.grid(row=3,column=1,columnspan=2)
    Password_text.grid(row=3,column=3,columnspan=2)
    login_btn.grid(row=4,column=5)
    Register_btn.grid(row=4,column=0)
    login_window.eval("tk::PlaceWindow . center")  
    login_window.mainloop()

model_extractor()
main()