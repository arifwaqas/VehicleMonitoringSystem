# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:27:49 2020

@author: TUSHAR GOEL 
"""
import re
state_region = ['AN','AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ','HR','HP','JH','JK','KL','KA','LD','MH','ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB']
number = ['0','1','2','3','4','5','6','7','8','9']
data = set()
def plate_no_recognizer(text):
    global data
    
    
    s = '{}'.format(text)
    filtered_text = re.sub(r'[^\w]','',s)
    
    if len(filtered_text)==10:
        state_code = filtered_text[0:2]
        if state_code in state_region:
           
            district_code = filtered_text[2:4]
       
            last_digit_counter=0
            last_numbers = filtered_text[-4:]
            for i in range(len(last_numbers)):
                if last_numbers[i] in number:
                    last_digit_counter = last_digit_counter+1
                    
            if last_digit_counter==4:
                a = '{}{}{}{}'.format(state_code,district_code,filtered_text[-6:-4],last_numbers)
                return a
    else:
        print('Please Check the number plate')
plate = plate_no_recognizer('“UP15AX1385')
print(plate)