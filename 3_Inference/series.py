# general Program to see the series:
# 20,18,16,14,12,10,8,6,4,2

number = int(input('enter the number: \n'))
multiple = int(input("enter the Multiple: \n"))
for i in range(number*multiple,number-1,-number):
    print(i,end=',')
