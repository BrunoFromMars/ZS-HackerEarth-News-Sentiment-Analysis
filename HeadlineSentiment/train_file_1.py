
import pandas as pd



df = pd.read_csv('dataset/train_file.csv')


import csv

with open('dataset/train_file.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)





for i in range(len(your_list)):
    for j in range(1,4):
        your_list[i][j] = your_list[i][j].replace('&#39;','\'')
        your_list[i][j] = your_list[i][j].replace('&quot;','\'')
        your_list[i][j] = your_list[i][j].replace(';','.')
    



df_list = pd.DataFrame(your_list[1:]) 
df_list.to_csv('dataset/train_file_1.csv')


df = pd.read_csv('dataset/test_file.csv')



with open('dataset/test_file.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
    
    
    
for i in range(len(your_list)):
    for j in range(1,4):
        your_list[i][j] = your_list[i][j].replace('&#39;','\'')
        your_list[i][j] = your_list[i][j].replace('&quot;','\'')
        your_list[i][j] = your_list[i][j].replace(';','.')
    
df_list = pd.DataFrame(your_list[1:]) 
df_list.to_csv('dataset/test_file_1.csv')


