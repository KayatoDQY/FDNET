import os
import re

#Dictionaries for training, test and validation sets
train={}
test={}
valid={}


def readfile(file):
     ###############################################################################################
     # Computes the number of fallen persons and non-fallen persons in each annotation file (txt)
     ###############################################################################################
     file_id=open(file)
     line=file_id.readline()
     cont_nonfallen=0
     cont_fallen=0
     
     while line:
          
          params=[int(n) for n in line.split()]
          if len(params)==0:
               break
          #Label Fallen Persons='1'
          #Label Non-Fallen Persons='-1'
          if params[0]!=1: 
               cont_nonfallen+=1 
          elif params[0]==1:
               cont_fallen+=1         
          line=file_id.readline()
     file_id.close()               
     return cont_fallen,cont_nonfallen



for root,dirs,files in os.walk(".", topdown=True):
     for name in files:
          if ('train' in root or 'valid' in root or 'test' in root) and 'txt' in name:
               path_file=os.path.join(root,name)
               print(path_file)
               set1, set2 = os.path.split(root)
               set1=set1[2:]
               if 'split' in set2:
                    ind=re.findall('\d+',set2)
                    l=len(ind[0])
                    if l==1:
                         set2='split'+'0'+ind[0]
          
               
               if set1=='train':
                    d=train
               elif set1=='valid':
                    d=valid
               elif set1=='test':
                    d=test
            

               fallen,nonfallen=readfile(path_file)
               if set2 not in d:
                    d[set2]={'NoImages':1,'FallenPersons':fallen,'NonFallenPersons':nonfallen}
               else:
                    d[set2]['NoImages']+=1
                    d[set2]['FallenPersons']+=fallen
                    d[set2]['NonFallenPersons']+=nonfallen
          
          
#Write results in info_splits.txt
fid=open('Info_splits.txt','w')
fid.write('-----Training-----\n')
for key in sorted(train):
     fid.write("Split:%s, N_Frames:%0.4d, N_FallenPersons:%0.4d, N_NonFallenPersons:%0.4d\n" %(key,train[key]['NoImages'],train[key]['FallenPersons'],train[key]['NonFallenPersons']))
fid.write('\n-----Validation-----\n')
for key in sorted(valid):
     fid.write("Split:%s, N_Frames:%0.4d, N_FallenPersons:%0.4d, N_NonFallenPersons:%0.4d\n" %(key,valid[key]['NoImages'],valid[key]['FallenPersons'],valid[key]['NonFallenPersons']))
fid.write('\n-----Test-----\n')
for key in sorted(test):
     fid.write("Split:%s, N_Frames:%0.4d, N_FallenPersons:%0.4d, N_NonFallenPersons:%0.4d\n" %(key,test[key]['NoImages'],test[key]['FallenPersons'],test[key]['NonFallenPersons']))
fid.close()






