#Script para copiar todas las imagenes de los splits juntas a la carpeta JPEGImages

import glob, os
from shutil import copyfile

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


path_splits = '/home/satur/FPDS/valid/'
#path_dst_img='/home/satur/Fallen_data_new/train/JPEGImages'
path_dst_labels='/home/satur/FPDS/labels'
splits=[f.path for f in os.scandir(path_splits) if f.is_dir()]


#Copia anotaciones (archivos.txt) de NuevosSplits a labels        
for dir in splits:
     for pathAndFilename in sorted(glob.iglob(os.path.join(dir, "*.txt"))):
          path,filename = os.path.split(pathAndFilename)
          dst=os.path.join(path_dst_labels, filename)      
          print(dst)
          copyfile(pathAndFilename,dst)

