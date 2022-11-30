import os
 
os.chdir(r'C:\Users\fuse\tensor\stylegan3\_training-runs\imgs')
print(os.getcwd())
 
for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)

    f_name = '%05d' %(count)
 
    # new_name = f'fake{count}{count}'
    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)
