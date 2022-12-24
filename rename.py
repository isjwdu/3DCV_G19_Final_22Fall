import os
path=input('input the path of files you want to rename(Please add ''/'' at last):')       

fileList=os.listdir(path)
n=0
for i in fileList:
    oldname=path+ os.sep + fileList[n]   # os.sep添加系统分隔符
    newname=path + os.sep +'COCO'+str(n+1)+'.JPG'
    os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
    print(oldname,'======>',newname)
    n+=1