import os
import subprocess

def disassemble(frompath, topath, num, start=0):
    files = os.listdir(frompath)
    files = files[start:num]#反汇编文件
    total = len(files)
    for i, file in enumerate(files):
        fullFrompath = os.path.join(frompath, file)
        fullTopath = os.path.join(topath, file)
        command = "apktool d " + fullFrompath + " -o " + fullTopath
        subprocess.call(command, shell=True)
        print("已反汇编", i+1, "个应用，进度：")
        print((i + 1) * 100 / total, "%")


#反汇编恶意软件样本

virus_root = "E:\\1XDXD\\XD_data\\ware\\malware"

disassemble(virus_root, "E:\\1XDXD\\XD_data\\smalis\\malware", 300)


#反汇编正常软件样本 
kind_root = "E:\\1XDXD\\XD_data\\ware\\normal"
#disassemble(kind_root, "E:\\1XDXD\\XD_data\\smalis\\normal", 268)