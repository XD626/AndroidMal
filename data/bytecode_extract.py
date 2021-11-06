import os
from infrastructure.ware import Ware
from infrastructure.fileutils import DataFile

virusroot = "E:\\1XDXD\\XD_data\\smalis\\malware"
kindroot = "E:\\1XDXD\\XD_data\\smalis\\normal"

f = DataFile("E:\\1XDXD\\XD_data\\data.csv")



def collect(rootdir, isMalware):
    wares = os.listdir(rootdir)
    total = len(wares)
    for i, ware in enumerate(wares):
        warePath = os.path.join(rootdir, ware)
        ware = Ware(warePath, isMalware)
        ware.extractFeature(f)
        print("已提取", i + 1, "个文件的特征，进度：")
        print((i + 1) * 100 / total, "%")
        
    
#1代表恶意软件
collect(virusroot, 1)
collect(kindroot, 0)

f.close()
        

 


