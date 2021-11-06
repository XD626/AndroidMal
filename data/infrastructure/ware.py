import os
import re
from .smali import Smali

class Ware:
    
    __smali_pat =  re.compile(r"\.smali$")#正则表达式对象

    def __init__(self, path, isMalware):
        self.name = os.path.split(path)[-1]#name
        smaliPath = os.path.join(path, "smali")
        self.smalis = []
        self.isMalware = isMalware
        for root, dirs, files in os.walk(smaliPath):
            for file in files:
                if Ware.__smali_pat.findall(file):#匹配
               #if file.endswith(".smali"):
                    self.smalis.append(Smali(
                            os.path.join(root, file)
                            ))
                    
    def extractFeature(self, datafile):#提取特征
        feature = ''
        for smali in self.smalis:
            feature += smali.getFeature()
        datafile.append(self.name, feature, self.isMalware)
        
                    
             
            