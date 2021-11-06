import sys
from infrastructure.mydict import MyDict
import pandas as pd

#n-gram的n值
n = int(sys.argv[1])
print("n = ", n)

origin = pd.read_csv("E:\\1XDXD\\XD_data\\data.csv")

 

mdict = MyDict()

feature = origin["Feature"].str.split("|")
total = len(feature)
for i, code in enumerate(feature):
    mdict.newLayer()
    if not type(code) == list:
        continue
    for method in code:
        length = len(method)
        if length < n:
            continue
        for start in range(length - (n - 1)):
            end = start + n
            mdict.mark(method[start:end])
    print("已完成", i+1, "个应用，进度：")
    print((i + 1) * 100 / total, "%")
            
result = mdict.dict
pd.DataFrame(result, index=origin.index)\
               .to_csv("./" + str(n) + "_gram.csv", index=False)
            
        
        
        

