import codecs
from .map import bytecode_map as bmap

class Smali:
    
    def __init__(self, path):
        self.path = path
        with codecs.open(path, 'r', 'utf-8') as f:#读
            self.lines = f.readlines()
        self.linenum = len(self.lines)
            
    def __to_next_method(self, begin):
        while begin < self.linenum:
            if self.lines[begin].startswith(".method"):#method开头
                return begin
            begin += 1
        return -1;
                
    def __analyze_line(self, line):
        words = line.split()
        if words:
            cmd = words[0]
            ctype = bmap.get(cmd, 0)#进行匹配
            if ctype != 0:
                self.featurelist.append(ctype)
    
    def getFeature(self):
        self.featurelist = []
        cursor = 0
        while True:
            cursor = self.__to_next_method(cursor)
            if cursor == -1:
                return "".join(self.featurelist)
            while True:
                cursor += 1;
                line = self.lines[cursor].strip()

                if line.startswith(".end"):#一个method提取字节码结束
                    self.featurelist.append("|")
                    break
                
                self.__analyze_line(line)
                
                
                 
                
                        
            
    
        