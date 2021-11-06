# AndroidMal
基于机器学习的安卓恶意代码检测
op_malware.py:训练模型并生成混淆矩阵图
roc2.py：绘制roc曲线
data文件夹下的py文件用于预处理Android样本：
batch_disasseble.py：调用apktool反汇编Android样本
bytecode_extract.py：提取操作码
n_gram.py：提取n-gram特征
