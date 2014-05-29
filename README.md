LDA
===

LDA Gibbs sampling

##part1

将要处理的文本预料放在./data/corpus局部路径下
运行dataprocess.py对文本预料进行处理(去除标点符号,全部转换成小写)
将每个文档以行的方式写入./data/test.csv文件中

##part2

lda.py实现lda
其中beta,alpha,迭代次数和主题数这些参数在文件开头进行设定
