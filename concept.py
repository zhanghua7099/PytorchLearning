'''
 1. batch_size、epoch、iteration是深度学习中常见的几个超参数：
（1）batch_size：每批数据量的大小。DL通常用SGD的优化算法进行训练，也就是一次（1 个iteration）一起训练batch_size个样本，计算它们的平均损失函数值，来更新参数。
（2）iteration：1个iteration即迭代一次，也就是用batchsize个样本训练一次。
（3）epoch：1个epoch指用训练集中的全部样本训练一次，此时相当于batchsize 等于训练集的样本数。
'''