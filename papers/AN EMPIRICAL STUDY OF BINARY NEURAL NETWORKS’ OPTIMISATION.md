### AN EMPIRICAL STUDY OF BINARY NEURAL NETWORKS’ OPTIMISATION   

使用Straight-Through-Estimator(STE)得到的具有确定的二值权重的网络实现了SOTA的结果，但是他们的训练过程并没有充足的依据。这是由于forword path 中的evaluated function和back-propagation的权重更新之间的discrepancy，权重的更新并不对应前向通道的梯度。   
高效的收敛和准确性依赖于 careful fine-tuning and various ad-hoc techniques (Ad Hoc源自于拉丁语，意思是“for this”引申为“for this purpose only”，即“为某种目的设置的，特别的”意思).因此本文的工作就是研究这些常用的ad-hoc技术的有效性。
- 为了成功使用STE，通常需要adapt learning rates using second-moment methods，而其他的优化器容易陷入局部最优。
- 有些常用的tricks只在训练的最后有效，使得早期的训练变慢   
本文的贡献：分析了必要的ad-hoc技术，为未来发展提供理论基础，新的方案使得训练进一步加速。   

### 介绍

BNN的动机：远程机器共享数据和模型所涉及的隐私问题，云运算不可行的环境和场景下应用DNN   
然而，这类设备（手机，可穿戴设备，IoT，机器人）的要求是非常苛刻的：有严格的计算、存储、内存和带宽限制；许多应用程序需要实时工作；许多设备要求电池寿命长，可供全天使用或一直使用；在设计轻薄设备时，需要考虑一个热上限。   
另一方面，准确性要求网络更深，计算密集，尤其是CNN，虽然参数量少，但是计算消耗大。   
压缩和高效实现的方法主要有：pruning, weight sharing, low-rank approximation, knowledge distillation and quantisation to lower precision.
