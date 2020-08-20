## Deep Learning With Spiking Neurons Opportunities and Challenges
**SNN 的特点**: 
- sparse and asynchronous binary signals are communicated and processed in a massively parallel fashion
- low power consumption, fast inference, online learning and event-driven information processing 
- event-based vision and audio sensors

**本文内容**:
- 脉冲神经网络提供的机会，以及脉冲神经网络训练方式的挑战，其训练应使得脉冲神经网络和传统的深度学习具有可竞争的表现能力，同时也能够高效的映射在硬件上
- 介绍了一系列的方法，分类，并总结优点缺点
   1. conversion of conventional deep networks into SNNs
   2. constrained training before conversion
   3. spiking variants of backpropagation 
   4. biologically motivated variants of STDP
- 讨论SNN和二进制网络之间的关系   

**本文观点**：  

&emsp; 神经方法和传统的机器学习不应该是同一类问题的两种不同解答方案，而应该去探索他们task-specific advantages. 深度脉冲神经网络能够和event-based sensors一起工作，利用temporal codes 并且可以local on-chip learning.

### 介绍   
神经科学中的大脑皮层的层级结构激发了DNN后的架构原则，但是在实施层面上，类脑计算和模拟神经网络之间只存在着微不足道的相似：
- ANN：non-linear but continuous function approximators that operate on a common clock cycle
- SNN: compute with asynchronous(event-based) spikes that signal the occurrence of some characteristic event by digital and temporally precise action potentials.
- BNN: typically executed in a synchronized manner   

### 1.1 脉冲神经网络   
假设所有的脉冲都是模式化的事件(stereoytypical events)，因此信息的传递由两个因素：
- 脉冲的时间：比如突触前和突触后脉冲的相对时间，独特的发射模式
- 使用的突触的identity：联结了那些神经元，突触是激活的还是抑制的，突触强度，可能的短期可塑性或者调整的影响   

根据仿真神经元的细节程度，可以分为：
- point neurons：到达的脉冲会立刻改变他们（somatic）膜电位
- multi-compartment models：有复杂的空间（dendritic）结构，在胞体电位被调整之前，树突电流会交互影响。
- eg: HH, spike response model, integrate-and-fire   

编码方式：
- rate codes：ANN和SNN最直接的关系就是将一个analog神经元的激活值作为一个spiking神经元在稳态时候的发射频率，也就是频率编码
- temporal codes: complex processes that depend on the relative timing between spikes or timing relative to some reference signal(network oscillations) 时域编码中一些小的变化（单个的脉冲或者是微笑的时间变化）就会产生不同的反应，通常在一个可信的脉冲频率被估计出来前就完成了决定过程。   

除了生物上的定义，从神经形态工程领域上有一个更加实用的以应用为导向的观点，在该领域SNN被叫做event-based 而不是 spiking:   
&emsp;其中的event表示一个信息的数字包，由他的origin and destination address, a timestamp组成，且携带a few bits of payload information. (address event representation protocol)。   
&emsp;这个协议被用来联结event-based sensors和神经形态芯片或者是数字后处理硬件。装载的信息可以传递任意种类的相关信息，且突触后目标可以完成更加复杂的功能，而不仅仅是简单得integrate-and-fire    

### 1.2 Deep SNNs的优点   
- 相比更加抽象的模型，和生物相近的模型能够实现近似的智能，或者起码计算更加高效
- 适合处理从神经形态传感器获得的spatio-temporal event-based 信息。一方面，外界环境下sparse的脉冲下计算高效，另一方面时域输入提供了更加额外的有价值的信息（相比frame-driven 方法）**（但在编程的过程中不还是要人工设定time step的吗），那还是要和硬件结合才能发挥出它真正的效率** 
- 快速性，不必要像传统的网络一样要等待前面的所有层都更新完毕才能输出，但是一开始的输出脉冲是依据不完整的信息做出的，所以处理输入的时间越久，产生的结果越准确。

### 1.3 Deep SNNs的局限
- 性能不好。有数据集的原因(conventional frame-based images)，需要将images转化为spike trains，这个过程lossy and inefficient.
    1. 在传统的AI数据集上的性能不应该作为最终目标，而要以behaviorally most relevant tasks
    2. 比如说是：在真实世界中连续的输入流下进行决策（自动驾驶），图片分类类似于突然闪现在视网膜上的一张随机图片的分类任务，并且没有支撑的上下文相关信息，和我们人脑的优化方向是不同的。
    3. 目前没有好的benchmark datasets（DVS）以及很好测量现实世界表现的evaluation metrics
- 缺乏能利用脉冲神经元能力（高效的time codes）的训练算法。
    1. 大多数方法采用传统DNN的rate-based近似，就不必期待会有性能提升了，但这种场景下，可以作为一种快速高效的实现。
    2. 异步以及不连续性导致BP不能直接使用   

### Inference with deep SNNs

SNN将输入信号转变为输出信号，这里要讨论输入层和输出层
- rate codes：sub-optimal but effective: Poisson processes with proportional firing rates 只考虑平均频率，忽略了精确的时间信息
- temporal codes：efficient（脉冲的数量减少）, fast, and map well to hardware，可以通过BP或者STDP训练，可以实现maxima pooling，但是准确性不高
   1. rank-order code in which every neuron can fire at most once
   2. the output of a neuron is the time of its first spike
- In order to tune the trade-off between rate-based and temporal coding, Lagorce et al. (2017) and Sironi et al. (2018) propose to use time surfaces around event input as hierarchical features.**(具体情况还要再看）**     

在输出的部分，也要进行转换：
- report the class corresponding to the neuron with the highest firing rate over some time period or over a fixed number of total output spikes
- report the neuron firing first as the output class
- 将 the number of output spikes 加入考量效果会最好
- 另外，可以被优化as early as possible产生正确的输出脉冲，使用larger populations of neurons，以及temporal smoothing

### 3 Training of Deep SNNs
