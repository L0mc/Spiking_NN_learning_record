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

the integration of the timing of spikes into the training process, only required for asynchronous SNNs, requires additional effort. 主要有五种方向的方法：
- Binarization of ANNs: Conventional DNNs are trained with binary activations, but maintain their synchronous mode of information processing
- Conversion from ANNs: Conventional DNNs are trained with BP, and then all analog neurons are converted into spiking ones
- Training of constrained networks: Before conversion, conventional DNN training methods are used together with constraints that model the properties of the spiking neuron models
- Supervised learning with spikes: Directly training SNNs sing variations of error backpropagation
- local learning rules at synapses, such as STDP, are used for more biologically realistic training.   

### 3.1 Binary DNNs   
介绍的是带有二进制激活值以及低比特权重的网络训练算法
- 相比SNNs：propagate information in a synchronized way and layer-by-layer like in conventional DNNs.
- 相比DNNs: energy-efficient due to sparse activations and computation on demand. 无论是CPUs，GPUs还是event-based neuromorphic systems，（考虑内存带宽，乘加运算（简化成bitwise XNORs and bit counting），权重kernel重用）   

不考虑浮点训练然后二值化的方案，从开始就采用二值化的激活值，则主要有两种方法：
- deterministic methods: apply straight-through-estimators to approximate non-differentiable activation functions during BP and accumulate gradients on so-called shadow weights. 前向的时候，激活值和浅权重被量化，反向时通过假设激活值和权重都是连续值来计算梯度。通过在浅权重中积累的方式，量化的权重可以在每次很小的改变下进行更新。
- stochastic methods: expectation backpropagation, where neuron activations and synaptic weights are represented by probability distributions updated by backpropagation.
- 此外： normalization of activations, modifications of regularizers, gradual transitions from soft to hard binarizations, adding noise on activations and weigts and knowledge distillation. 通常性能会下降，通过增加宽度来补偿。输入输出通常不进行二值化，计算代价不高，如果二值化性能下降很大。二值化也提高了鲁棒性。将二值化从01变成-1 1 提高了收敛性。1bit的数据会导致神经硬件的实现不再是sparse的，也可以将payload中的bit数增加。    

总结：高效的inference，带来的是性能的降低，训练时间的提升，训练算法更加复杂，网络的增大。由于高效性，属于独立于SNN的一个研究领域。   

### 3.2 Conversion of DNNs
为了规避（circumvent）脉冲网络中的梯度下降问题，将传统训练好的DNNs转化为SNNs by adapting weights and parameters of the spiking neurons. 目标是是先同样的input-output mapping。这个mapping不止包含网络本身，还包括输入输出的编码。
- 所有都是rate-coding（激活值对应放射率） **可以不吗**
- 权重需要被rescaled，根据脉冲神经元的参数 eg：leak rates or refractory times。（在conversion之前作为参数，不参与原来网络的训练）  

优缺点：
- 优点：可以利用DNN的训练工具，不考虑后续转换，性能好：benchmark records
- 缺点：不是所有的ANN都可以轻松转换成SNN
    1. ANN的激活值有负值，而SNN的频率没有负数；ANN在不同的输入下会产生可正可负的激活值，脉冲神经元分为激活和抑制（唯一带有正的还是负的突触）解决方法：1.对每个ANN产生两组互斥的脉冲神经元 2. ReLU激活函数。 Sigmoid函数由于非线性就需要额外的近似并且引入了额外的误差，相比ReLU。负的激活值对于输出层中的softmax层有很大的影响，但也有相应的解决方案。
    2. max-pooling很难实现，因为是非线性的，and cannot be computed on a spike-by-spike basis。大多数通过利用average pooling来避开这个问题，但是会产生性能的下降。通过一些机制可以做到只让最大发射频率善生的脉冲通过，这产生了更好的准确性。通过latency codes的方式可以实现max operations，但是这和conversion中用到的rate codes不兼容。
- 问题：ReLU全层的线性rescale不会改变最后输出的类别，但是SNN会受到影响。
    1. Low firing rates 对于noisy firing rates and temporal jitter（不稳定） of spikes很敏感，会增加每个估计之间的variance并且拉长可靠估计产生的时间。
    2. High firing rates 当predicted firing rates 超过了由参数决定的神经元最大发射频率时候会出问题。方法是通过propagate训练集中的一个子集，观测每一层的发射频率，然后rescale没层的输入权重，使得达到最终的目标频率   

conversion 和 weight normalization会伴随着更多脉冲的问题，进而就没那么高效。latency and accuracy之间的trade-off可以通过让SNN今早达到目标来补偿产生的影响。   
为了解决在频率编码上转换不高效的问题，一个重要的研究方向就是alternative spike codes based on the timing information.(相关的研究论文在文中被列出）
