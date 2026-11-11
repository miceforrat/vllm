# 架构和接口设计文档

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDVmMjkyMDg3M2NhMDVhYTUzMDNjZTYxNjg5ZmE0NjBfYTcwMjcwMzJjZjEwN2ZmNTE5NTM3YjFjNDViZmE5NTVfSUQ6NzcwNDkzNzg3NTQ5ODIwODI0OV8xNzkzOTU0MDYzOjE3OTQwNDA0NjNfVjM)

我先放一张图在这里，成品可能和这个文档不太一样



目前得先做好worker级别的规整

scheduler方法暂时不做（不然复杂度太高嘞）



## rank级别的类设计（worker及以下部分）

我想的是用一套container\-like的思路

同时需要把model runner之上再套一层抽象——抽象成执行单元

这样我们就可以用一套scheduler统一管理执行单元的推进（资源调度则根据执行单元提供的profile展开）



同时，每个rank的worker都要为各个执行单元（和"container"）提供统一的资源管理

Unitscheduler 负责的是细粒度的调度，未来这个部分需要扩展

现在要做的工作就是仲裁

以及decode的仲裁粒度还可以讨论下



（还有一个隐患，在这里记录一下：如果是multi model serving的场景，会有一个风险——未来如果出现超长上下文，则prefill（计算密集型负载）会更多）



### ExecUnit抽象基类

```Python
class ExecUnit():
    # 这里以及往下的args只代表没想好放什么参数，不代表这个args就是一个args
    # 初始化函数，接收资源，只进行传递
    def __init__(self, args)
    
    # 核心方法，定义步进逻辑
    def step(self)
    
    # 核心方法，由scheduler调用，负责给当前exec unit一个活跃的任务
    def recv_job(self)
```



#### 实现类：`LLMExecUnit`和`OmniModelExecUnit`

这样我们可以分别写两个类来继承这个ExecUnit：`LLMExecUnit`和`OmniModelExecUnit`

从而，我们可以在这两个包裹类写patch和兼容化接口（比如初始化时安装patch），也可以自己import omni的代码然后修改（比如我们现在已经给omni提供了一个资源provider，用来接入stream和graph资源）



### MiniContainer（？）

我们称这个类为minicontainer，主要是因为这个类负责和更外层的unitScheduler（姑且这么叫）和各类ResourceManager交互，而且实际上，miniContainer\-owned的资源在container终止的时候就可以释放了，也就是说，资源的生命周期是和container强绑定的

```Python
class MiniContainer():
    # 根据execUnit的资源需求申请资源，创建execUnit
    # 这样container可能需要持有resourceManager的引用？
    def __init__(self)
    
    # 这里默认就是用当前stream
    # graph参数的作用：主要对于vLLM的
    # step结束自动调用UnitScheduler的Schedule方法？
    def step(stream=None, **, graph)
    
    # 获取这个执行单元需要的执行资源的定义
    def get_resc_desc(self)
    
    # 获取性能数据，未来schedule有用
    def get_prof_res(self)
    
    # 核心方法，由scheduler调用，负责给当前exec unit一个活跃的任务
    def recv_job(self)
    
    # 关闭容器，由容器释放申请的资源（？）
    # 不过也不一定，可能由更高层的MultiModelWorker来释放资源也不为过，取决于谁执行shutdown动作
    def shutdown(self)
```



### UnitScheduler

```Python
# 未来可能可以考虑根据资源变化动态调整分区
class UnitScheduler():
    # 接收一个新工作，这是目前为数不多能改变分区状态的方法
    def recv_job(self, job)
    
    # 这里应该是某个job完全完成——另一种可能可以改变分区状态的方法
    def job_finished(self, job)
    
    # 调度，这里也是没想好传递什么参数
    # 逻辑应该是检查：当前完成调度的工作是否需要转换分区进行调度
    # 需要的话就进一步等待完成事件
    # 否则无事发生
    # 当然这里也可以考虑用同步逻辑？不知道会有多少性能开销
    # 以及还有个问题，这种schedule方式，如果出现资源动态变化的情况，目前绑死在参数传入的任务上有点难搞
    # 还是说我们可以记录参数传入的任务的pair 任务，一起等待？（怎么个一起等待+统一调度法子）？
    def schedule(self, args) 

    # 创建新的执行单元
    # 一般来说就是初始化就能做了
    # 不过允许追加
    def create_exec_unit(self, args)
    
    # 关闭执行单元们
    def shutdown(self)
```



### ResManager

#### CommManager

没想好（通信和多种并行部署方案是一个复杂的问题）

所以目前先只抽象两个方法

```Python
# 目前只支持管理同构TP
class CommManager():
    # 创建多个并行组（？）
    # 取决于通信方法
    def create(self, args):->key, comm groups
    
    # 根据特定key获取并行组
    def get(self,key):->comm groups
    
    # 释放并行组
    def release(self, key)
```





#### GCCudaGraphManager


原先的cuda graph manager采用的key只有batchdesc

需要额外加上stream

需要统一管理graph capture**和input/output buffer**

不然async output还需要额外同步一次，这个不太好接受

接口略（能否只针对官方的类做一些继承拓展）



#### MemoryManager和MemoryProfiler

还没想好



#### GCStreamManager

```Python
class GCStreamManager()
    # 创建目前看来只要传入数字？你只负责管
    def create(self, args): -> key, pair/list[stream]
    
    # 根据key获取stream资源，一般是scheduler调用
    def get(self, key)
    
    def release(self, key)
```



### MultiModelWorker

大体接口应该能考虑复用一般的worker？

但是初始化方法需要额外创建上述manager，以及等价地创建执行单元

此外就是处理请求和请求路由的方法？需要调用Scheduler的提交任务的方法




