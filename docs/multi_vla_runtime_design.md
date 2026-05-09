# 两类 VLA 共存时，runtime / pipeline 应该怎么设计

这份文档不再只看 `wall-x` 一个模型，而是把下面两套系统放在一起思考：

- `wall-x`
  一个偏 **单模型高性能推理** 的 VLA，核心是本地多模态模型推理，重点在 `Flow Action`、`KV cache`、ODE 循环、量化和 C++ runtime。

- `/Users/sam/project/ai/amap_dog/vla`
  一个偏 **机器人任务编排** 的 VLA 系统，核心是 ROS2 节点、云边通信、任务状态管理、导航/简单动作/问答等能力分发。

把这两者放在一起看，结论会比“怎么把一个模型推快”更明确：

> **未来真正需要的不是“一个 VLA 推理引擎”，而是“一个多 VLA 能力 runtime”。**

也就是说，系统要同时处理：

- 高性能本地动作模型
- 云端或远程视觉语言服务
- 导航型 VLA
- 机械臂/操控型 VLA
- VQA / 诊断型模型
- 传统规划器 / controller / safety 模块

这时设计重点已经从“单模型推理优化”变成了：

> **怎么把不同类型的 VLA 能力组织成一个有 deadline、有状态、有回退策略的实时系统。**

---

## 1. 先看这两个项目分别代表什么

### 1.1 `wall-x` 代表的是“模型 runtime”

从当前实现看，`wall-x` 的重点是：

- 单个 VLA 模型如何在本地高效执行
- 多模态 token 如何组装
- `Flow Action` 如何做 prefix-prefill + postfix-only ODE
- `KV cache` 如何截断复用
- `INT8/W8A8` 如何嵌进推理路径
- 如何把 Python 调度开销换成 C++ runtime

换句话说，`wall-x` 擅长的是：

> **把一个复杂的 VLA 模型本身跑到接近硬件极限。**

它解决的是“模型内部”的问题。

### 1.2 `amap_dog/vla` 代表的是“任务 runtime”

另一个项目则明显是另一类东西。

从代码结构看：

- 有 `vla_control_node`
- 有 `vla_worker_node`
- 有 `state_manager`
- 有 websocket / rtc 通信
- 有云端/本地推理切换
- 有 `VLAType` 分发
- 有导航、point、follow、simple_action、VQA 等不同任务流

它擅长的是：

> **把不同能力串成一个机器人任务系统。**

它解决的是“任务外部”和“系统协同”的问题。

### 1.3 两者不是竞争关系，而是不同层

这两个项目放在一起，不是二选一，而是天然分层：

- `wall-x` 更适合做 **模型执行内核**
- `amap_dog/vla` 更适合做 **系统编排外壳**

所以未来的架构重点不是“选哪个”，而是：

> **如何让“任务 runtime”调用多个“模型 runtime”。**

---

## 2. 如果系统里有两个 VLA，真正的问题是什么

一旦系统里不止一个 VLA，问题会立刻变化。

原来单模型时你问的是：

- 怎么更快
- 怎么量化
- 怎么减少同步
- 怎么管理 KV cache

现在多模型时你必须问：

- 哪个任务该走哪个 VLA
- 哪个 VLA 在本地跑，哪个走云端
- 两个 VLA 是否共享图像预处理和状态缓存
- VQA、导航、操控是否要抢同一块 GPU
- deadline 冲突时先保哪个能力
- 不同 VLA 输出如何统一成控制接口
- 出错时谁负责 fallback

这说明系统的抽象必须升级。

> **你不再是在设计“模型推理流程”，而是在设计“能力调度系统”。**

---

## 3. 这两个 VLA 的职责边界应该怎么分

如果强行让两套 VLA 平级竞争，很快会乱。

正确做法是先给它们定角色。

### 3.1 `wall-x` 的最佳角色

最适合定位为：

- 本地高性能动作模型
- 近实时 / 短时域动作生成器
- 机械臂/操控闭环的低层策略模型
- 本地视觉条件动作生成能力

关键词：

- 高频
- 低延迟
- GPU 常驻
- deadline 强
- 尾延迟敏感

### 3.2 `amap_dog/vla` 的最佳角色

最适合定位为：

- 系统级任务编排器
- 高层任务理解与能力路由
- 云边协同入口
- 导航、跟随、point goal、simple action、VQA 的任务外壳

关键词：

- 多任务
- 多状态
- 多节点
- 可接云端
- 任务粒度比控制粒度大

### 3.3 关键区别

可以简单理解成：

- `wall-x` 负责 **怎么做动作**
- `amap_dog/vla` 负责 **什么时候做、做哪种能力、出了问题怎么办**

如果把这两个职责混进一个大节点里，后面会很难维护。

---

## 4. 所以 runtime 应该从“单模型”升级成“三层系统”

我建议直接按三层来设计。

### 4.1 第一层：Capability Runtime Layer

这一层是“能力执行层”，每个 VLA 都被包装成一个统一能力接口。

典型能力：

- `ManipulationVLA`
  由 `wall-x` 这类本地 `Flow Action` 模型实现

- `NavigationVLA`
  由 `amap_dog/vla` 当前的导航/point/follow 等能力实现

- `VQACapability`
  由本地或云端 VLM 实现

- `SimpleActionCapability`
  前进、后退、转圈、打招呼等离散能力

每个能力都要提供统一接口，例如：

```text
prepare(request, world_state)
infer(session)
postprocess(result)
fallback(reason)
```

注意：

> 这里的“统一”不是统一模型结构，而是统一调度接口。

### 4.2 第二层：Policy Orchestrator Layer

这一层是“任务编排层”。

它负责：

- 根据任务类型选择能力
- 处理任务优先级
- 管理 deadline
- 做能力切换
- 合并多模型输出
- 管理 fallback

例如：

- “去会议室 304” -> `NavigationVLA`
- “走到人前面然后打招呼” -> `NavigationVLA + SimpleActionCapability`
- “抓起红色杯子” -> `ManipulationVLA`
- “看看前面有没有垃圾桶” -> `VQACapability`

这一层更接近 `amap_dog/vla` 当前的 `control_node + worker_node + state_manager` 逻辑。

### 4.3 第三层：Robot System Layer

这一层是 ROS2 / 控制 / 感知 / 安全系统。

它负责：

- 相机采集
- odom / tf / map / GNSS
- 控制器接口
- planner / DWA / occupancy map
- 安全策略
- actuator 命令下发

这层不该知道 `Flow Action` 或 `ODE` 细节，它只关心：

- 请求
- 状态
- 控制输出
- 超时和故障

---

## 5. 多 VLA 系统里，最重要的不是模型，而是共享状态

如果有两个 VLA，最容易出问题的是状态各自维护，互相不认。

所以你应该先统一 **World State**，而不是先统一模型类。

### 5.1 统一的 World State 应该包含什么

建议至少包含：

- 多相机图像缓存
- 当前机器人位姿
- 当前地图/目标状态
- 最近 N 帧观测
- 最近动作历史
- 当前任务上下文
- 当前安全状态
- 当前 deadline / 预算

也就是：

```text
WorldState
  ├── ObservationState
  ├── RobotState
  ├── TaskState
  ├── SafetyState
  └── RuntimeBudgetState
```

### 5.2 为什么它比统一模型更重要

因为两个 VLA 真正共享的不是权重，而是：

- 图像
- 里程计
- 时间
- 任务上下文
- 控制预算

如果这层不统一，会出现：

- 两套系统各自 resize 一遍图片
- 各自维护一份 pose
- 各自做一份任务状态机
- 出现冲突时没人裁决

这才是多 VLA 系统最先会炸的地方。

---

## 6. 要把“模型路由”升级成“能力路由”

很多团队一开始会想做：

> 输入来了一下，判断走哪个模型。

这太浅了。

真正应该路由的是 **能力**，不是模型。

### 6.1 为什么不是模型路由

因为同一个能力可能有多个实现：

- 本地 `wall-x`
- 云端 VLM
- 传统 planner
- 简单规则引擎

例如：

- “看前面有没有障碍物”
  可以走云端 VQA，也可以走本地 VQA，也可以走传统感知模块

- “向前走 2 米”
  可以走 simple action，也可以走 DWA，也可以走导航 VLA

### 6.2 所以要路由的是：

```text
Task -> Capability -> Implementation
```

而不是：

```text
Task -> Model
```

这个区别非常大。

因为前者给你保留了：

- fallback 空间
- 本地/云端切换空间
- A/B 测试空间
- 延迟预算调度空间

---

## 7. 多 VLA 系统的调度原则应该是什么

有两个 VLA 后，最需要明确的是优先级。

建议按下面三类分。

### 7.1 Hard-real-time-like 能力

例如：

- 近实时动作生成
- 动作修正
- 避障控制

特征：

- deadline 强
- 对 jitter 敏感
- 一旦超时，后果直接体现在控制不稳定

这类能力优先走：

- 本地
- 常驻 session
- 预分配 buffer
- 高优先级 stream / queue

这里 `wall-x` 更适合承担。

### 7.2 Soft-real-time 能力

例如：

- 导航子目标判断
- 路径更新
- point/follow 策略判断

特征：

- 也有 deadline
- 但比控制环稍松
- 可以和规划器配合

这里更适合由 `amap_dog/vla` 这样的 orchestrator 统一管理。

### 7.3 Best-effort 能力

例如：

- VQA
- 情绪回应
- 解释型输出
- 监控图像上报

特征：

- 最适合云端或低优先级本地资源
- 不能抢占控制主链

---

## 8. 具体到你的系统，我建议的整体架构

### 8.1 不要让 `wall-x` 直接替代 `amap_dog/vla`

正确关系应该是：

- `amap_dog/vla` 做系统编排
- `wall-x` 作为一个本地能力后端挂进去

也就是说：

```text
amap_dog/vla
  └── CapabilityManager
        ├── wallx_local_manipulation
        ├── local_vqa
        ├── cloud_vqa
        ├── navigation_policy
        └── simple_action_executor
```

### 8.2 `wall-x` 不应该暴露“模型类”，而应该暴露“能力服务”

对上层来说，最好不是直接看到：

- `generate_flow_action()`
- `generate_text()`

而是看到：

- `predict_manipulation_action(request)`
- `answer_visual_query(request)`

这样上层 orchestrator 不需要懂 token、ODE、KV cache。

### 8.3 `amap_dog/vla` 的 control/worker 思路是对的，但抽象还不够高

当前它已经有：

- control node
- worker node
- state manager
- service config
- cloud/local inference switch

这已经很接近真正的系统 runtime 了。

下一步缺的是：

- 更明确的 capability interface
- 更明确的 deadline / priority 模型
- 对本地大模型 session 的统一管理
- 多能力共享 world state

---

## 9. 建议的核心模块

如果按两个项目合并思路来演进，我会建议最后出现下面这些模块。

### 9.1 `WorldStateManager`

统一维护：

- 图像 ring buffer
- pose / tf / map state
- 任务上下文
- 最近动作和结果

### 9.2 `CapabilityRegistry`

注册系统里可用的能力：

- navigation
- manipulation
- local_vqa
- cloud_vqa
- follow
- simple_action

### 9.3 `RuntimeSessionManager`

专门管理本地重模型 session：

- `wall-x` manipulation session
- 本地 VQA session
- 未来其他本地模型 session

负责：

- GPU 资源占用
- warm session
- KV cache / workspace 生命周期
- quant profile

### 9.4 `TaskOrchestrator`

负责任务到能力的路由：

- capability planning
- deadline 分配
- fallback
- 多阶段任务拆分

### 9.5 `ActionArbiter`

这是多 VLA 系统里非常关键但很容易漏掉的模块。

负责：

- 多能力输出冲突裁决
- 例如导航在前进，但操控模型请求停住
- 或 VQA 发现前方障碍，需要打断当前任务

没有这个模块，多 VLA 共存最后一定会打架。

---

## 10. 一个更合理的任务流长什么样

以“走到人前面然后打招呼，再回答问题”为例：

```text
用户任务输入
  -> TaskOrchestrator
  -> 解析为多个能力阶段
     1. NavigationVLA: 走到目标前
     2. SimpleActionCapability: 打招呼
     3. VQACapability: 回答问题

运行时：
  WorldStateManager 持续更新图像/位姿
  RuntimeSessionManager 维护本地模型 session
  ActionArbiter 决定当前哪条输出生效
  Robot System Layer 执行控制
```

如果换成操控任务：

```text
用户任务输入
  -> TaskOrchestrator
  -> 若需要移动底盘先走 navigation
  -> 进入操作区域后切到 wall-x manipulation
  -> ManipulationVLA 持续输出局部动作
  -> Safety / Controller 监控执行
```

这才是“两类 VLA 共存”时真正的 runtime。

---

## 11. 对你最重要的一个设计判断

现在最应该避免的是这两种错误路线。

### 11.1 错误路线一：做一个超级大统一模型入口

例如：

```text
infer(task_type, images, state, instruction, ...)
```

里面 if/else 越来越多，最后所有逻辑都塞进一个巨函数。

结果：

- 模型逻辑和系统逻辑耦死
- 多 VLA 很难演进
- fallback 很难做

### 11.2 错误路线二：每个 VLA 各自维护自己的小世界

结果：

- 状态不一致
- 资源冲突
- 调度失控
- 没有统一优先级

### 11.3 正确路线

应该是：

> **统一世界状态 + 统一任务编排 + 多个能力 runtime。**

而不是：

> 一个超大模型入口，或者多个完全孤立的小系统。

---

## 12. 你下一步最值得做的事

如果按工程优先级，我建议是：

1. 先定义统一的 `WorldState`
2. 把 `wall-x` 封装成一个 `ManipulationCapability`
3. 把 `amap_dog/vla` 现有能力整理成 `CapabilityRegistry`
4. 单独实现一个 `TaskOrchestrator`
5. 再补 `RuntimeSessionManager` 和 `ActionArbiter`

注意这个顺序。

不要一上来就做：

- 分布式
- 多 GPU
- 通用插件系统
- 超复杂调度器

先把单机、多能力、统一状态、明确优先级跑通。

---

## 13. 一句话总结

把 `wall-x` 和 `amap_dog/vla` 放在一起看，最重要的变化是：

> **你现在设计的已经不是“一个模型怎么推理”，而是“多个 VLA 能力如何在同一个机器人系统里协同工作”。**

所以最终需要的不是：

- 一个更大的单模型推理框架

而是：

- **一个统一的多 VLA capability runtime**

再压缩一句就是：

> **`wall-x` 负责把模型跑快，`amap_dog/vla` 负责把任务跑通；真正的下一层系统，要把这两者组织起来。**

