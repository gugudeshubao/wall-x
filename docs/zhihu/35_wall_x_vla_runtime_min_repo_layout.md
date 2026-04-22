# 35. VLA Runtime 的最小目录结构应该怎样组织

一路把 `VLA Runtime` 的对象、路径、策略和阶段都讲到这里，下一步其实很自然会落到一个更工程化的问题：

> 如果真的开始把这套 runtime 往仓库里落，目录结构应该怎么组织？

这是一个很容易被低估的问题。

因为很多系统在概念阶段其实什么都能讲清楚：

- 有 planner
- 有 benchmark
- 有 capability matrix
- 有 report
- 有 backend

但一旦开始写进代码库，如果目录结构没有提前想清楚，最后很容易变成：

- 逻辑都存在
- 但散在一堆看似顺手的位置
- 每层都在偷持有别层的对象
- 系统语言最后还是没有真正落地

所以这篇文章就只讲一件事：

> `VLA Runtime` 的最小目录结构，到底应该怎样组织？

这里我强调的是“最小目录结构”，不是“大而全平台目录”。

因为这个阶段最重要的不是让 repo 看起来像个大框架，而是：

> 让前面已经讲清楚的系统对象和分层，能在代码目录里有明确的着陆位置。

## 一、先说结论：目录结构不该按“谁写起来顺手”来长，而该按“哪些对象和边界必须长期稳定”来长

如果把我的判断压缩成一句话，我会说：

> `VLA Runtime` 的最小目录结构，不应该按当前哪个文件最方便往里塞代码来组织，而应该围绕那些必须长期稳定的对象和边界来组织。

这里的关键词是：

- 长期稳定
- 对象
- 边界

为什么？

因为目录结构本质上不是审美问题，而是：

> 你打算让哪些对象被谁正式持有、哪些边界被谁长期维护。

如果一开始就按“顺手”长，后面最常见的结果就是：

- planner 代码跑到 benchmark 旁边
- capability 描述藏在 backend 私有目录里
- report 在 logging 工具里另长一套
- path taxonomy 只存在于 README

这样系统会越来越难收敛。

## 二、为什么目录结构对 VLA Runtime 特别重要

如果只是单个推理脚本，目录结构当然没那么关键。  
但 `VLA Runtime` 一旦真的开始有这些层：

- planner
- backend
- benchmark
- report
- capability
- fallback

目录结构其实就在默默决定：

- 哪些对象是 runtime 核心对象
- 哪些对象只是工具层附属物
- 哪些模块在系统里拥有正式语义地位

也就是说，目录结构本身其实是在给系统“排座次”。

对 `VLA Runtime` 来说，这件事之所以特别重要，是因为它天然比很多系统更容易被几种不同逻辑拉扯：

- 模型逻辑
- backend 逻辑
- benchmark 工具逻辑
- runtime 治理逻辑

如果目录结构不提前把边界稍微立住，后面这些东西会越来越混。

## 三、最小目录结构首先不该怎么长

在说“该怎么长”之前，我更想先把几个常见坏味道说清楚。

## 四、第一种坏味道：所有 runtime 逻辑都塞进 `scripts/`

这在项目早期非常常见。

因为：

- 最快
- 最不需要解释
- 改起来顺手

但只要对象开始增多，问题会立刻出现：

- `ExecutionPlan` 没有正式归属
- `ExecutionReport` 只在某个脚本里临时生成
- benchmark、planner、backend 的边界全都模糊

所以如果已经决定真的要做 runtime，继续让核心对象长期停留在 `scripts/` 里，通常不是好主意。

## 五、第二种坏味道：按“技术实现类型”而不是按“系统语义”分目录

例如：

- 所有 CUDA 的东西放一边
- 所有 benchmark 的东西放一边
- 所有 config 放一边

这种方式在工具仓库里当然有意义。  
但如果你想表达 runtime 结构，它往往不够。

因为 runtime 最关键的问题往往不是：

- 这段代码用的是 Python 还是 CUDA

而是：

- 这段代码在系统语义上属于 planner 还是 backend

如果目录只按实现技术分，很容易让系统对象的所有权越来越模糊。

## 六、第三种坏味道：每个模型自带一套私有 runtime 目录

这会让系统很快失去“runtime 是一层独立逻辑”的可能。

因为一旦每个模型都在自己目录下偷偷长：

- planner 逻辑
- capability
- report

最后你得到的不是一个多模型 runtime，而是很多模型私有胶水层的集合。

所以如果前面那些文章真想落到工程里，这一点必须早点避免。

## 七、那最小目录结构到底该围绕什么来立

如果不按“技术实现”来分，也不按“脚本顺手”来分，那第一版最小目录结构到底该围绕什么来立？

我的答案很明确：

> 围绕 runtime 的核心对象和边界来立。

也就是说，最小目录结构最应该稳定承载的，至少是下面几类东西：

- workload / plan / report
- path / capability / taxonomy
- planner
- backend
- benchmark

这五类东西一旦在目录上有清楚位置，很多系统边界就会自然清楚很多。

## 八、一个更现实的最小目录草图

如果现在就要给 `VLA Runtime` 落一版最小目录，我会更推荐像下面这样理解。

## 九、第一层：`runtime/core/`

这层放 runtime 最核心、最稳定的对象定义。

我更建议这里放：

- `workload.py`
- `plan.py`
- `report.py`
- `path.py`
- `capability.py`

这层的意义很简单：

> 这些对象不应该散落在脚本和 backend 里，而应该有一个正式、稳定、被所有层共同依赖的位置。

## 十、第二层：`runtime/planner/`

这层放 planner 本身的逻辑。

例如：

- path selection
- precision policy 解释
- deployment mode 分流
- fallback chain 生成

为什么这一层要单独立出来？

因为前面已经反复强调过：

> planner 是 runtime 的决策中心，不是 benchmark 或 backend 的附属逻辑。

如果这一层没有正式位置，它很容易慢慢退化成 scattered if/else。

## 十一、第三层：`runtime/backends/`

这层放真正承接执行路径的 backend。

例如：

- baseline path
- fused path
- low-precision path
- graph path
- local megakernel experiments

这层应该是：

> 执行能力的实现层

而不是对象定义层。

也就是说：

- `PathCapability` 的正式语义对象可以在 `core`
- 但具体 backend 实现和它们的注册逻辑更适合放在这里

## 十二、第四层：`runtime/benchmark/`

这层放的不是“单个打点函数”，而是：

> 对 runtime 核心对象做聚合和解释的评估逻辑。

例如：

- workload grouping
- path-level benchmark
- report aggregation
- edge/cloud 统计视图

如果 benchmark 只是散落在 scripts 里，它很难真正成为 runtime 的反馈层。

## 十三、第五层：`runtime/integrations/`

这一层非常容易被漏掉，但我觉得很有必要。

它的职责不是定义 runtime 核心语言，而是：

> 把 runtime 接到具体模型或具体服务入口上。

例如：

- `wall_x/`
- 其他未来模型
- service / policy server 入口

这样可以避免一个很典型的问题：

> runtime 核心对象直接被塞进某个模型目录里，最后失去独立层次。

## 十四、为什么我没有把 benchmark、report、capability 都塞进同一层

这点值得单独说。

有些人可能会自然觉得：

> 既然这些对象都相关，是不是应该都放进一个“analysis”或“runtime utils”目录里？

我不太建议这么做。

原因是：

### `report`

更接近 runtime 核心事实对象，应该进入 `core`

### `capability`

也是 runtime 核心对象，应该进入 `core`

### benchmark

更像是消费这些对象的一层聚合与解释系统，应该单独存在

如果把它们都塞在一起，系统很快又会回到“对象有了，但所有权糊了”的老问题。

## 十五、为什么最小目录结构也必须承认“模型接入层”存在

这点我认为非常重要。

如果没有一层明确的模型接入层，系统很容易在两种极端之间摆荡：

### 极端 A

所有 runtime 逻辑都回流进具体模型目录。

### 极端 B

runtime 变得过度抽象，和真实模型接不起来。

所以我更推荐显式保留一层：

> `integrations`

它的意义在于：

- 不让模型私有逻辑污染 runtime 核心层
- 也不让 runtime 核心层脱离真实模型

这层通常就是系统演化时非常重要的缓冲带。

## 十六、一个更现实的目录示意

如果把上面的讨论收成一个最小草图，我会更推荐类似这种结构：

```text
runtime/
  core/
    workload.py
    plan.py
    report.py
    path.py
    capability.py

  planner/
    planner.py
    policies/

  backends/
    baseline/
    fused/
    low_precision/
    graph/
    experimental/

  benchmark/
    suites/
    aggregation/
    reports/

  integrations/
    wall_x/
    ...
```

这不是唯一正确答案。  
但它至少体现了一个我很看重的原则：

> 先按系统语义分层，再让不同实现技术填进去，而不是反过来。

## 十七、为什么这件事和多模型扩展直接相关

如果这套目录结构一开始就按 runtime 核心对象和边界来立，那么后面多模型扩展时，路径会清楚很多：

- 新模型更多加在 `integrations/`
- 新 backend 更多加在 `backends/`
- 核心对象仍然留在 `core/`

这会让你更自然地回答：

> 这是新增了一个模型，还是新增了一种 runtime 能力？

而如果目录一开始就是混的，后面这个问题通常会越来越难回答。

## 十八、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `VLA Runtime` 的最小目录结构，不该按“谁写起来顺手”或“用了什么技术”来组织，而应该围绕 runtime 核心对象和系统边界来组织：把 `workload/plan/report/path/capability` 放进 `core`，把决策逻辑放进 `planner`，把执行实现放进 `backends`，把聚合解释放进 `benchmark`，再用 `integrations` 作为模型接入层。

因为对这类系统来说，真正决定它后面能不能继续长大的，不只是：

> 现在代码能不能跑。

而是：

> 这些对象和边界，是否已经在仓库结构里有了正式的位置。
