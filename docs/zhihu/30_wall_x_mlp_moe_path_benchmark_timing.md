# 30. Wall-X 的 MLP MoE path 到底什么时候值得做 end-to-end path benchmark

一路把 `Wall-X` 的 `MLP MoE` 路径写到这里，讨论已经自然分成了两层：

- 一层是局部 kernel / stage 级问题  
- 另一层是 runtime / path 级问题

前面几篇文章里，我已经把局部层面的很多东西拆开了：

- `gate/up/down` 的 fused compute
- `low-precision` 的优先落点
- `scatter/output` 什么时候值得继续 fused
- `down+scatter` 什么时候值得联合优化

但只要这些局部工作做得越来越多，马上就会遇到一个非常现实的问题：

> 到底什么时候，应该停止只看单个 kernel benchmark，而开始认真做这条 `MLP MoE path` 的 end-to-end benchmark？

这个问题如果回答得太晚，很容易让系统陷入一个常见误区：

> 每个 kernel 看起来都越来越漂亮，但整条路径的收益始终说不清楚。

这篇文章就只讲这个判断。

## 一、先说结论：当局部优化已经开始相互影响时，就该把 benchmark 从 kernel 级提升到 path 级

如果把我的判断压成一句话，我会说：

> 对 `Wall-X` 的 `MLP MoE` 来说，一旦你的优化已经不再只是“替换一个独立 kernel”，而开始影响输入重排、输出回写、low-precision 命中、fallback 行为和 planner 决策，那么 benchmark 就不该只停留在 kernel 级，而应该上升到 end-to-end path 级。

这里最关键的是“开始相互影响”这五个字。

因为 path benchmark 真正要解决的问题，不是：

> 某个 kernel 到底快不快

而是：

> 这条路径作为 runtime 中的一个执行单元，到底值不值得被保留、继续深化、或者提升为常规候选 path。

这两件事不是同一个问题。

## 二、为什么 kernel benchmark 很容易不够用

这件事其实非常常见。

很多优化在早期阶段，的确更适合先看 kernel benchmark。  
因为那个阶段你最关心的是：

- 单个 kernel 有没有明显收益
- 算法或 tile 设计有没有方向性错误
- 某个 low-precision 方案是不是完全不成立

但随着系统往下走，kernel benchmark 很快就会遇到三个天然局限。

### 1. 它看不到前后路径成本

例如：

- `layout prepare`
- `scatter/output`
- input/output adapter

这些东西并不会自然出现在某个单一 kernel benchmark 里。

### 2. 它看不到 runtime 真实命中情况

在系统里，一条路径最后到底有没有被稳定命中，往往还受这些因素影响：

- shape 稳定性
- packed weight 是否就绪
- planner 是否选择了它
- fallback 是否频繁发生

这些东西在局部 kernel benchmark 里通常是缺位的。

### 3. 它看不到“局部快了但全局没变”的问题

这其实是最危险的。

因为你很容易在局部 benchmark 里看到很漂亮的数字，但放回完整 path 之后才发现：

- 上游更重了
- 下游更重了
- 总时延没怎么变
- jitter 反而变差了

所以 kernel benchmark 在早期很重要，但它天然不是终点。

## 三、那什么时候还“不值得”上 path benchmark

我不认为应该从第一天起就把所有事情都上升到 path 级。

如果下面这些情况还没出现，那太早做 end-to-end path benchmark，收益未必高。

## 四、第一种情况：你还在验证单个 kernel 是否基本成立

如果当前还处在这种阶段：

- 某个 fused compute 还不稳
- low-precision 还只是实验性的
- pack format 还反复在换

那这个时候更重要的还是先把局部问题做清楚。

因为 path benchmark 一旦引入，观察成本会更高。  
如果局部实现本身还没站稳，它反而可能让你更难看清真正的问题。

## 五、第二种情况：path 边界本身还没稳定

也就是你还没明确：

- `MLP MoE path` 到底从哪里开始
- 到哪里结束
- output 回接是否已经固定
- 某些后半段是否还在反复调整

这时做 end-to-end path benchmark，很容易测出一组数字，但路径边界本身还在漂。

这样得到的数据，对后面长期系统设计价值并不高。

## 六、第三种情况：planner 还没开始真正把它当 path 选

如果当前系统里还没有下面这些能力：

- planner 能显式区分这条 path
- fallback 能显式记录它
- benchmark 能显式归类它

那么这时就算你做了“end-to-end path benchmark”，也很容易只是手工拼一个局部实验，而不是测真实 runtime path。

这会让 path benchmark 看起来存在，但其实还没真正进入系统语义。

## 七、什么时候“值得”认真上 end-to-end path benchmark

如果要说积极条件，我更认可下面四条。

## 八、第一条：这条 path 已经有了明确的起止边界

这是最基本的一条。

你至少要能清楚回答：

- path 从哪个输入形态开始
- 中间包含哪些 stage
- path 在哪里回接主干或 output adapter

没有这条边界，path benchmark 只是一个模糊的“某一串代码的总时间”。

## 九、第二条：局部优化已经开始互相作用

这通常是最重要的时机信号。

例如你已经同时在做：

- `BF16 fused`
- low-precision
- packed weight
- `scatter/output` 优化
- `fallback` 规则

这时候如果还只盯着单个 kernel，很容易失真。

因为这些东西已经在共同决定：

> 这条 path 到底像不像一条真正的执行路径。

## 十、第三条：planner 已经开始把它作为候选 path

一旦 planner 开始能说出：

- `local_mlp_moe_fused`
- `low_precision_local_mlp_moe`

这类路径，end-to-end path benchmark 的价值就会迅速上升。

因为它不再只是给人类看的实验数据，而会开始真正影响 runtime 决策。

## 十一、第四条：fallback 和 capability 已经开始影响真实命中率

这一点很关键。

只要你开始发现下面这些问题：

- 某条 path 理论上很好，但经常命不中
- 某条 low-precision path benchmark 漂亮，但 runtime 常退回
- 某种 pack format 在实验里成立，但系统里命中率不稳

那这时候，path benchmark 就已经不是“可选项”，而是必须品。

因为你需要看的已经不再是：

> 理论执行性能

而是：

> 实际执行行为

## 十二、一个更现实的 path benchmark 应该测什么

如果现在要给 `MLP MoE path` 做一版真正有系统价值的 benchmark，我会至少要求它回答下面这些问题。

## 十三、第一类：这次 path 是怎么被选中的

也就是：

- 当前 workload 是什么
- 当前 planner 为什么选了这条 path
- 当前 precision policy 是什么

如果这部分不记录，你后面看到结果时就很难知道“为什么是这条 path”。

## 十四、第二类：这次 path 实际是怎么命中的

例如：

- 是否真的命中了预期 path
- 是否发生 fallback
- 实际命中的 precision family 是什么
- pack/layout 是否按预期命中

这一步是 kernel benchmark 完全看不到，但 path benchmark 非常需要看到的东西。

## 十五、第三类：这条 path 的 stage breakdown

我会建议至少包含：

- layout/routing
- core compute
- scatter/output
- path total

这样你才能真正知道：

> 这条 path 到底是被哪一段拖住了。

## 十六、第四类：稳定性信息

也就是：

- P50 / P90 / P99
- fallback 频率
- 命中率
- 可能的 jitter

对 `VLA Runtime` 来说，这些信息往往比“最好看的一次 latency”更重要。

## 十七、为什么 path benchmark 是 planner 进化的真正起点

我越来越觉得，planner 什么时候真正开始像一个 runtime 核心模块，取决于它什么时候开始吃 path benchmark，而不只是吃局部 benchmark。

因为局部 benchmark 更像是在回答：

- 某个组件值不值得存在

而 path benchmark 更像是在回答：

- 某个组件在 runtime 语境里，值不值得被经常选中

这两件事之间，差着整整一层系统语义。

所以从 planner 的视角看，path benchmark 通常意味着：

> 这条路径已经不只是一个实验性实现，而开始进入运行时治理范围。

## 十八、为什么 path benchmark 又不能取代 kernel benchmark

这里我也不想走到另一个极端。

我并不是说一旦有了 path benchmark，就不需要 kernel benchmark 了。  
恰恰相反，我更愿意把它们看成两个层级：

- kernel benchmark 负责看局部实现是否有方向性价值
- path benchmark 负责看这条路径放回 runtime 后是否真的成立

只有这两层同时存在，系统的优化闭环才会比较完整。

## 十九、我的最终判断

如果把这篇文章压缩成一句话，我想留下的结论是：

> `Wall-X` 的 `MLP MoE` 路径一旦开始同时引入 fused compute、low-precision、packed weight、planner 选择和 fallback 语义，就不应该再只停留在单个 kernel benchmark，而应该正式上升到 end-to-end path benchmark，用来回答这条路径在真实 runtime 里到底值不值得被长期保留和继续深化。

因为对 `VLA Runtime` 来说，最值钱的问题从来不只是：

> 某个 kernel 快了多少。

而是：

> 这条 path 作为系统里的正式路径，到底能不能稳定命中、稳定解释、稳定带来收益。
