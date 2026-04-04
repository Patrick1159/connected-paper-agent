# Future Ideas

## 1. 工程：arXiv 请求队列与负载均衡

### 目标
为所有 arXiv 请求维护一个统一队列，避免当前串行限流 + 局部重试导致的请求拥塞、429 集中爆发和整体吞吐不稳定。

### 动机
- 当前已经有全局限流，但仍可能在 title search、metadata fetch、pdf download 阶段出现局部突发
- 429 的指数回退会提升稳定性，但也会拉长整体 pipeline 时间
- 需要一个比“每次请求临时 sleep”更系统的调度层

### 初版思路
- 建立统一 `ArxivRequestQueue`
- 所有 arXiv 相关操作统一入队：
  - metadata fetch
  - title search
  - pdf download
- 队列调度器负责：
  - 节流
  - 优先级分配
  - 失败重试
  - 429 后动态降速
  - future worker pool / async dispatch

### 可考虑的策略
- 不同请求类型设置不同优先级
  - metadata fetch > title search > pdf download
- 同一论文链路上的关键请求优先
- 对 429 引入全局 cooldown，而不是只对单个请求 sleep
- 维护简单指标：
  - 队列长度
  - 平均等待时间
  - 429 频率
  - 各请求类型成功率

### 预期收益
- 均衡请求负载
- 降低 burst 导致的 429
- 为未来并行/异步化铺路
- 更容易做全局观测和调优

---

## 2. Idea（1.0 版本）：纯 heuristic 探索 + sub-agent 终止评估

### 目标
引入一个专门的 sub-agent / evaluator，在 tracing 过程中判断：
- 当前 explore 方向是否正确
- 是否已经偏离 root paper 太多
- 是否应该继续探索当前分支
- 是否应该提前停止整体探索

### 核心定位
这个 sub-agent 不负责扩展图，而是负责“探索控制”。
它像一个 reviewer / critic，对当前图状态、候选链、节点摘要做 heuristic 评分。

### 可评估问题
- 当前 frontier 与 root paper 的问题设定是否仍然一致
- 方法演进是否连续，还是已经跳到旁支领域
- 当前新增论文是否只是弱相关引用，而非核心演化节点
- 当前继续扩展的收益是否已经下降
- 当前是否已经找到足够好的一条 lineage

### 初版可做成纯 heuristic
不强依赖复杂 agent 规划，先从规则和分数开始：
- relevance to root
- method continuity
- problem continuity
- citation-chain integrity
- chronology consistency
- novelty / marginal gain of expanding one more round

### 可输出信号
- `continue_exploration: true/false`
- `branch_score`
- `drift_score`
- `confidence`
- `reason`

### 可能的接入点
- 每轮结束后，评估是否继续下一轮
- frontier 生成后，评估哪些分支值得保留
- evaluation 前，判断图是否已经足够形成高质量 lineage

### 潜在收益
- 避免探索过深、跑偏、浪费 token 和 arXiv 配额
- 让 lineage 更聚焦 root paper
- 为后续“搜索深度”和“最终链长”解耦提供控制层

---

## 3. 可参考的产品思路：Deep Research / 高阶研究代理

### 说明
没有它们的内部实现细节，不能准确断言其具体工程方案。
但从公开表现和通用 agent 设计经验看，这类系统通常会包含以下能力：

### 常见模式
1. **Planner / Executor 分层**
   - 一层负责规划研究路径
   - 一层负责执行检索、阅读、总结
   - 一层负责审查当前结果是否足够

2. **显式中间状态**
   - 维护 research state / evidence table / source map
   - 不是每步都从零开始，而是持续积累可追踪证据

3. **多阶段检索与重排**
   - 先广召回，再逐步收缩
   - 不是一次就决定最终链条，而是反复筛选

4. **Critic / Reflection 机制**
   - 周期性判断：方向是否正确、证据是否不足、是否需要补检索
   - 和你提出的 sub-agent 思路非常接近

5. **预算感知**
   - 对时间、请求数、token 数、信息增益做权衡
   - 达到“收益递减”时停止

### 对本 repo 可借鉴的优化方向
1. **引入 Planner / Critic 双层结构**
   - 当前主 agent 负责 tracing
   - 新增 critic sub-agent 负责：
     - 打分
     - 判断 drift
     - 判断 stop / continue

2. **把“扩图”和“选链”分开**
   - tracing 阶段尽量稳定找候选
   - evaluation 阶段再做高质量选链
   - 中间由 critic 控制探索边界

3. **显式维护 evidence / score 表**
   - 为每个 node 增加：
     - relevance score
     - drift score
     - lineage potential score
     - expansion value score

4. **从固定轮数改成预算 + 收益递减停止**
   - 不只看 `max_rounds`
   - 再加：
     - 最大请求预算
     - 最大 token 预算
     - 连续若干轮低收益则停止

5. **候选链先枚举，再打分**
   - 不只依赖 LLM 直接输出 chain
   - 先从 citation graph 枚举少量候选路径
   - 再让 evaluator / LLM 从候选中选最佳

### 对 1.0 最实用的落地建议
优先级建议：
1. arXiv 请求队列
2. critic sub-agent（纯 heuristic 版）
3. node / branch scoring
4. 停止条件从固定轮数升级为“固定轮数 + 收益递减”
5. 候选链枚举 + LLM 重排

这几项都能明显提升稳定性与 report 质量，而且不会把工程复杂度一下抬得太高。

---

## 4. 一个建议中的阶段化路线

### v1.1
- 请求队列
- 429 全局 cooldown
- 分支/节点 heuristic scoring

### v1.2
- critic sub-agent
- drift detection
- stop / continue 决策

### v1.3
- 候选 lineage 枚举
- score-based fallback
- `max_rounds` 与 `max_lineage_length` 解耦

### v2.0
- async worker pool
- 动态预算控制
- 更完整的 planner / executor / critic 结构

---

## Note
以上更偏工程与 agent 设计思路，不代表必须一次性全部实现。当前 repo 已经具备不错的一阶段基础，建议继续按“最小可验证改动”迭代。
