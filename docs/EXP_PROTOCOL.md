# EXP_PROTOCOL.md — 实验协议（最高优先级，全程冻结）

> 本文件为全项目最高优先级协议。任何阶段的执行必须严格遵守本协议。
> 本协议一经冻结，不得单方面修改研究问题、环境设定、数据集结构或算法范围。

---

## 1. 研究问题

### 1.1 唯一主问题

> **在 Offline RL 中，state-action coverage 是否比 dataset size 更能决定最终性能上限？**

核心主对照：

```
small-wide  vs  large-narrow
```

- small-wide：数据量小，但 coverage 广
- large-narrow：数据量大，但 coverage 窄
- 控制变量：二者 transition 数量不相等，但各自内部 coverage 与 size 的维度分离

### 1.2 第二问题（从属问题）

> 不同数据质量条件下，BC 是否已足够；在何种数据条件下，CQL 等更保守的 offline RL 方法更占优？

- 第二问题只能作为从属问题
- 第二问题的实验依赖主问题结果完成后才允许展开
- 第二问题不得喧宾夺主，不得变成独立主线

---

## 2. 核心假设

**主假设（H1）：**
在统一任务、统一评估协议、受控数据构造的条件下，coverage 是决定 offline RL 策略性能上限的主导因素，而非 dataset size。具体而言：较小但 coverage 更广的数据集（small-wide）在训练的 offline RL 策略上，可以达到或超越较大但 coverage 更窄的数据集（large-narrow）所能达到的性能上限。size 的增大若不伴随 coverage 的扩展，其对性能上限的贡献有限。

**子假设（H1a）：**
small-wide 的最终性能上限 ≥ large-narrow，即 coverage 的作用可以抵消甚至超越 size 的优势。此为核心主对照的直接检验，二者 size 故意不相等（50k vs 200k），以检验 coverage 优势能否跨越 size 劣势。

**子假设（H1b）：**
当 coverage 足够广时，BC 已能达到可接受性能；当 coverage 窄时，CQL 相较 BC 的优势更明显。

**子假设（H1c）：**
上述关系在不同数据质量档（random / suboptimal / medium / expert / mixed）下表现出系统性规律。

---

## 3. 环境定义（冻结）

### 3.1 Env-A：主环境（唯一主环境）

| 参数 | 值 |
|---|---|
| 类型 | 15×15 deterministic FourRooms |
| 动作空间 | {up, down, left, right} |
| 转移 | 确定性 |
| Horizon | 100 |
| 奖励 | goal reward = +1，step penalty = −0.01 |

**作用：**
- 承担完整主实验（Coverage vs Size 四格矩阵）
- 承担 quality 调制实验
- 承担最核心机制分析
- Env-A 是唯一允许承担主结论的环境

### 3.2 Env-B：复验环境 1

| 参数 | 值 |
|---|---|
| 类型 | FourRooms-Bottleneck / Obstacle 风格 |
| 结构 | 比 Env-A 更强的 bottleneck 与绕行结构 |
| 转移 | 离散、确定性、可枚举 |

**作用：**
- 仅做复杂度复验（关键对照：small-wide vs large-narrow）
- 不承担完整主实验
- 不允许扩成与 Env-A 平级的完整四格实验

### 3.3 Env-C：复验环境 2

| 参数 | 值 |
|---|---|
| 类型 | Key-Door Maze / DoorKey 风格 |
| 结构 | 阶段依赖：先拿钥匙 → 开门 → 到 goal |
| 转移 | 离散、可定义 coverage |

**作用：**
- 仅做复杂度复验（关键对照：small-wide vs large-narrow）
- 不承担完整主实验
- 不允许扩成与 Env-A 平级的完整四格实验

**明确层级：Env-A 是主环境，Env-B / Env-C 是复验环境，三者不平级。**

---

## 4. 算法范围（冻结）

### 4.1 离散环境（必做）

| 算法 | 状态 |
|---|---|
| BC（Behavior Cloning） | 必做 |
| CQL（Conservative Q-Learning） | 必做 |

### 4.2 离散环境（增强项）

| 算法 | 状态 |
|---|---|
| IQL | 增强项，非必做 |

- IQL 只有在 BC + CQL 主实验全部稳定完成后，方可考虑加入
- 不允许在主实验阶段擅自引入其他算法

### 4.3 连续控制 benchmark（趋势复验用）

| 算法 | 状态 |
|---|---|
| BC | 必做 |
| CQL | 必做 |
| IQL | 必做 |
| TD3+BC | 必做 |

- 连续控制 benchmark 只做趋势复验
- 不承担主结论

---

## 5. 数据集矩阵（冻结）

### 5.1 Env-A 完整主实验数据矩阵

固定 quality = **medium**，正式数据集为：

| 条件 | Size | Coverage |
|---|---|---|
| small-wide | 50k transitions | wide |
| small-narrow | 50k transitions | narrow |
| large-wide | 200k transitions | wide |
| large-narrow | 200k transitions | narrow |

### 5.2 Coverage 构造原则

| 类型 | 构造方式 |
|---|---|
| wide | 4–6 个 medium checkpoints 混合生成，epsilon 较大 |
| narrow | 1 个 medium checkpoint 生成，epsilon 较小 |

- wide = 多 checkpoint 混合，覆盖更多 state-action
- narrow = 单 checkpoint 生成，覆盖集中
- "4–6 个"为 pilot 留有少量冻结空间，但不允许退化为单 checkpoint

### 5.3 Quality 调制实验数据集（Env-A）

固定 coverage 结构与 size，扫以下质量档：

| Quality 档 | 说明 |
|---|---|
| random | 完全随机策略 |
| suboptimal | 低质量行为策略 |
| medium | 中等质量（主实验默认档） |
| expert | 接近最优策略 |
| mixed | 多质量混合 |

### 5.4 Env-B / Env-C 复验数据集

默认只做关键对照：

| 条件 | 说明 |
|---|---|
| small-wide | 核心对照之一 |
| large-narrow | 核心对照之二 |

- 除非主实验完成后有明确理由，否则不扩展为完整四格

---

## 6. Seed 规则（冻结）

### 6.1 正式实验 Seed

- **数量：20 个随机 seeds**
- 适用：Env-A 主实验、Env-A quality 调制实验、Env-B/C 关键复验、连续控制 benchmark 正式结果
- 正式结果均基于 20 seeds 的均值与标准差

### 6.2 Pilot / Sanity Seed

- **数量：3–5 个随机 seeds**
- 仅用于前期检查与试运行
- 不得直接进入最终结果图

### 6.3 行为策略源 Seed

- **数量：8 个 DQN 训练 seeds**
- 用于构建行为策略池，支持 coverage 构造
- 不用于最终统计显著性计算

---

## 7. 评估指标

评估指标分为四层，各层服务不同阶段的执行目标。

### 7.1 主性能指标（服务 Phase 7 / 8 / 9 / 11 / 12）

| 指标 | 说明 |
|---|---|
| normalized return | 归一化 Episode Return，使用专家策略 return 归一化，主要汇报指标 |
| success rate | 到达 goal 的回合比例 |
| episode length | 到达 goal 的平均步数（成功回合内统计） |

### 7.2 主数据指标（服务 Phase 3 / 5 / 10）

| 指标 | 说明 |
|---|---|
| normalized state coverage | 数据集覆盖的独立 state 数量 / 环境总可达 state 数量 |
| normalized state-action coverage | 数据集覆盖的独立 (s, a) pair 数量 / 理论总 (s, a) pair 数量 |
| trajectory diversity | 数据集中不同轨迹的路径多样性度量（如独立轨迹起始状态数） |
| return coverage | 数据集中 return 分布的范围与密度（用于区分 quality 档） |

### 7.3 机制指标（服务 Phase 10）

| 指标 | 说明 |
|---|---|
| OOD-action tendency | 训练策略在 eval 中选择数据集未覆盖动作的频率 |
| behavior-support distance | eval 策略访问的 (s, a) 与数据集支撑集之间的距离度量 |
| Q-value overestimation proxy | CQL 训练过程中 Q 值超出数据集 return 范围的程度 |

### 7.4 统计指标（服务 Phase 12）

| 指标 | 说明 |
|---|---|
| 95% confidence interval | 所有正式结果均报告 20 seeds 的 95% CI |
| effect size | 核心对照（small-wide vs large-narrow）的 Cohen's d 或等价效应量 |
| 显著性检验 | 必要时使用 t-test 或 Mann-Whitney U 检验，报告 p 值 |

---

## 8. 图表计划概览

详见 `FIGURE_PLAN.md`。共规划 6 张核心图，不允许无序扩张。

| 图号 | 内容 | 所属阶段 |
|---|---|---|
| 图1 | Env-A 主实验四格矩阵总结果图 | Phase 7 |
| 图2 | small-wide vs large-narrow 核心对照图 | Phase 7 |
| 图3 | Env-A quality 调制结果图 | Phase 9 |
| 图4 | 机制解释图 | Phase 10 |
| 图5 | Env-B/C 关键复验图 | Phase 8 |
| 图6 | 连续控制 benchmark 趋势复验图 | Phase 11 |

---

## 9. 阶段执行流程（硬约束）

### 阶段列表

| 阶段 | 名称 | 关键输出 |
|---|---|---|
| Phase 0 | 协议冻结 | EXP_PROTOCOL.md, PROJECT_SCOPE.md, FIGURE_PLAN.md |
| Phase 1 | 环境搭建与单元测试 | Env-A/B/C 实现，单元测试通过 |
| Phase 2 | 行为策略池训练与分档 | 8 个 DQN seeds 训练完成，质量分档验证 |
| Phase 3 | 数据生成 pilot | 小规模 pilot 数据集，coverage 指标验证 |
| Phase 4 | 正式数据集生成 | 全量正式数据集 |
| Phase 5 | 数据审计与冻结 | 数据审计报告，数据集冻结 |
| Phase 6 | 训练框架与 sanity check | BC/CQL 框架，sanity check 通过 |
| Phase 7 | Env-A 主实验 | Coverage vs Size 四格完整结果 |
| Phase 8 | Env-B/C 复杂度复验 | 关键对照复验结果 |
| Phase 9 | Env-A quality 调制实验 | 5 个质量档结果 |
| Phase 10 | 机制解释 | 机制分析图与解释 |
| Phase 11 | 连续控制 benchmark 趋势复验 | benchmark 结果（趋势验证） |
| Phase 12 | 统计、图表、写作定稿 | 最终报告 |

### 硬约束规则

1. **未通过当前阶段审查，不得进入下一阶段。**
2. **数据审计（Phase 5）未通过，不得开始主实验（Phase 7）。**
3. **Env-B / Env-C 不得擅自扩展为完整平级主实验。**
4. **benchmark（Phase 11）不得提前抢占主线。**
5. **不得边做边改研究问题。**
6. **不得边做边扩展实验范围。**
7. **阶段顺序不可跳过、不可颠倒。**

### 阶段通过标准与回滚规则

**Phase 1 通过标准：**
- Env-A/B/C 实现完成
- 单元测试全部通过：deterministic 转移验证、reward 计算验证、horizon 终止验证
- 回滚条件：任何单元测试失败 → 修复后重新验证，不得跳过

**Phase 2 通过标准：**
- 8 个 DQN seeds 全部训练完成
- 质量分档结果符合预期（各档区分度明显）
- 行为策略库存档
- 回滚条件：分档失败 → 重新调整训练参数，不得使用未达标策略

**Phase 3 通过标准：**
- Pilot 数据集生成成功
- Wide / narrow coverage 指标差异可测量且显著
- Coverage 构造方法可复现
- 回滚条件：coverage 差异不显著 → 重新审查构造参数，不得跳过进入 Phase 4

**Phase 4 通过标准：**
- 全量正式数据集生成完成（四格 × quality 档 × Env-B/C 关键对照）
- 所有数据集元数据记录完整

**Phase 5 通过标准：**
- 数据集审计报告完成
- 各数据集 coverage 指标核查通过
- 数据集 md5/hash 冻结存档
- 回滚条件：审计发现 coverage 构造错误 → 重新生成对应数据集，不得跳过进入 Phase 6

**Phase 6 通过标准：**
- BC / CQL 训练框架完成
- Sanity check：在 medium 数据集上 BC 收敛，CQL 不崩溃
- 回滚条件：sanity check 失败 → 修复框架 bug，不得跳过进入 Phase 7

**Phase 7 通过标准：**
- 四格数据集 × BC + CQL 全部跑完（20 seeds）
- 主假设 H1 可以定性判断
- 回滚条件：结果异常（极端方差、NaN）→ 排查训练 bug，不得直接进入写作

**Phase 8 通过标准：**
- Env-B / Env-C 关键对照（small-wide vs large-narrow）各完成（20 seeds）
- 复验方向与 Env-A 一致性可判断

**Phase 9 通过标准：**
- 5 个 quality 档 × BC + CQL 全部跑完（20 seeds）
- H1b / H1c 可判断

**Phase 10 通过标准：**
- 机制图完成（至少包含 coverage 指标与性能的散点或相关图）
- 机制解释与主假设逻辑一致

**Phase 11 通过标准：**
- 连续控制 benchmark 正式结果完成（20 seeds）
- 趋势方向与离散主实验吻合或差异可解释

**Phase 12 通过标准：**
- 所有 6 张核心图完成
- 统计检验完成（t-test 或等价检验）
- 报告定稿

---

## 10. 明确禁止事项

- 禁止在正式实验中使用少于 20 seeds 的结果
- 禁止将 pilot 数据或 sanity 结果直接进入最终图表
- 禁止 Env-B / Env-C 承担主结论
- 禁止 benchmark 结果用于支撑主结论
- 禁止在主实验阶段引入 IQL 或其他未列算法
- 禁止擅自扩大数据集矩阵（如增加 size 档位）
- 禁止擅自增加研究问题
- 禁止在协议外自行新增环境
- 禁止在任何阶段跳过审查标准
- 禁止以"后续可以考虑"为由在本协议中写入发散内容
