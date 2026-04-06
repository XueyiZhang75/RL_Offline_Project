# PROJECT_SCOPE.md — 项目边界文件

> 本文件定义本项目的做与不做、必做项与增强项、主线边界与复验边界。
> 执行过程中如有边界争议，以本文件为准。

---

## 1. 本项目做什么

### 1.1 必做项

| 项目 | 说明 |
|---|---|
| 搭建 Env-A（15×15 FourRooms） | 主环境，承担全部主实验 |
| 搭建 Env-B（Bottleneck 风格） | 复验环境，仅做关键对照 |
| 搭建 Env-C（Key-Door 风格） | 复验环境，仅做关键对照 |
| 训练行为策略池（8 DQN seeds） | 支持 coverage 构造 |
| 生成四格主实验数据集（Env-A） | small-wide / small-narrow / large-wide / large-narrow |
| 生成 quality 调制数据集（Env-A） | random / suboptimal / medium / expert / mixed |
| 生成 Env-B/C 关键复验数据集 | small-wide vs large-narrow 两格 |
| 数据审计与冻结 | 审计通过后冻结，不再修改 |
| 实现 BC 算法（离散） | 必做 |
| 实现 CQL 算法（离散） | 必做 |
| Env-A 主实验（Coverage vs Size） | 四格 × BC + CQL × 20 seeds |
| Env-A quality 调制实验 | 5 档 × BC + CQL × 20 seeds |
| Env-B/C 关键复验 | small-wide vs large-narrow × 20 seeds |
| 机制解释分析 | coverage 指标与性能关系 |
| 连续控制 benchmark 趋势复验（必做最小集） | Hopper-medium / Hopper-medium-replay / Hopper-medium-expert × BC / CQL / IQL / TD3+BC × 20 seeds |
| 统计检验与最终图表 | 6 张核心图 + 统计显著性 |

### 1.2 增强项（非必做，有条件才做）

| 项目 | 条件 |
|---|---|
| IQL（离散环境） | 仅在 BC + CQL 主实验全部稳定完成后，方可考虑加入 |
| Env-B/C 完整四格扩展 | 仅在有明确必要性时，方可考虑，需重新审批 |
| 连续控制 benchmark 增强集 | Walker2d-medium / Walker2d-medium-replay / Walker2d-medium-expert，仅在 Hopper 三档全部稳定完成且资源允许时方可加入 |

---

## 2. 本项目不做什么

### 2.1 算法层面

- 不做 AWAC、IBC、Decision Transformer、MOPO 等其他 offline RL 算法
- 不做在线 RL 算法（PPO、SAC 等）
- 不做模仿学习变体（GAIL、AIRL 等）
- 不做 reward shaping 研究
- 不做超参数搜索作为主实验

### 2.2 环境层面

- 不做连续状态空间的自建环境
- 不做随机转移环境（stochastic dynamics）
- 不做多智能体环境
- 不做图像输入环境
- 不做超出 Env-A/B/C 以外的自建离散环境

### 2.3 数据层面

- 不做 size 档位超出 small（50k）/ large（200k）的数据集
- 不做 coverage 超出 wide / narrow 两极的细粒度中间档
- 不做人类示范数据
- 不做真实世界数据集

### 2.4 结论层面

- 不以 Env-B / Env-C 的结果承担主结论
- 不以连续控制 benchmark 的结果承担主结论
- 不声称结论可泛化至在线 RL

---

## 3. 环境层级说明

### 为什么 Env-A 承担主结论

1. Env-A（15×15 FourRooms）具有明确的几何结构，state-action coverage 可精确定义和测量。
2. Env-A 完全确定性，排除随机性对 coverage 测量的干扰。
3. Env-A 足够复杂（FourRooms 结构使 bottleneck 天然存在）又足够可控（可枚举 state space）。
4. Env-A 承担完整四格矩阵实验，具备回答主问题所需的全部对照。

### 为什么 Env-B / Env-C 只做复验

1. Env-B / Env-C 的引入目的是验证 Env-A 结论的跨环境一致性，而非独立建立新结论。
2. 若 Env-B / Env-C 独立承担完整实验，将使实验规模失控且分散论证焦点。
3. 仅做关键对照（small-wide vs large-narrow）已足够判断趋势一致性。

### 为什么连续控制只做趋势验证

1. 连续控制 benchmark（如 D4RL）的 coverage 定义与离散环境不完全等价，不能直接做主对照。
2. 连续控制结果受超参数、网络结构影响更大，控制变量困难。
3. 连续控制 benchmark 的作用是"外部趋势是否与离散主实验方向吻合"，不是建立新结论。

### 连续控制 benchmark 任务范围冻结

| 层级 | 任务集 | 说明 |
|---|---|---|
| 必做最小集 | Hopper-medium、Hopper-medium-replay、Hopper-medium-expert | 默认执行范围，不允许缩减 |
| 增强项 | Walker2d-medium、Walker2d-medium-replay、Walker2d-medium-expert | 仅在 Hopper 三档全部稳定完成且资源允许时加入 |
| 延后项（默认不做） | HalfCheetah-* 系列 | 不列入默认计划，不允许以"顺手跑一下"方式加入 |

**锁定理由：** Hopper 三档已覆盖 medium / replay / expert 三种质量梯度，足以判断趋势一致性；扩展至 Walker2d 提供跨任务复验；HalfCheetah 行为特性与本项目 coverage 假设相关性弱，不默认纳入。

---

## 4. 证据层级

| 层级 | 内容 | 作用 |
|---|---|---|
| 主证据 | Env-A 完整主实验 | 直接回答主问题，承担主结论 |
| 离散复验证据 | Env-B、Env-C 关键对照 | 验证结论跨结构一致性 |
| 外部趋势验证 | 连续控制 benchmark | 验证结论在连续控制领域是否有趋势一致性 |

**原则：只有主证据可以承担主结论；复验证据与外部趋势验证只能说明"趋势一致"或"趋势不一致"。**

---

## 5. 防止范围失控的执行规则

1. **任何新增实验条件**（如新 size 档位、新 quality 档位、新算法）必须在协议中注明，不得自行添加。
2. **任何新增环境**必须经过重新审批，不得在 Phase 1 之后擅自引入。
3. **Env-B / Env-C 不得扩展为完整四格**，除非有明确书面授权。
4. **连续控制 benchmark 不得提前于主实验启动**。
5. **所有阶段必须严格按 EXP_PROTOCOL.md 中的阶段顺序执行**，不得跳步。
6. **每个阶段的输出必须明确对应其通过标准**，不得以"大致完成"替代通过标准。
