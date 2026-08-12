# 连享会 (lianxh.cn) — Curated Stata Causal-Inference Link Index

A curated **index of external links** to Chinese-language Stata tutorials on 连享会 (lianxh.cn), scoped to the causal-inference methods this skill covers. Each entry maps to a section of `method-patterns.md`.

> **Copyright / usage**: All linked content is © 连享会 (lianxh.cn), **保留所有权利 / All Rights Reserved**. This file reproduces **only post titles, URLs, and short factual descriptions written here** — no article text is copied. Use as a personal study reference and cite the original posts. Do not paste article bodies into this repo.

**In-Stata discovery** (the site's own search): from Stata's command window, `lianxh DID csdid` or `lianxh RDD` searches these posts by keyword; `songbl` searches a companion set; `lianxh, update` refreshes the local index. Category pages (`/blogs/NN.html`) are lianxh's own maintained indexes and are the most durable links.

---

## §2–§4 Difference-in-Differences / Event Study / Staggered DiD

**Category index**: [倍分法 DID](https://www.lianxh.cn/blogs/39.html)

| Post | URL | Maps to |
|------|-----|---------|
| 倍分法DID详解(二)：多时点 DID (渐进DID) | https://www.lianxh.cn/details/72.html | §4 staggered |
| DID新进展：异质性多期DID估计的新方法-csdid (Callaway–Sant'Anna) | https://www.lianxh.cn/details/1071.html | §4 `csdid` |
| DID偏误问题：多时期DID的双重稳健估计量(下)-csdid | https://www.lianxh.cn/news/762e878e7063b.html | §4 `csdid` |
| DID最新进展：异质性处理下的双向固定效应DID (TWFEDD) | https://www.lianxh.cn/news/bafafbd2aa5ca.html | §4 TWFE bias |
| Stata倍分法新趋势：did2s-两阶段双重差分 | https://www.lianxh.cn/news/cb98eb5208c55.html | §4 `did2s` |
| tfdiff：多期DID的估计及图示 | https://www.lianxh.cn/news/d8241c4e502ed.html | §4 `tfdiff` |
| Stata：各种DID估计量的比较分析 | https://www.lianxh.cn/news/f4c29c86690f8.html | §4 estimator comparison |
| Stata：双重差分的固定效应模型 (DID) | https://www.lianxh.cn/news/f7499048842cc.html | §2 baseline |
| DID：仅有几个实验组样本的倍分法 | https://www.lianxh.cn/news/c1c8700574729.html | §2 few-treated |

### Event study (dynamic DiD) → §3

| Post | URL |
|------|-----|
| Stata：一文读懂事件研究法 Event Study | https://www.lianxh.cn/news/3820f71099fd9.html |
| Stata：面板事件研究法-eventdd | https://www.lianxh.cn/details/826.html |
| Stata：图示事件研究分析结果-eventcoefplot | https://www.lianxh.cn/news/819d4064f139c.html |
| Stata：短期事件研究法 (Event_Study) 教程 | https://www.lianxh.cn/news/90de95e42e8ff.html |

---

## §5 Regression Discontinuity (RDD)

**Category index**: [断点回归 RDD](https://www.lianxh.cn/blogs/40.html)

| Post | URL |
|------|-----|
| RDD-断点回归：实践指南 | https://www.lianxh.cn/details/1193.html |
| Stata+R：一文读懂精确断点回归-RDD | https://www.lianxh.cn/details/590.html |
| Stata: 断点回归(RDD)中的平滑性检验 (McCrary-style) | https://www.lianxh.cn/details/118.html |
| RDD：断点回归可以加入控制变量吗？ | https://www.lianxh.cn/details/517.html |
| 断点回归RDD：样本少时如何做？ | https://www.lianxh.cn/details/427.html |
| RDD最新进展：多断点RDD、多分配变量RDD | https://www.lianxh.cn/details/40.html |
| Stata：RDD-DID-断点回归与倍分法完美结合 (diff-in-disc) | https://www.lianxh.cn/details/763.html |

---

## §6 Instrumental Variables / GMM

**Category index**: [工具变量 IV-GMM](https://www.lianxh.cn/blogs/38.html)

| Post | URL |
|------|-----|
| IV的标准动作：工具变量法实用指南 | https://www.lianxh.cn/details/1351.html |
| IV在哪里？奇思妙想的工具变量 (instrument construction) | https://www.lianxh.cn/details/619.html |
| 多个(弱)工具变量如何应对-IV-mivreg | https://www.lianxh.cn/news/807b616b11aae.html |

---

## §7 Synthetic Control

**Category index**: [合成控制法](https://www.lianxh.cn/blogs/42.html)

| Post | URL |
|------|-----|
| 合成控制法 (Synthetic Control Method) 及 Stata 实现 | https://www.lianxh.cn/details/119.html |
| Synth_Runner 命令：合成控制法高效实现 | https://www.lianxh.cn/details/205.html |
| Stata：合成控制法介绍-synth2 | https://www.lianxh.cn/details/1133.html |
| Stata：合成控制法-synth-命令无法加载 plugin 的解决办法 | https://www.lianxh.cn/details/204.html |

---

## §8 Matching / PSM

**Category index**: [PSM-Matching](https://www.lianxh.cn/blogs/41.html)

| Post | URL |
|------|-----|
| Stata：psestimate-倾向得分匹配(PSM)中协变量的筛选 | https://www.lianxh.cn/details/370.html |
| Stata：psestimate-倾向得分匹配(PSM)中匹配变量的筛选 | https://www.lianxh.cn/details/230.html |

---

## §15 Double/Debiased Machine Learning & Causal ML

**Category indexes**: [机器学习](https://www.lianxh.cn/blogs/47.html) · [内生性-因果推断](https://www.lianxh.cn/blogs/19.html)

| Post | URL |
|------|-----|
| 因果推断：双重机器学习-ddml | https://www.lianxh.cn/news/5529578569a81.html |
| Stata：双重机器学习-多维聚类标准误的估计方法-crhdreg | https://www.lianxh.cn/news/7519c2f054479.html |
| Cai博士笔记(上)：因果推断核心方法和文献速览 | https://www.lianxh.cn/details/1741.html |
| ML·机器学习与因果推断 (course hub) | https://kc.lianxh.cn/list/body/Machine-Learning.html |

---

## Mechanism tests — Mediation / Moderation (supports the channel-decomposition patterns)

**Category index**: [交乘项-调节-中介](https://www.lianxh.cn/blogs/21.html)

| Post | URL |
|------|-----|
| 路径分析实操指南：调节效应、中介效应和调节中介 | https://www.lianxh.cn/details/1475.html |
| Stata：调节中介效应检验 | https://www.lianxh.cn/details/268.html |
| medsem-中介效应：基于结构方程模型 SEM 的中介效应分析 | https://www.lianxh.cn/details/581.html |

> **Caution**: the mediation/moderation posts use Baron–Kenny / SEM framing. For causal mechanism claims, prefer the joint-pattern / channel-decomposition logic in `identification-writing-patterns.md` over stepwise mediation — a mediator regression is correlational, not a causal channel proof.

---

## Panel / Fixed Effects (supports §1, §13)

**Category index**: [面板数据](https://www.lianxh.cn/blogs/20.html)

| Post | URL |
|------|-----|
| Stata实操陷阱：动态面板数据模型 (dynamic panel / GMM) | https://www.lianxh.cn/news/cc6c5ea80d70c.html |

---

*Index compiled 2026-07-13 from lianxh.cn category pages and site search. Links may move; the `/blogs/NN.html` category pages are the stable entry points. Report broken links by re-running `lianxh <keyword>` in Stata.*
