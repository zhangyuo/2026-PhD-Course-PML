你手头的数据/模型

对象	说明
ogbn_arxiv_balanced_subgraph.pt	基础图，含 x_orig（原始特征）、edge_index、labels
水印数据集（mark_save/*.pt）	含 x_wm（扰动后特征）、wm_nodes（被选水印节点集合）
Benign 模型	训练于 x_orig，产生正常节点 embedding
Mark 模型	训练于 x_wm，产生水印注入后的 embedding
可以分析的变化维度


维度1：特征层面 (x_orig → x_wm)

Δx = x_wm - x_orig 的分布（哪些维度被扰动最大？）
水印节点 vs 非水印节点的特征分布差异
能否从 Δx 的统计特性反推水印节点？（隐蔽性分析）


维度2：嵌入层面 (H_benign → H_mark)

同一节点在两个模型下的 embedding 偏移方向是否一致（即是否对齐到 carrier direction u_c）
非水印节点的 embedding 是否受影响（污染范围）
水印节点的 embedding 变化是否更大


维度3：预测层面（概率/不确定性）

两个模型对水印节点的预测置信度变化
水印注入后，分类决策是否稳定？
哪些节点预测翻转了？