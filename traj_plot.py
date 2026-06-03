# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据准备
# x = np.linspace(0, 19, 20)
# traj1 = np.array([
#     0, 1.098609161, 2.194446138, 3.287562756, 4.377927714, 5.465358216, 6.549562551, 7.630254554, 8.707257571,
#     9.7804896, 10.85118114, 11.92176923, 12.99438127, 14.07061326, 15.1516738, 16.23859046, 17.33107714, 18.4277741,
#     19.5276816, 20.63012265
# ])
# traj2 = np.array([
#     13.59817503, 14.70166398, 15.80415313, 16.90405944, 17.99945543, 19.09492681, 20.18978811, 21.27140927,
#     22.3382751, 23.39957978, 24.46064836, 25.52182442, 26.58415503, 27.65150407, 28.7266095, 29.80665146,
#     30.89265881, 31.98668479, 33.08567766, 34.18673833
# ])
#
# # 创建透明画布
# fig, ax = plt.subplots(figsize=(3, 2), facecolor='#F6F5F5')
# # fig, ax = plt.subplots(figsize=(6, 6))
# # fig.patch.set_alpha(0.0)
#
# # 颜色设置
# color1 = 'deepskyblue'
# color2 = 'deeppink'
# ax.set_facecolor('#F6F5F5')  # 设为白底
# # 前10个点（实线）
# ax.plot(x[:10], traj1[:10], color=color1, linewidth=2.5, linestyle='-')
# ax.plot(x[:10], traj2[:10], color=color2, linewidth=2.5, linestyle='-')
#
# # 后10个点（虚线，浅）
# ax.plot(x[9:], traj1[9:], color=color1, linewidth=2.5, linestyle='--', alpha=0.1)
# ax.plot(x[9:], traj2[9:], color=color2, linewidth=2.5, linestyle='--', alpha=0.1)
#
# # 标记关键点：t-P, t, t+F（实心点）
# for i in [0, 9]:
#     ax.scatter(x[i], traj1[i], color=color1, s=60, zorder=5)
#     ax.scatter(x[i], traj2[i], color=color2, s=60, zorder=5)
# for i in [19]:
#     ax.scatter(x[i], traj1[i], color=color1, s=60, zorder=5, alpha=0.2)
#     ax.scatter(x[i], traj2[i], color=color2, s=60, zorder=5, alpha=0.2)
# # 只标注 t-P, t, t+F 文字，放在x轴下方
# # for i, label in zip([0, 9, 19], ['t-P', 't', 't+F']):
# #     ax.text(x[i], min(traj1.min(), traj2.min()) - 3, label,
# #             fontsize=16, ha='center', va='top')
#
# # 去掉默认边框
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# # 去除刻度
# ax.set_xticks([])
# ax.set_yticks([])
#
# # 添加坐标轴箭头
# ax.annotate('', xy=(20.5, -1.5), xytext=(-0.6, -1.5),
#             arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))
# ax.annotate('', xy=(-0.5, max(traj2) + 2), xytext=(-0.5, min(traj1) - 1.7),
#             arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))
#
# # 添加坐标轴标签
# # ax.text(21, 0, 'Time', fontsize=16, ha='left', va='center')
# # ax.text(-1.3, max(traj2) // 2 -2, 'Position', fontsize=20, ha='center', va='bottom', rotation=90)
#
# # 设置边界范围
# ax.set_xlim(-1, 21)
# ax.set_ylim(min(traj1.min(), traj2.min()) - 2, max(traj1.max(), traj2.max()) + 2.5)
#
# plt.tight_layout()
# plt.savefig("trajectory_with_arrows.png", dpi=600, transparent=True)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 时间步
x = np.linspace(0, 99, 100)
dt = 1.0

# 定义速度段（v1, v2, v3）
v_segments = [2.0, 1.0, 3.0]
lengths = [20, 20, 20, 20, 20]  # 每段持续时间
accel1 = np.linspace(v_segments[0], v_segments[1], lengths[1])
accel2 = np.linspace(v_segments[1], v_segments[2], lengths[3])

# 构造完整速度曲线
velocity1 = np.concatenate([
    np.ones(lengths[0]) * v_segments[0],  # 匀速1
    accel1,                               # 减速
    np.ones(lengths[2]) * v_segments[1],  # 匀速2
    accel2,                               # 加速
    np.ones(lengths[4]) * v_segments[2],  # 匀速3
])

# 位置轨迹1 = 速度积分（累加）
traj1 = np.cumsum(velocity1 * dt)

# 轨迹2 = 轨迹1 + 偏置
traj2 = traj1 + 30

# 创建画布
fig, ax = plt.subplots(figsize=(4, 3), facecolor='#F6F5F5', dpi=600)
ax.patch.set_alpha(0.0)
fig.patch.set_alpha(0.0)
# 颜色设置
color1 = 'deepskyblue'
color2 = 'deeppink'

# 绘图
ax.plot(x[5:50], traj1[5:50], color=color1, linewidth=2.5)
ax.plot(x[5:50], traj2[5:50], color=color2, linewidth=2.5)

# ax.plot(x[:10], traj1[:10], color=color1, linewidth=2.5, linestyle='-', alpha=0.1)
# ax.plot(x[:10], traj2[:10], color=color2, linewidth=2.5, linestyle='-', alpha=0.1)
#
# 后10个点（虚线，浅）
ax.plot(x[49:95], traj1[49:95], color=color1, linewidth=2.5, linestyle='--', alpha=0.1)
# ax.plot(x[49:95], traj2[49:95], color=color2, linewidth=2.5)
# ax.plot(x[49:95], traj1[49:95], color=color1, linewidth=2.5, linestyle='--', alpha=0.1)
ax.plot(x[49:95], traj2[49:95], color=color2, linewidth=2.5)

# 标记关键点：t-P, t, t+F（示意为起点、中点、终点）
for i in [5, 49]:
    ax.scatter(x[i], traj1[i], color=color1, s=60, zorder=5)
    ax.scatter(x[i], traj2[i], color=color2, s=60, zorder=5)
# 标记关键点：t-P, t, t+F（示意为起点、中点、终点）
for i in [95, 49]:
    ax.scatter(x[i], traj1[i], color=color1, s=60, zorder=5, alpha=0.1)
    ax.scatter(x[i], traj2[i], color=color2, s=60, zorder=5)
# 文字标签（在x轴下方）
for i, label in zip([5, 49, 95], ['t-P', 't', 't+F']):
    ax.text(x[i], min(traj1.min(), traj2.min()) - 10, label,
            fontsize=18, ha='center', va='top')
ax.text(x[0] - 8, max(traj1.max(), traj2.max()), 'x',
            fontsize=18, ha='center', va='top')
# 去掉边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 去除刻度
ax.set_xticks([])
ax.set_yticks([])

# 添加箭头坐标轴
ax.annotate('', xy=(102, traj1[0]-5), xytext=(-5, traj1[0]-5),
            arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))
ax.annotate('', xy=(-4, max(traj2)+5), xytext=(-4, traj1[0]-10),
            arrowprops=dict(arrowstyle='->', linewidth=2, color='black'))

# # 坐标轴标签
# ax.text(103, traj1[0]-5, 'Time', fontsize=16, ha='left', va='center')
# ax.text(-6, (traj1[-1]+traj1[0])/2, 'Position', fontsize=16, ha='center', va='bottom', rotation=90)

# 边界
ax.set_xlim(-5, 105)
ax.set_ylim(traj1[0] - 10, traj2[-1] + 10)

plt.tight_layout()
plt.savefig("trajectory_piecewise_motion.png", dpi=600, transparent=True)
plt.show()
