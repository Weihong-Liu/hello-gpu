"""Ch6 Roofline：把 Ch5 的 coalesced / linecross 两个实测点画到 9070XT Roofline 上。

对比 Ch3 的版本，本章多画一个 linecross（跨 cache line）点，直观展示
"访存合并被破坏后，工作点从带宽斜线跌下来多远"——这是 Part 1 profiling
闭环的最终交付图。

实测数据来源（均在 9070XT + ROCm 7.13 + 原生 Ubuntu 24.04 测得）：
  - 标称峰值带宽：~760 GB/s
  - coalesced vector add 有效带宽：603 GB/s（Ch5 §5.2，192MiB footprint 部分 L2 命中）
  - linecross stride=32 有效带宽：90 GB/s（Ch5 §5.2，跨 cache line 合并破坏）
  - fp32 算力：~14.6 TFLOPS（Ch2 实测 torch.matmul）
  - fp16 算力：~124 TFLOPS（Ch2 实测 torch.matmul, WMMA）

两个工作点算术强度相同（都处理 n 个元素、做 1 次加法、搬 3n×4 字节，AI=1/12≈0.083），
唯一差别是 linecross 的有效带宽因合并破坏跌到 90 GB/s，所以它的实测性能点
(P = AI × B_ach) 比 coalesced 低 6.7×。

标签全用英文，避免中文字体缺失乱码；中文解读放正文图注。

用法：
    python plot_roofline_ch6.py            # 弹窗显示
    python plot_roofline_ch6.py --save      # 存 PNG 到同目录
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# ---- 实测硬件参数 ----
BW_NOMINAL = 760          # GB/s, 标称峰值
P_FP32 = 14.6             # TFLOPS, torch.matmul fp32
P_FP16 = 124.0            # TFLOPS, torch.matmul fp16 (WMMA)

# ---- 两个工作点实测（Ch5 §5.2）----
AI_VADD = 1.0 / 12        # ≈ 0.083 FLOP/Byte，两版相同
BW_COALESCED = 603        # GB/s, coalesced 有效带宽
BW_LINECROSS = 90         # GB/s, linecross stride=32 有效带宽
P_COALESCED = AI_VADD * BW_COALESCED / 1e3   # TFLOPS ≈ 0.050
P_LINECROSS = AI_VADD * BW_LINECROSS / 1e3   # TFLOPS ≈ 0.0075

# 拐点
KNEE_FP32 = P_FP32 / (BW_COALESCED / 1e3)
KNEE_FP16 = P_FP16 / (BW_COALESCED / 1e3)


def plot(save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)

    ai = np.logspace(-2, 3.2, 500)

    # 带宽斜线（标称 / coalesced 实测）。大 footprint copy 线待 Ch4 原生复跑后再加回。
    ax.plot(ai, (BW_NOMINAL / 1e3) * ai, ":", color="#2563eb", lw=1.2, alpha=0.6,
            label=f"Slope: BW = {BW_NOMINAL} GB/s (nominal)")
    ax.plot(ai, (BW_COALESCED / 1e3) * ai, "-", color="#2563eb", lw=1.8,
            label=f"Slope: BW = {BW_COALESCED} GB/s (coalesced measured)")

    # 算力水平线
    ax.axhline(P_FP32, color="#dc2626", lw=1.8, ls="--",
               label=f"fp32 ceiling = {P_FP32} TFLOPS")
    ax.axhline(P_FP16, color="#059669", lw=1.8, ls="--",
               label=f"fp16 ceiling = {P_FP16} TFLOPS (WMMA)")

    # coalesced 实测点（贴着带宽斜线）
    ax.scatter([AI_VADD], [P_COALESCED], color="#ea580c", zorder=6, s=95,
               marker="*", edgecolors="black", linewidths=0.6)
    ax.annotate("coalesced (measured)\n"
                f"AI={AI_VADD:.3f}, P={P_COALESCED:.3f} TFLOPS\n"
                f"BW={BW_COALESCED} GB/s",
                (AI_VADD, P_COALESCED), textcoords="offset points",
                xytext=(20, 22), fontsize=8.5, color="#ea580c",
                arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.2))

    # linecross 实测点（从斜线跌下来）
    ax.scatter([AI_VADD], [P_LINECROSS], color="#7c3aed", zorder=6, s=95,
               marker="X", edgecolors="black", linewidths=0.6)
    ax.annotate("linecross s=32 (measured)\n"
                f"AI={AI_VADD:.3f}, P={P_LINECROSS:.4f} TFLOPS\n"
                f"BW={BW_LINECROSS} GB/s  (6.7x slower)",
                (AI_VADD, P_LINECROSS), textcoords="offset points",
                xytext=(20, -32), fontsize=8.5, color="#7c3aed",
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.2))

    # 两点之间的退化箭头
    ax.annotate("", (AI_VADD, P_LINECROSS), (AI_VADD, P_COALESCED),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(AI_VADD * 1.15, (P_COALESCED + P_LINECROSS) / 2,
            "coalesced -> linecross\n(bandwidth collapse)", fontsize=8,
            color="#64748b", va="center")

    # 区域标注
    ax.text(0.02, 0.5, "memory-bound\n(slope side)", transform=ax.transAxes,
            fontsize=10, color="#2563eb", alpha=0.55, va="center")
    ax.text(0.80, 0.5, "compute-bound\n(ceiling side)", transform=ax.transAxes,
            fontsize=10, color="#dc2626", alpha=0.55, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title("Roofline: 9070XT — coalesced vs linecross (Ch5/Ch6)",
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="center left", fontsize=8, framealpha=0.92)
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-3, 400)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved: {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", default=os.path.join(here, "roofline-ch6.png"))
    args = p.parse_args()
    plot(args.out if args.save else None)


if __name__ == "__main__":
    main()
