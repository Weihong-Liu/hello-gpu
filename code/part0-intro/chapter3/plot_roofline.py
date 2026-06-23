"""Ch3 §3.4 Roofline 曲线：把 vector add 实测点画到 9070XT 的 Roofline 上。

硬件参数来自第 2 章 §2.8 实测：
  - GDDR6 平台带宽（copy ≥1GiB）：~500 GB/s
  - vector add 实测带宽（16M 元素 ≈ 64MiB，部分 L2 命中）：~601 GB/s
  - fp32 算力（torch.matmul，SIMD FMA）：~14.6 TFLOPS
  - fp16 算力（torch.matmul，WMMA）：~124 TFLOPS
vector add 实测点：AI ≈ 0.083 FLOP/Byte，P ≈ 0.05 TFLOPS（memory-bound）。

标签全用英文，避免中文字体缺失导致的乱码；中文解读放在正文图注里。

用法：
    python plot_roofline.py            # 弹窗显示
    python plot_roofline.py --save      # 存 PNG 到同目录
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# ---- 实测硬件参数（来自第 2 章 §2.8）----
BW_GDDR6 = 500      # GB/s, copy ≥1GiB 平台
BW_VADD = 601       # GB/s, vector add 实测（64MiB，部分 L2 命中）
P_FP32 = 14.6       # TFLOPS, torch.matmul fp32 (SIMD FMA)
P_FP16 = 124.0      # TFLOPS, torch.matmul fp16 (WMMA)

# ---- vector add 实测点 ----
AI_VADD = 1.0 / 12             # ≈ 0.083 FLOP/Byte
P_VADD = AI_VADD * BW_VADD / 1e3  # TFLOPS ≈ 0.050

# 拐点（算术强度 = P_peak / B_peak）
KNEE_FP32 = P_FP32 / (BW_VADD / 1e3)   # ≈ 24
KNEE_FP16 = P_FP16 / (BW_VADD / 1e3)   # ≈ 206


def plot(save_path=None):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)

    ai = np.logspace(-2, 3.2, 500)

    # 带宽斜线（两条：GDDR6 平台 / vector add 实测）
    ax.plot(ai, (BW_VADD / 1e3) * ai, "-", color="#2563eb", lw=1.8,
            label=f"Slope: BW = {BW_VADD} GB/s (vector add measured)")
    ax.plot(ai, (BW_GDDR6 / 1e3) * ai, ":", color="#2563eb", lw=1.3, alpha=0.7,
            label=f"Slope: BW = {BW_GDDR6} GB/s (GDDR6 plateau, copy)")

    # 算力水平线（fp32 / fp16）
    ax.axhline(P_FP32, color="#dc2626", lw=1.8, ls="--",
               label=f"fp32 ceiling = {P_FP32} TFLOPS (SIMD FMA)")
    ax.axhline(P_FP16, color="#059669", lw=1.8, ls="--",
               label=f"fp16 ceiling = {P_FP16} TFLOPS (WMMA)")

    # 拐点标记
    ax.scatter([KNEE_FP32], [P_FP32], color="#dc2626", zorder=5, s=45)
    ax.annotate(f"knee(fp32)\nAI\u2248{KNEE_FP32:.0f}", (KNEE_FP32, P_FP32),
                textcoords="offset points", xytext=(8, -22), fontsize=8, color="#dc2626")
    ax.scatter([KNEE_FP16], [P_FP16], color="#059669", zorder=5, s=45)
    ax.annotate(f"knee(fp16)\nAI\u2248{KNEE_FP16:.0f}", (KNEE_FP16, P_FP16),
                textcoords="offset points", xytext=(8, -10), fontsize=8, color="#059669")

    # vector add 实测点
    ax.scatter([AI_VADD], [P_VADD], color="#ea580c", zorder=6, s=90,
               marker="*", edgecolors="black", linewidths=0.6)
    ax.annotate("vector add (measured)\n"
                f"AI\u2248{AI_VADD:.3f}, P\u2248{P_VADD:.3f} TFLOPS",
                (AI_VADD, P_VADD), textcoords="offset points",
                xytext=(18, 18), fontsize=9, color="#ea580c",
                arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.2))

    # 区域标注
    ax.text(0.02, 0.5, "memory-bound\n(slope side)", transform=ax.transAxes,
            fontsize=10, color="#2563eb", alpha=0.55, va="center")
    ax.text(0.80, 0.5, "compute-bound\n(ceiling side)", transform=ax.transAxes,
            fontsize=10, color="#dc2626", alpha=0.55, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title("Roofline: AMD Radeon RX 9070 XT (gfx1201) + vector add",
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="center left", fontsize=8.5, framealpha=0.92)
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(1e-2, 400)

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
    p.add_argument("--out", default=os.path.join(here, "roofline-vector-add.png"))
    args = p.parse_args()
    plot(args.out if args.save else None)


if __name__ == "__main__":
    main()
