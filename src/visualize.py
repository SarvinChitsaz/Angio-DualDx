"""
Visualization utilities: fixed per-class coloring (so a segment always
renders the same color across every figure), overlap-free legend
placement, and three ready-to-use comparison figures:
  - vessel-only  (Ground Truth vs Prediction)
  - stenosis-only (Ground Truth vs Prediction)
  - combined      (both tasks + legend + optional severity index panel)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def build_fixed_color_map(num_classes=26, seed=1):
    """One stable BGR color per class id, independent of which classes
    happen to appear in a given image."""
    rng = np.random.default_rng(seed)
    return {c: tuple(int(v) for v in rng.integers(60, 230, size=3)) for c in range(1, num_classes)}


def visualize_multitask(gray_img, vessel_mask, stenosis_mask, fixed_colors, alpha=0.5):
    """Color-overlay vessel classes + draw a contour around stenosis regions.
    No text is drawn on the image itself; pair with a separate legend panel."""
    out = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    classes = [int(c) for c in np.unique(vessel_mask) if c != 0]
    overlay = out.copy()
    for c in classes:
        overlay[vessel_mask == c] = fixed_colors.get(c, (128, 128, 128))
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    stenosis_u8 = (stenosis_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(stenosis_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

    colors_rgb = {c: (fixed_colors[c][2] / 255, fixed_colors[c][1] / 255, fixed_colors[c][0] / 255) for c in classes}
    return out, colors_rgb


def _draw_legend(ax4, present_classes, fixed_colors, id_to_description, y_start=1.0, include_stenosis=True):
    y = y_start
    if len(present_classes) > 0:
        line_height = min(0.055, max(y - 0.02, 0.05) / max(len(present_classes), 1))
    else:
        line_height = 0.055

    for c in present_classes:
        col_bgr = fixed_colors[c]
        col_rgb = (col_bgr[2] / 255, col_bgr[1] / 255, col_bgr[0] / 255)
        name = id_to_description.get(c, str(c))
        box_h = line_height * 0.6
        ax4.add_patch(plt.Rectangle((0.0, y - box_h / 2), 0.06, box_h, color=col_rgb,
                                     transform=ax4.transAxes, clip_on=True))
        ax4.text(0.09, y, name, va="center", ha="left",
                  fontsize=min(14, max(8, line_height * 300)), transform=ax4.transAxes, clip_on=True)
        y -= line_height
    return y


def show_vessel_only(img_gray, gt_vessel_mask, vessel_pred, fixed_colors, id_to_description, save_path=None):
    empty = np.zeros_like(vessel_pred)
    gt_overlay, _ = visualize_multitask(img_gray, gt_vessel_mask, empty, fixed_colors)
    pred_overlay, _ = visualize_multitask(img_gray, vessel_pred, empty, fixed_colors)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Vessel only", fontsize=18, fontweight="bold")
    axes[0].imshow(img_gray, cmap="gray"); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)); axes[1].set_title("Ground Truth — Vessels"); axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)); axes[2].set_title("Prediction — Vessels"); axes[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def show_stenosis_only(img_gray, gt_stenosis_mask, stenosis_pred, save_path=None):
    def draw_contours(img, mask, color):
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        return out

    gt_vis = draw_contours(img_gray, gt_stenosis_mask, (0, 0, 255))
    pred_vis = draw_contours(img_gray, stenosis_pred, (0, 0, 255))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Stenosis only", fontsize=18, fontweight="bold")
    axes[0].imshow(img_gray, cmap="gray"); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)); axes[1].set_title("Ground Truth — Stenosis"); axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(pred_vis, cv2.COLOR_BGR2RGB)); axes[2].set_title("Prediction — Stenosis"); axes[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def show_combined(fname, img_gray, gt_vessel_mask, gt_stenosis_mask, vessel_pred, stenosis_pred,
                   fixed_colors, id_to_description, severity_score=None, lesion_breakdown=None,
                   min_pixels_for_legend=400, save_path=None):
    gt_overlay, _ = visualize_multitask(img_gray, gt_vessel_mask, gt_stenosis_mask, fixed_colors)
    pred_overlay, _ = visualize_multitask(img_gray, vessel_pred, stenosis_pred, fixed_colors)

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.85])

    ax1 = fig.add_subplot(gs[0]); ax1.imshow(img_gray, cmap="gray"); ax1.set_title("Image", fontsize=20); ax1.axis("off")
    ax2 = fig.add_subplot(gs[1]); ax2.imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)); ax2.set_title("Ground Truth", fontsize=20); ax2.axis("off")
    ax3 = fig.add_subplot(gs[2]); ax3.imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)); ax3.set_title("Prediction", fontsize=20); ax3.axis("off")

    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.axis("off")
    y = 1.0

    ax4.text(0.0, y, f"Image: {fname}", va="top", ha="left", fontsize=16, fontweight="bold", transform=ax4.transAxes)
    y -= 0.06

    if severity_score is not None:
        ax4.text(0.0, y, f"Severity Index: {severity_score:.1f}", va="top", ha="left",
                  fontsize=16, color="darkred", fontweight="bold", transform=ax4.transAxes)
        y -= 0.065

        ax4.text(0.0, y, "Detected lesions:", va="top", ha="left", fontsize=14, fontweight="bold", transform=ax4.transAxes)
        y -= 0.045
        if lesion_breakdown:
            for b in lesion_breakdown:
                ax4.text(0.02, y, f"• {b['segment']}  (w={b['weight']})", va="top", ha="left", fontsize=13, transform=ax4.transAxes)
                y -= 0.04
        else:
            ax4.text(0.02, y, "none", va="top", ha="left", fontsize=13, style="italic", transform=ax4.transAxes)
            y -= 0.04
        y -= 0.02
        ax4.axhline(y, color="gray", linewidth=1.2)
        y -= 0.04

    ax4.text(0.0, y, "Vessel segments:", va="top", ha="left", fontsize=14, fontweight="bold", transform=ax4.transAxes)
    y -= 0.045

    present_classes = sorted(set(np.unique(gt_vessel_mask)) | set(np.unique(vessel_pred)))
    present_classes = [
        c for c in present_classes
        if c != 0 and ((gt_vessel_mask == c).sum() >= min_pixels_for_legend
                        or (vessel_pred == c).sum() >= min_pixels_for_legend)
    ]
    _draw_legend(ax4, present_classes, fixed_colors, id_to_description, y_start=y)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
