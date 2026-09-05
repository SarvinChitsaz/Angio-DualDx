"""
Simplified, SYNTAX-inspired severity index.

For each detected stenosis lesion, the lesion is assigned to the vessel
segment it overlaps (or, if it lies outside every annotated vessel --
which happens for ~48% of ARCADE stenosis instances, see data/README.md --
to the nearest segment by pixel distance). Each assignment contributes
`segment_weight * OCCLUSION_MULTIPLIER` to a running total.

This is explicitly NOT a clinical-grade SYNTAX score: percent diameter
narrowing (needed to distinguish occlusive vs. non-occlusive lesions),
lesion length, calcification, tortuosity, and bifurcation involvement are
not available in the ARCADE annotations, so a fixed occlusion multiplier
is used instead. Treat the output as a relative severity ranking between
images in this dataset, not as a value comparable to a real SYNTAX score.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as ndi_label

from configs.config import OCCLUSION_MULTIPLIER


def build_class_weight_table(id_to_name, syntax_segment_weight):
    return {cid: syntax_segment_weight.get(name, 1.0) for cid, name in id_to_name.items()}


def assign_lesions_to_segments(vessel_pred, stenosis_pred_mask, class_id_to_weight):
    lesion_labels, num_lesions = ndi_label(stenosis_pred_mask)

    results = []
    for lesion_id in range(1, num_lesions + 1):
        lesion_pixels = lesion_labels == lesion_id

        overlap_classes = vessel_pred[lesion_pixels]
        overlap_classes = overlap_classes[overlap_classes != 0]

        if len(overlap_classes) > 0:
            assigned_class = int(np.bincount(overlap_classes).argmax())
            method = "overlap"
        else:
            dist_maps = {}
            for c in class_id_to_weight:
                class_pixels = vessel_pred == c
                if class_pixels.sum() == 0:
                    continue
                dist_map = distance_transform_edt(~class_pixels)
                ys, xs = np.where(lesion_pixels)
                dist_maps[c] = dist_map[ys, xs].mean()

            if dist_maps:
                assigned_class = min(dist_maps, key=dist_maps.get)
                method = "nearest"
            else:
                assigned_class, method = None, "none"

        results.append({
            "lesion_id": lesion_id,
            "pixel_count": int(lesion_pixels.sum()),
            "assigned_class": assigned_class,
            "method": method,
        })

    return results


def compute_severity_index(vessel_pred, stenosis_pred_mask, class_id_to_weight, id_to_description):
    lesions = assign_lesions_to_segments(vessel_pred, stenosis_pred_mask, class_id_to_weight)

    total_score = 0.0
    breakdown = []
    for l in lesions:
        c = l["assigned_class"]
        if c is None:
            continue
        w = class_id_to_weight.get(c, 1.0)
        contribution = w * OCCLUSION_MULTIPLIER
        total_score += contribution
        breakdown.append({
            "segment": id_to_description.get(c, str(c)),
            "weight": w,
            "contribution": contribution,
            "assignment_method": l["method"],
        })

    return total_score, breakdown
