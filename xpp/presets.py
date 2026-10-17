"""Named configurations: one per row of the paper's tables.

`python scripts/train.py --preset xpp` is the same as passing the flags below by hand. Flags given
explicitly on the command line override the preset. The common training budget (only layer4,
lr 1e-5, 1500 iterations, validation every 100) is the default of scripts/train.py.

Old run-directory names used during the experiments are listed in OLD_NAMES (see
scripts/rename_old_runs.py).
"""

PRESETS = {
    # name                 flags                                                     paper name
    "imagenet":         dict(mode="none"),                                         # ImageNet features
    "instance":         dict(mode="views", sources="aug"),                         # Instance training
    "colour_jitter":    dict(mode="views", sources="aug", color_jitter=True),      # Colour-jitter control
    "region":           dict(mode="views", sources="aug,region"),                  # + other regions
    "angle":            dict(mode="views", sources="aug,angle"),                   # + other stage angles
    "region_angle":     dict(mode="views", sources="aug,region,angle"),            # + regions + angles
    "xpp":              dict(mode="views", sources="aug,ppl"),                     # XPP (ours)
    "xpp_region_angle": dict(mode="views", sources="aug,region,angle,ppl"),        # + PPL + regions + angles
    "neighbour":        dict(mode="nn"),                                           # Neighbour pseudo-tasks
    "supervised":       dict(mode="supervised"),                                   # Supervised (all 78 types)
}

# run-directory names used while the experiments were run  ->  preset names
OLD_NAMES = {
    "imnet": "imagenet", "aug": "instance", "aug_color": "colour_jitter", "region_only": "region",
    "angle_only": "angle", "region_angle": "region_angle", "ppl_only": "xpp", "full": "xpp_region_angle",
    "nn": "neighbour", "sup": "supervised", "sup10": "supervised_k10", "sup20": "supervised_k20",
    "sup40": "supervised_k40",
}
