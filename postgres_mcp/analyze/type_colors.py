import os
from typing import List, Dict, Tuple
from matplotlib.colors import cnames

# You can modify it yourself
COLOR_BORDERS = [
    (cnames["lightgrey"], cnames["dimgrey"]),
    (cnames["lightcoral"], cnames["darkred"]),
    (cnames["cornflowerblue"], cnames["darkblue"]),
    (cnames["palegreen"], cnames["darkgreen"]),
]


"""
# Two examples for TYPE_INFO

TYPE_INFO = {
    "Empty": {
        "edgecolor": (cnames["lightgrey"], cnames["dimgrey"]),
        "subtype": ["sub1", "sub2", "sub3", "sub4"],
    },
    "Amorphous": {
        "edgecolor": (cnames["lightcoral"], cnames["darkred"]),
        "subtype": ["sub1", "sub2", "sub3", "sub4"],
    },
    "Crystalline": {
        "edgecolor": (cnames["peachpuff"], cnames["saddlebrown"]),
        "subtype": ["sub1", "sub2", "sub3", "sub4"],
    },
    "Mixing": {
        "edgecolor": (cnames["palegreen"], cnames["darkgreen"]),
        "subtype": ["sub1", "sub2", "sub3", "sub4"],
    },
}

TYPE_INFO = {
    "T0": {"edgecolor": ("#d3d3d3", "#d3d3d3"), "subtype": [""]},
    "T1": {"edgecolor": ("#696969", "#696969"), "subtype": [""]},
    "T2": {"edgecolor": ("#f08080", "#f08080"), "subtype": [""]},
    "T3": {"edgecolor": ("#bd4040", "#bd4040"), "subtype": [""]},
    "T4": {"edgecolor": ("#8b0000", "#8b0000"), "subtype": [""]},
    "T5": {"edgecolor": ("#ffdab9", "#ffdab9"), "subtype": [""]},
    "T6": {"edgecolor": ("#8b4513", "#8b4513"), "subtype": [""]},
    "T7": {"edgecolor": ("#98fb98", "#98fb98"), "subtype": [""]},
    "T8": {"edgecolor": ("#65c865", "#65c865"), "subtype": [""]},
    "T9": {"edgecolor": ("#329632", "#329632"), "subtype": [""]},
    "T10": {"edgecolor": ("#006400", "#006400"), "subtype": [""]},
    "T11": {"edgecolor": ("#4169E1", "#4169E1"), "subtype": [""]},
    "T12": {"edgecolor": ("#ee82ee", "#ee82ee"), "subtype": [""]},
    "T13": {"edgecolor": ("#9c41B8", "#9c41B8"), "subtype": [""]},
    "T14": {"edgecolor": ("#4b0082", "#4b0082"), "subtype": [""]},
}
"""


def generate_color_gradient(
    color_head: str, color_tail: str, color_cnt: int
) -> List[str]:
    """
    Generate a list of colors forming a gradient between color_head and color_tail.
    """
    if color_cnt < 1:
        raise Exception("Color count too small")
    if color_cnt == 1:
        return [color_head]

    try:
        start_rgb = tuple(int(color_head.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
        end_rgb = tuple(int(color_tail.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
    except Exception:
        raise Exception("Invalid color format")

    colors = []
    for i in range(color_cnt):
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i // (color_cnt - 1)
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i // (color_cnt - 1)
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i // (color_cnt - 1)
        color_hex = "#{0:02x}{1:02x}{2:02x}".format(r, g, b)
        colors.append(color_hex)

    return colors


def process_type_info(
    borders: Dict[str, Dict[str, str]],
    is_subtype_only: bool = False,
) -> Dict[str, str]:
    """
    Process the TYPE_INFO to generate a color map and subtype labels.
    """
    color_map = {}
    subtype_cnt = 0
    for type_name, info in borders.items():
        edge_start, edge_end = info["edgecolor"]
        subtypes = info["subtype"]
        type_num = len(subtypes)
        subtype_cnt += type_num
        gradient_colors = generate_color_gradient(edge_start, edge_end, type_num)
        for i, subtype_name in enumerate(subtypes):
            label = subtype_name if is_subtype_only else f"{type_name}_{subtype_name}"
            color_map[label] = gradient_colors[i]

    return color_map


def get_type_info_from_user_map(
    user_map: dict, color_borders: list = COLOR_BORDERS
) -> dict:
    type_info = {}
    type_cnt = 0
    for cluster_id, category in user_map.items():
        if category not in type_info:
            type_info[category] = {
                "edgecolor": color_borders[type_cnt],
                "subtype": [str(cluster_id)],
            }
            type_cnt += 1
        else:
            type_info[category]["subtype"].append(str(cluster_id))

    return type_info


def get_color_map(user_map: dict) -> dict:
    return process_type_info(
        get_type_info_from_user_map(user_map), is_subtype_only=True
    )
