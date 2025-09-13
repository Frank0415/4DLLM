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

INITIALIZED_COLORS = [
    [
        "#74c0fc",
        "#66d9e8",
        "#4dabf7",
        "#3bc9db",
        "#339af0",
        "#22b8cf",
        "#228be6",
        "#15aabf",
        "#61b6fa",
        "#51d1e2",
        "#40a3f4",
        "#2fc1d5",
        "#2b93eb",
        "#1cb1c7",
        "#117cdc",
        "#089caf",
    ],
    [
        "#c0eb75",
        "#b5e760",
        "#8ce99a",
        "#a9e34b",
        "#7be28d",
        "#9fde3c",
        "#69db7c",
        "#94d82d",
        "#8bd126",
        "#5dd571",
        "#82c91e",
        "#51cf66",
        "#49c85f",
        "#70ba0f",
        "#40c057",
        "#2fb148",
    ],
    [
        "#ffd43b",
        "#ffa94d",
        "#fcc419",
        "#ff922b",
        "#fab005",
        "#fd7e14",
        "#f59f00",
        "#f76707",
        "#fecc2a",
        "#ff9e3c",
        "#fbba0f",
        "#fe8820",
        "#f8a803",
        "#fa730e",
        "#f08e00",
        "#f15000",
    ],
    [
        "#faa2c1",
        "#f783ac",
        "#ff8787",
        "#f06595",
        "#ff6b6b",
        "#e64980",
        "#fa5252",
        "#f03e3e",
        "#f892b6",
        "#fb8599",
        "#f7768e",
        "#f76880",
        "#f25a75",
        "#f04d69",
        "#f54848",
        "#e62a2a",
    ],
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
        raise Exception(f"Color count must be at least 1, got: {color_cnt}")
    if color_cnt == 1:
        return [color_head]

    try:
        start_rgb = tuple(int(color_head.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
        end_rgb = tuple(int(color_tail.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
    except Exception:
        raise Exception("Invalid hex color format. Expected format: #RRGGBB")

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
    use_initialized_color: bool = False,
) -> Dict[str, str]:
    """
    Process the TYPE_INFO to generate a color map and subtype labels.
    """
    color_map = {}
    subtype_cnt = 0
    type_id = 0
    for type_name, info in borders.items():
        edge_start, edge_end = info["edgecolor"]
        subtypes = info["subtype"]
        type_num = len(subtypes)
        subtype_cnt += type_num
        if not use_initialized_color:
            gradient_colors = generate_color_gradient(edge_start, edge_end, type_num)
        else:
            gradient_colors = INITIALIZED_COLORS[type_id]
        for i, subtype_name in enumerate(subtypes):
            label = subtype_name if is_subtype_only else f"{type_name}_{subtype_name}"
            color_map[label] = gradient_colors[i]
        type_id += 1

    return color_map


def get_type_info_from_user_map(user_map: dict) -> dict:
    type_info = {}
    type_cnt = 0
    for cluster_id, category in user_map.items():
        if category not in type_info:
            type_info[category] = {
                "edgecolor": COLOR_BORDERS[type_cnt],
                "subtype": [str(cluster_id)],
            }
            type_cnt += 1
        else:
            type_info[category]["subtype"].append(str(cluster_id))

    return type_info


def get_color_map(user_map: dict, use_initialized_color: bool = True) -> dict:
    return process_type_info(
        get_type_info_from_user_map(user_map),
        is_subtype_only=True,
        use_initialized_color=use_initialized_color,
    )
