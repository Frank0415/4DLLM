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
        "#80d926",
        "#73cc26",
        "#66bf26",
        "#59b326",
        "#4da626",
        "#409926",
        "#338c26",
        "#267f26",
        "#1a7326",
        "#0d6626",
        "#8cdd33",
        "#99e640",
        "#a6ef4d",
        "#b3f259",
        "#bff566",
        "#ccf873",
    ],
    [
        "#ffdd33",
        "#ffd426",
        "#ffcb1a",
        "#ffc20d",
        "#ffb900",
        "#f2b000",
        "#e6a700",
        "#d99e00",
        "#cc9500",
        "#bf8c00",
        "#ffe640",
        "#ffea4d",
        "#ffed59",
        "#fff166",
        "#fff473",
        "#fff880",
    ],
    [
        "#4da6ff",
        "#3399ff",
        "#1a8cff",
        "#007fff",
        "#0073e6",
        "#0066cc",
        "#0059b3",
        "#004c99",
        "#003f80",
        "#003266",
        "#6bb3ff",
        "#80bfff",
        "#99ccff",
        "#b3d9ff",
        "#cce6ff",
        "#e6f2ff",
    ],
    [
        "#ffffff",
        "#fdf9ff",
        "#fbf3ff",
        "#f9edff",
        "#f7e7ff",
        "#f5e1ff",
        "#f3dbff",
        "#f1d5ff",
        "#efcfff",
        "#edc9ff",
        "#ebc3ff",
        "#e9bdff",
        "#e7b7ff",
        "#e5b1ff",
        "#e3abff",
        "#e1a5ff",
    ],
]


def generate_color_gradient(
    color_head: str, color_tail: str, color_cnt: int
) -> List[str]:
    """
    Generate a list of colors forming a gradient between color_head and color_tail.
    """
    if color_cnt < 1:
        raise ValueError(f"Color count must be at least 1, got: {color_cnt}")
    if color_cnt == 1:
        return [color_head]

    try:
        start_rgb = tuple(int(color_head.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
        end_rgb = tuple(int(color_tail.lstrip("#")[i : i + 2], 16) for i in [0, 2, 4])
    except Exception:
        raise ValueError("Invalid hex color format. Expected format: #RRGGBB")

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
