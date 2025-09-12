import numpy as np
import scipy.io
import os


def process_one_mib(mib_path, out_root, scan_size=(256, 256)):
    """
    处理单个 .mib 文件，生成对应 .mat 文件夹
    """
    base_name = os.path.basename(mib_path)
    name_without_ext = os.path.splitext(base_name)[0]

    # 输出文件夹
    mat_folder = os.path.join(out_root, name_without_ext)
    os.makedirs(mat_folder, exist_ok=True)

    print(f"\n=== Processing {mib_path} ===")

    # 读 MIB
    data_cube = load_mib(
        mib_path,
        mem="MEMMAP",
        binfactor=1,
        reshape=True,
        scan=scan_size,
    )  # shape = (Rx, Ry, H, W) or (scan[0], scan[1], H, W)

    H, W = data_cube.shape[2], data_cube.shape[3]

    # 每行写一个 mat
    for i in range(scan_size[0]):
        row_patterns = data_cube[i, :, :, :]  # shape = (256, H, W)

        mat_filename = os.path.join(mat_folder, f"{i + 1}.mat")
        scipy.io.savemat(mat_filename, {"data": row_patterns})

        if (i + 1) % 50 == 0 or (i + 1) == scan_size[0]:
            print(f"Saved row {i + 1} to {mat_filename} shape={row_patterns.shape}")

    print(f"Done: {mib_path} -> {mat_folder}")


# def batch_process_mibs(mib_folder, out_root):
#     """
#     批量处理文件夹里所有 .mib 文件
#     """
#     mib_files = [f for f in os.listdir(mib_folder) if f.lower().endswith(".mib")]

#     print(f"Found {len(mib_files)} MIB files in {mib_folder}")

#     for mib_file in mib_files:
#         mib_path = os.path.join(mib_folder, mib_file)
#         process_one_mib(mib_path, out_root)

def load_mib(file_path, mem="MEMMAP", binfactor=1, reshape=True, scan=(256, 256)):
    """Minimal loader for Merlin .mib files that returns a numpy datacube.

    Returns an ndarray shaped (Rx, Ry, H, W) when reshape=True, else (N, 1, H, W).
    This implementation replicates the behavior required by step1_mib2mat.py.
    """
    header = parse_hdr(file_path)
    width = header["Detector width"]
    height = header["Detector height"]
    width_height = width * height

    scan = scan_size(file_path, scan)

    data_mem = get_mib_memmap(file_path)
    depth = get_mib_depth(header, file_path)
    hdr_bits = get_hdr_bits(header)

    # reshape raw memmap into frames and strip header bits per-frame
    if header["Counter Depth (number)"] == 1:
        # 1-bit frames are packed; here we unpack to uint8 per-pixel
        # reshape into frames including per-frame header bytes
        frames = data_mem.reshape(-1, int(width_height // 8 + hdr_bits))
        frames = frames[:, hdr_bits:]
        # unpack bits to 0/1 per pixel
        unpacked = np.unpackbits(frames.view(np.uint8), axis=1)
        # Take the first width*height bits
        unpacked = unpacked[:, :width_height]
        data = unpacked.reshape(-1, width, height)
    else:
        frames = data_mem.reshape(-1, int(width_height + hdr_bits))
        frames = frames[:, hdr_bits:]
        data = frames.reshape(-1, width, height)

    if header.get("raw", "MIB") != "MIB":
        raise RuntimeError("Only MIB raw format is supported by this loader")

    if reshape:
        # reshape into scan grid
        try:
            data = data.reshape(scan[0], scan[1], width, height)
        except Exception:
            # fallback: try (scan[1], scan[0]) ordering
            data = data.reshape(scan[1], scan[0], width, height)
    else:
        data = data[:, None, :, :]

    if mem == "RAM":
        data = np.array(data)

    # binfactor not implemented here; return as-is
    return data


# Helper functions (simplified versions)


def manageHeader(fname):
    Header = str()
    with open(fname, "rb") as input:
        aByte = input.read(1)
        Header += str(aByte.decode("ascii", errors="ignore"))
        while aByte and ord(aByte) != 0:
            aByte = input.read(1)
            Header += str(aByte.decode("ascii", errors="ignore"))

    elements_in_header = Header.split(",")
    DataOffset = int(elements_in_header[2])
    NChips = int(elements_in_header[3])
    PixelDepthInFile = elements_in_header[6]
    sensorLayout = elements_in_header[7].strip()
    Timestamp = elements_in_header[9]
    shuttertime = float(elements_in_header[10])

    try:
        ScanX, ScanY = None, None
        ScanX = int(elements_in_header[18])
        ScanY = int(elements_in_header[19])
    except Exception:
        ScanX, ScanY = None, None

    if PixelDepthInFile == "R64":
        bitdepth = int(elements_in_header[18])
    elif PixelDepthInFile == "U16":
        bitdepth = 12
    elif PixelDepthInFile == "U08":
        bitdepth = 6
    elif PixelDepthInFile == "U32":
        bitdepth = 24

    hdr = (
        DataOffset,
        NChips,
        PixelDepthInFile,
        sensorLayout,
        Timestamp,
        shuttertime,
        bitdepth,
        ScanX,
        ScanY,
    )
    return hdr


def scan_size(path, scan):
    header_path = path[:-3] + "hdr"
    if os.path.exists(header_path):
        result = {}
        with open(header_path, encoding="UTF-8") as f:
            for line in f:
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                k, v = parts
                k = k.rstrip(":")
                v = v.rstrip("\n")
                result[k] = v
                if "ScanX" in result and "ScanY" in result:
                    return (int(result["ScanY"]), int(result["ScanX"]))
    return scan


def parse_hdr(fp):
    read_hdr = manageHeader(fp)
    hdr_info = {}
    if read_hdr[3] == "1x1":
        hdr_info["Detector width"] = 256
        hdr_info["Detector height"] = 256
    elif read_hdr[3] == "2x2":
        hdr_info["Detector width"] = 512
        hdr_info["Detector height"] = 512
    elif read_hdr[3] == "2x2G":
        hdr_info["Detector width"] = 514
        hdr_info["Detector height"] = 514

    hdr_info["scan size X"] = read_hdr[7]
    hdr_info["scan size Y"] = read_hdr[8]
    hdr_info["Assembly Size"] = read_hdr[3]
    hdr_info["offset"] = read_hdr[0]
    hdr_info["data-type"] = "unsigned"
    data_length = read_hdr[6]
    if read_hdr[6] == "1":
        hdr_info["data-length"] = "8"
    else:
        cd_int = int(read_hdr[6])
        hdr_info["data-length"] = str(int((cd_int + cd_int / 3)))
    hdr_info["Counter Depth (number)"] = int(read_hdr[6])
    hdr_info["raw"] = "R64" if read_hdr[2] == "R64" else "MIB"
    hdr_info["byte-order"] = "dont-care"
    hdr_info["record-by"] = "image"
    hdr_info["title"] = fp.split(".")[0]
    hdr_info["date"] = None
    hdr_info["time"] = None
    hdr_info["data offset"] = read_hdr[0]
    return hdr_info


def get_mib_memmap(fp, mmap_mode="r"):
    hdr_info = parse_hdr(fp)
    data_length = hdr_info["data-length"]
    data_type = hdr_info["data-type"]
    endian = hdr_info["byte-order"]
    read_offset = 0

    if data_type == "signed":
        data_type = "int"
    elif data_type == "unsigned":
        data_type = "uint"
    elif data_type == "float":
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    endian = ">"
    data_type += str(int(data_length))
    if data_type == "uint1":
        data_type = "uint8"
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    data_mem = np.memmap(fp, offset=read_offset, dtype=data_type, mode=mmap_mode)
    return data_mem


def get_mib_depth(hdr_info, fp):
    if hdr_info["Assembly Size"] == "2x2":
        mib_file_size_dict = {
            "1": 33536,
            "6": 262912,
            "12": 525056,
            "24": 1049344,
        }
    elif hdr_info["Assembly Size"] == "1x1":
        mib_file_size_dict = {
            "1": 8576,
            "6": 65920,
            "12": 131456,
            "24": 262528,
        }
    elif hdr_info["Assembly Size"] == "2x2G":
        mib_file_size_dict = {
            "1": 33536,
            "6": 264964,
            "12": 529160,
            "24": 1057552,
        }
    else:
        raise RuntimeError("Unknown assembly size")

    file_size = os.path.getsize(fp)
    if hdr_info["raw"] == "R64":
        single_frame = mib_file_size_dict.get(str(hdr_info["Counter Depth (number) "]))
        depth = int(file_size / single_frame)
    elif hdr_info["raw"] == "MIB":
        if hdr_info["Counter Depth (number)"] == 1:
            single_frame = mib_file_size_dict.get("6")
            depth = int(file_size / single_frame)
        else:
            single_frame = mib_file_size_dict.get(
                str(hdr_info["Counter Depth (number)"])
            )
            depth = int(file_size / single_frame)
    return depth


def get_hdr_bits(hdr_info):
    data_length = hdr_info["data-length"]
    data_type = hdr_info["data-type"]

    if data_type == "signed":
        data_type = "int"
    elif data_type == "unsigned":
        data_type = "uint"
    elif data_type == "float":
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    endian = ">"
    data_type += str(int(data_length))
    if data_type == "uint1":
        data_type = "uint8"
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    if data_length == "1":
        hdr_multiplier = 1
    else:
        hdr_multiplier = (int(data_length) / 8) ** -1

    hdr_bits = int(hdr_info.get("data offset", 0) * hdr_multiplier)
    return hdr_bits
