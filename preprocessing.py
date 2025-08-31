from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import gzip
import json
import logging
from joblib import Parallel, delayed

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse

from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr


def transform_pcap(path, output_path: Optional[Path] = None, output_batch_size=10000):
    if output_path is None:
        return

    # 检查是否已经处理完成
    success_file = Path(str(output_path.absolute()) + "_SUCCESS")
    if success_file.exists():
        print(f"{output_path} 已处理完成，跳过")
        return

    print("Processing", path)

    rows = []
    batch_index = 0
    
    # 检查是否存在部分文件，如果存在则从最后一个部分文件继续
    existing_parts = []
    for part_file in output_path.parent.glob(f"{output_path.name}_part_*.json.gz"):
        existing_parts.append(part_file)
    
    if existing_parts:
        # 找到最后一个部分文件的索引
        existing_parts.sort()
        last_part = existing_parts[-1]
        # 从文件名中提取批次索引
        batch_index = int(last_part.name.split("_part_")[1].split(".")[0]) + 1
        print(f"发现已存在的部分文件，从批次 {batch_index} 继续处理")
    
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split(".")[0].lower()
            app_label = PREFIX_TO_APP_ID.get(prefix)
            traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
            row = {
                "app_label": app_label,
                "traffic_label": traffic_label,
                "feature": arr.todense().tolist()[0],
            }
            rows.append(row)

        # write every batch_size packets, by default 10000
        if rows and i > 0 and i % output_batch_size == 0:
            part_output_path = Path(
                str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz"
            )
            with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
                for row in rows:
                    f_out.write(f"{json.dumps(row)}\n")
            batch_index += 1
            rows.clear()

    # final write
    if rows:
        part_output_path = Path(
            str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz"
        )
        with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
            for row in rows:
                f_out.write(f"{json.dumps(row)}\n")

    # write success file
    with Path(str(output_path.absolute()) + "_SUCCESS").open("w") as f:
        f.write("")

    print(output_path, "Done")


@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing raw pcap files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting preprocessed files",
    required=True,
)
@click.option("-n", "--njob", default=4, help="num of executors", type=int)
@click.option("--resume", is_flag=True, help="resume from last interrupted point", default=True)
def main(source, target, njob, resume):
    data_dir_path = Path(source)
    target_dir_path = Path(target)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有需要处理的文件
    pcap_files = sorted(data_dir_path.iterdir())
    
    if resume:
        # 过滤掉已经处理完成的文件
        files_to_process = []
        for pcap_path in pcap_files:
            output_path = target_dir_path / (pcap_path.name + ".transformed")
            success_file = Path(str(output_path.absolute()) + "_SUCCESS")
            if not success_file.exists():
                files_to_process.append(pcap_path)
            else:
                print(f"跳过已完成的文件: {pcap_path.name}")
        
        print(f"总共 {len(pcap_files)} 个文件，其中 {len(files_to_process)} 个需要处理")
        
        if not files_to_process:
            print("所有文件都已处理完成！")
            return
    else:
        files_to_process = pcap_files
        print(f"将处理所有 {len(files_to_process)} 个文件")
    
    if njob == 1:
        for pcap_path in files_to_process:
            transform_pcap(
                pcap_path, target_dir_path / (pcap_path.name + ".transformed")
            )
    else:
        Parallel(n_jobs=njob, verbose=10)(
            delayed(transform_pcap)(
                pcap_path, target_dir_path / (pcap_path.name + ".transformed")
            )
            for pcap_path in files_to_process
        )


if __name__ == "__main__":
    main()  # pyright: ignore [reportMissingParameter]
