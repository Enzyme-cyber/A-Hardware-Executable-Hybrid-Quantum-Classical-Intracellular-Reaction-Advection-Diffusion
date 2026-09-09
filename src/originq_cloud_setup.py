#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本源量子云前置检查：API Key、后端、芯片信息和可选 Bell 真机测试。"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path


def load_qpanda():
    try:
        import pyqpanda3
        from pyqpanda3.core import CNOT, H, QProg, measure
        from pyqpanda3.qcloud import DataBase, JobStatus, QCloudOptions, QCloudService
    except ImportError as exc:
        raise RuntimeError("未安装 pyqpanda3，请先执行：pip install -U pyqpanda3") from exc
    return pyqpanda3, CNOT, H, QProg, measure, DataBase, JobStatus, QCloudOptions, QCloudService


def get_api_key(prompt: bool) -> str:
    key = os.environ.get("QPANDA_QCLOUD_API_KEY", "").strip()
    if not key and prompt:
        key = getpass.getpass("请输入本源量子云 API Key（输入不会显示）：").strip()
    if not key:
        raise RuntimeError(
            "未找到 QPANDA_QCLOUD_API_KEY。\n"
            "Windows PowerShell：$env:QPANDA_QCLOUD_API_KEY=\"你的API_KEY\"\n"
            "Linux/macOS：export QPANDA_QCLOUD_API_KEY=\"你的API_KEY\""
        )
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本源量子云前置设置与真机连通测试")
    parser.add_argument("--backend", default=os.environ.get("QPANDA_QCLOUD_BACKEND", "auto"))
    parser.add_argument("--prompt-token", action="store_true")
    parser.add_argument("--run-bell", action="store_true")
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--save-report", default="originq_cloud_report.json")
    return parser.parse_known_args()[0]


def is_simulator_name(name: str) -> bool:
    text = name.lower()
    return any(x in text for x in ("amplitude", "simulator", "noise", "density", "stabilizer"))


def choose_backend(service, requested: str, min_qubits: int = 2):
    available = service.backends()
    print("\n账号可见后端：")
    for name, enabled in available.items():
        print(f"  {name}: {'可用' if enabled else '不可用'}")

    if requested != "auto":
        if requested not in available:
            raise RuntimeError(f"账号中不存在后端：{requested}")
        if not available[requested]:
            raise RuntimeError(f"后端当前不可用：{requested}")
        return requested, service.backend(requested)

    candidates = []
    for name, enabled in available.items():
        if not enabled or is_simulator_name(name):
            continue
        backend = service.backend(name)
        try:
            info = backend.chip_info()
            qn = int(info.qubits_num())
            aq = list(info.available_qubits())
            if qn >= min_qubits and len(aq) >= min_qubits:
                candidates.append((len(aq), qn, name, backend))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("没有自动找到可用真实QPU，请使用 --backend 后端名明确指定。")
    candidates.sort(reverse=True)
    _, _, name, backend = candidates[0]
    return name, backend


def probs_to_plain_dict(probs) -> dict[str, float]:
    return {str(k): float(v) for k, v in dict(probs).items()}


def main() -> int:
    args = parse_args()
    pyqpanda3, CNOT, H, QProg, measure, DataBase, JobStatus, QCloudOptions, QCloudService = load_qpanda()
    api_key = get_api_key(args.prompt_token)
    print("pyqpanda3版本：", getattr(pyqpanda3, "__version__", "unknown"))
    service = QCloudService(api_key)
    backend_name, backend = choose_backend(service, args.backend, min_qubits=2)
    print("\n选择后端：", backend_name)

    report = {"pyqpanda3_version": getattr(pyqpanda3, "__version__", "unknown"), "backend": backend_name}
    try:
        info = backend.chip_info()
        report.update({
            "qubits_num": int(info.qubits_num()),
            "available_qubits": list(info.available_qubits()),
            "basic_gates": [str(x) for x in info.get_basic_gates()],
            "topology": info.get_chip_topology(),
        })
        print("芯片总比特：", report["qubits_num"])
        print("当前可用物理比特数：", len(report["available_qubits"]))
        print("原生门：", report["basic_gates"])
    except Exception as exc:
        report["chip_info_warning"] = str(exc)
        print("读取芯片信息失败：", exc)

    if args.run_bell:
        prog = QProg()
        prog << H(0) << CNOT(0, 1)
        prog << measure([0, 1], [0, 1])
        options = QCloudOptions()
        options.set_mapping(True)
        options.set_optimization(True)
        options.set_amend(True)
        job = backend.run(prog, shots=args.shots, options=options)
        print("Bell任务ID：", job.job_id())
        result = job.result()
        report["bell_job_id"] = job.job_id()
        report["bell_status"] = str(result.job_status())
        if result.job_status() == JobStatus.FAILED:
            report["bell_error"] = result.error_message()
            print("Bell任务失败：", result.error_message())
        else:
            try:
                probs = result.get_probs(base=DataBase.Binary)
            except Exception:
                probs = result.get_counts(base=DataBase.Binary)
            report["bell_probs"] = probs_to_plain_dict(probs)
            report["timing_info"] = result.timing_info()
            print("Bell结果：", report["bell_probs"])

    Path(args.save_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告已保存：", args.save_report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n前置检查失败：", exc, file=sys.stderr)
        raise
