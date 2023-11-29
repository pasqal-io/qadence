from __future__ import annotations

import time
from typing import Any, Callable

import torch
from memory_profiler import memory_usage

from qadence import (
    QuantumCircuit,
    QuantumModel,
    hea,
    total_magnetization,
)
from qadence.types import BackendName, DiffMode

N_QUBITS = 20
BATCH_SIZE = 1
DEPTH = 10
BACKEND = BackendName.PYQTORCH
DIFF_MODE = "adjoint"


def measure_max_memory(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> float:
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, max_usage=True)
        return mem_usage

    return wrapper


def measure_time(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> float:
        start = time.time()
        func(*args)
        end = time.time()
        return end - start

    return wrapper


@measure_time
def _model_expectation(
    diff_mode: DiffMode = DIFF_MODE,
    n_qubits: int = N_QUBITS,
    depth: int = DEPTH,
    batch_size: int = BATCH_SIZE,
) -> None:
    model = QuantumModel(
        QuantumCircuit(n_qubits, hea(n_qubits, depth)),
        observable=total_magnetization(n_qubits),
        backend=BACKEND,
        diff_mode=diff_mode,
    )
    nx = torch.rand(batch_size, requires_grad=True)
    ny = torch.rand(batch_size, requires_grad=True)
    values = {"theta_0": nx, "theta_1": ny}
    expvals = model.expectation(values)
    model.zero_grad()
    loss = torch.mean(expvals)
    loss.backward()


mem_ad_res = {
    1: 1004.90625,
    2: 1365.03125,
    3: 1688.1875,
    4: 2039.078125,
    5: 2330.40625,
    6: 2667.796875,
    7: 2976.5,
    8: 3301.1875,
    9: 3622.234375,
}
mem_adjoint_res = {
    1: 781.0,
    2: 820.96875,
    3: 831.5625,
    4: 853.46875,
    5: 822.59375,
    6: 841.4375,
    7: 825.8125,
    8: 827.578125,
    9: 830.453125,
}

# if __name__ == "__main__":
#     time_usages_ad = []
#     time_usages_adjoint = []

#     depth_to_time_ad = {}
#     depth_to_time_adjoint = {}
#     for depth in range(1, DEPTH):
#         ad_usage = _model_expectation("ad", N_QUBITS, depth, BATCH_SIZE)
#         adjoint_usage = _model_expectation("adjoint", N_QUBITS, depth, BATCH_SIZE)
#         depth_to_time_ad[depth] = ad_usage
#         depth_to_time_adjoint[depth] = adjoint_usage

#     print(depth_to_time_ad)
#     print(depth_to_time_adjoint)

#     # run_20qubits = {'ad': ad_res, 'adjoint': adjoint_res}

#     import matplotlib.pyplot as plt

#     depths = [i+1 for i in range(len(depth_to_time_adjoint.values()))]
#     plt.plot(depths , depth_to_time_ad.values())
#     plt.plot(depths , depth_to_time_adjoint.values())
#     plt.xlabel('hea_depth')
#     plt.ylabel('memory_usage')
#     plt.legend(['ad', 'adjoint'])
#     plt.title('hea(n_qubits=20, depth=X-AXIS)')
#     plt.show()

# mem_ad_res=  {1: 1004.90625, 2: 1365.03125, 3: 1688.1875, 4: 2039.078125, 5: 2330.40625, 6: 2667.796875, 7: 2976.5, 8: 3301.1875, 9: 3622.234375}
# mem_adjoint_res = {1: 781.0, 2: 820.96875, 3: 831.5625, 4: 853.46875, 5: 822.59375, 6: 841.4375, 7: 825.8125, 8: 827.578125, 9: 830.453125}

# if __name__ == "__main__":
#     mem_usages_ad = []
#     mem_usages_adjoint = []

#     depth_to_mem_ad = {}
#     depth_to_mem_adjoint = {}
#     for depth in range(1, DEPTH):
#         ad_usage = _model_expectation("ad", N_QUBITS, depth, BATCH_SIZE)
#         adjoint_usage = _model_expectation("adjoint", N_QUBITS, depth, BATCH_SIZE)
#         depth_to_mem_ad[depth] = ad_usage
#         depth_to_mem_adjoint[depth] = adjoint_usage

#     print(depth_to_mem_ad)
#     print(depth_to_mem_adjoint)

#     run_20qubits = {'ad': ad_res, 'adjoint': adjoint_res}

#     import matplotlib.pyplot as plt

#     depths = [i+1 for i in range(len(ad_res.values()))]
#     plt.plot(depths , ad_res.values())
#     plt.plot(depths , adjoint_res.values())
#     plt.xlabel('hea_depth')
#     plt.ylabel('memory_usage')
#     plt.legend(['ad', 'adjoint'])
#     plt.title('hea(n_qubits=20, depth=X-AXIS)')
#     plt.show()


ad_times = {
    1: 1.2803080081939697,
    2: 1.6290419101715088,
    3: 2.4054598808288574,
    4: 3.1631059646606445,
    5: 4.1364829540252686,
    6: 4.69971776008606,
    7: 5.466135025024414,
    8: 6.270625829696655,
    9: 7.085698127746582,
}
adjoint_times = {
    1: 2.094740867614746,
    2: 4.136566162109375,
    3: 6.4286887645721436,
    4: 8.321171045303345,
    5: 10.796124935150146,
    6: 12.045706033706665,
    7: 14.314074039459229,
    8: 15.948777914047241,
    9: 18.345812797546387,
}

if __name__ == "__main__":
    # time_usages_ad = []
    # time_usages_adjoint = []

    # depth_to_time_ad = {}
    # depth_to_time_adjoint = {}
    # for depth in range(1, DEPTH):
    #     ad_usage = _model_expectation("ad", N_QUBITS, depth, BATCH_SIZE)
    #     adjoint_usage = _model_expectation("adjoint", N_QUBITS, depth, BATCH_SIZE)
    #     depth_to_time_ad[depth] = ad_usage
    #     depth_to_time_adjoint[depth] = adjoint_usage

    # print(depth_to_time_ad)
    # print(depth_to_time_adjoint)

    # # run_20qubits = {'ad': ad_res, 'adjoint': adjoint_res}

    import matplotlib.pyplot as plt

    depths = [i + 1 for i in range(len(ad_times.values()))]
    plt.plot(depths, ad_times.values())
    plt.plot(depths, adjoint_times.values())
    plt.xlabel("hea_depth")
    plt.ylabel("runtime")
    plt.legend(["ad", "adjoint"])
    plt.title("hea(n_qubits=20, depth=X-AXIS)")
    plt.show()
