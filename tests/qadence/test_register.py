from __future__ import annotations

import json
import os
from math import isclose

import networkx as nx
import numpy as np
from metrics import ATOL_64
from pytest import approx

from qadence.register import Register


def calc_dist(graph: nx.Graph) -> np.ndarray:
    coords = {i: node["pos"] for i, node in graph.nodes.items()}
    coords_np = np.array(list(coords.values()))
    center = np.mean(coords_np, axis=0)
    distances = np.array(np.sqrt(np.sum((coords_np - center) ** 2, axis=1)))
    return distances


def test_register() -> None:
    # create register with number of qubits only
    reg = Register(4, spacing=4.0)
    assert reg.n_qubits == 4
    assert reg.min_distance == 4.0

    # create register from arbitrary graph
    graph = nx.Graph()
    graph.add_edge(0, 1)
    reg = Register(graph)
    assert reg.n_qubits == 2
    assert isclose(reg.min_distance, 0.0, rel_tol=ATOL_64)

    # test linear lattice node number
    spacing = 2.0
    r = Register.line(4, spacing=spacing)
    assert len(r.graph) == 4
    assert r == Register.lattice("line", 4, spacing=spacing)
    assert isclose(sum(r.edge_distances.values()), 3 * spacing, rel_tol=ATOL_64)
    assert len(r.distances) == len(r.all_edges)

    # test circular lattice node number
    r = Register.circle(8)
    assert len(r.graph) == 8
    assert r == Register.lattice("circle", 8)
    assert isclose(r.min_distance, 1.0, rel_tol=ATOL_64)

    # test shape of circular lattice
    distances = calc_dist(r.graph)
    assert distances == approx(np.ones(len(distances)) * distances[0])

    # test square loop lattice node number
    r = Register.square(4)
    assert len(r.graph) == 12
    assert r == Register.lattice("square", 4)

    # test rectangular lattice node number
    r = Register.rectangular_lattice(2, 3)
    assert len(r.graph) == 6
    assert r == Register.lattice("rectangular_lattice", 2, 3)

    # test shape of rectangular lattice
    r = Register.rectangular_lattice(2, 2)
    distances = calc_dist(r.graph)
    assert distances == approx(np.ones(len(distances)) * distances[0])

    # test triangular lattice node number
    r = Register.triangular_lattice(1, 3)
    assert len(r.graph) == 5
    assert r == Register.lattice("triangular_lattice", 1, 3)

    # test shape of triangular lattice
    r = Register.triangular_lattice(1, 1)
    distances = calc_dist(r.graph)
    assert distances == approx(np.ones(len(distances)) * distances[0])

    # test honeycomb lattice node number
    r = Register.honeycomb_lattice(1, 3)
    assert len(r.graph) == 14
    assert r == Register.lattice("honeycomb_lattice", 1, 3)

    # test shape of honeycomb lattice
    r = Register.honeycomb_lattice(1, 1)
    distances = calc_dist(r.graph)
    assert distances == approx(np.ones(len(distances)) * distances[0])

    # test arbitrary lattice node number
    r = Register.from_coordinates([(0, 1), (0, 2), (0, 3), (1, 3)])
    assert len(r.graph) == 4
    assert isclose(r.min_distance, 1.0, rel_tol=ATOL_64)

    # test rescale coordinates
    r = r.rescale_coords(scaling=2.0)
    assert isclose(r.min_distance, 2.0, rel_tol=ATOL_64)


def test_register_to_dict(BasicRegister: Register) -> None:
    reg = BasicRegister
    reg_dict = reg._to_dict()
    reg_from_dict = Register._from_dict(reg_dict)
    assert reg == reg_from_dict


def test_json_dump_load_register_to_dict(BasicRegister: Register) -> None:
    reg = BasicRegister
    reg_dict = reg._to_dict()
    dumpedregdict = json.dumps(reg_dict)
    file_name = "tmp.json"
    with open(file_name, "w") as file:
        file.write(dumpedregdict)
    with open(file_name, "r") as file:
        loaded_dict = json.load(file)

    os.remove(file_name)
    reg_from_loaded_dict = Register._from_dict(loaded_dict)
    assert reg == reg_from_loaded_dict
