from __future__ import annotations

from typing import Any, Callable
import numpy as np
from sympy import Basic

from qadence.backend import BackendConfiguration
from qadence.blocks import chain, kron
from qadence.blocks.abstract import AbstractBlock
from qadence.blocks.composite import ChainBlock, KronBlock
from qadence.blocks.utils import tag, add
from qadence.parameters import Parameter
from qadence.circuit import QuantumCircuit
from qadence.constructors import (
    analog_feature_map,
    feature_map,
    hamiltonian_factory,
    rydberg_feature_map,
    rydberg_hea,
    rydberg_tower_feature_map,
)
from qadence.constructors.ala import ala_digital
from qadence.constructors.hamiltonians import ObservableConfig, TDetuning
from qadence.constructors.hea import hea_digital, hea_sDAQC
from qadence.constructors.iia import iia
from qadence.measurements import Measurements
from qadence.noise import NoiseHandler
from qadence.operations import CNOT, RX, RY, RZ, I
from qadence.register import Register
from qadence.types import (
    AnsatzType,
    BackendName,
    DiffMode,
    InputDiffMode,
    Interaction,
    MultivariateStrategy,
    ReuploadScaling,
    Strategy,
    TParameter,
    TArray,
)

from .config import AnsatzConfig, FeatureMapConfig
from .models import QNN


def _create_support_arrays(
    num_qubits: int,
    num_features: int,
    multivariate_strategy: MultivariateStrategy,
) -> list[tuple[int, ...]]:
    """
    Create the support arrays for the digital feature map.

    Args:
        num_qubits (int): The number of qubits.
        num_features (int): The number of features.
        multivariate_strategy (MultivariateStrategy): The multivariate encoding strategy.
            Either 'MultivariateStrategy.SERIES' or 'MultivariateStrategy.PARALLEL'.

    Returns:
        list[tuple[int, ...]]: The list of support arrays. ith element of the list is the support
            array for the ith feature.

    Raises:
        ValueError: If the number of features is greater than the number of qubits
            and the strategy is `MultivariateStrategy.PARALLEL` not possible to assign a support
            array to each feature.
        ValueError: If the multivariate strategy is not `MultivariateStrategy.SERIES` or
            `MultivariateStrategy.PARALLEL`.
    """
    if multivariate_strategy == MultivariateStrategy.SERIES:
        return [tuple(range(num_qubits)) for i in range(num_features)]
    elif multivariate_strategy == MultivariateStrategy.PARALLEL:
        if num_features <= num_qubits:
            return [tuple(x.tolist()) for x in np.array_split(np.arange(num_qubits), num_features)]
        else:
            raise ValueError(
                f"Number of features {num_features} must be less than or equal to the number of \
                qubits {num_qubits} if the features are to be encoded parallely."
            )
    else:
        raise ValueError(
            f"Invalid encoding strategy {multivariate_strategy} provided. Only \
                `MultivariateStrategy.SERIES` or `MultivariateStrategy.PARALLEL` are allowed."
        )


def _encode_features_series_digital(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Encode the features in series using digital feature map.

    Args:
        register (int | Register): The number of qubits or the register object.
        config (FeatureMapConfig): The feature map configuration.

    Returns:
        list[AbstractBlock]: The list of digital feature map blocks.
    """
    num_qubits = register if isinstance(register, int) else register.n_qubits

    support_arrays_list = _create_support_arrays(
        num_qubits=num_qubits,
        num_features=config.num_features,
        multivariate_strategy=config.multivariate_strategy,
    )

    support_arrays = {
        key: support
        for key, support in zip(config.inputs, support_arrays_list)  # type: ignore[union-attr, arg-type]
    }

    num_uploads = {key: value + 1 for key, value in config.num_repeats.items()}  # type: ignore[union-attr]

    if config.param_prefix is None:
        param_prefixes = {feature: None for feature in config.inputs}  # type: ignore[union-attr]
    else:
        param_prefixes = {feature: f"{config.param_prefix}_{feature}" for feature in config.inputs}  # type: ignore

    fm_blocks: list[AbstractBlock] = []

    for i in range(max(num_uploads.values())):
        for feature in config.inputs:  # type: ignore[union-attr]
            if num_uploads[feature] > 0:
                fm_blocks.append(
                    feature_map(
                        num_qubits,
                        support=support_arrays[feature],
                        param=feature,
                        op=config.operation,  # type: ignore[arg-type]
                        fm_type=config.basis_set[feature],  # type: ignore[arg-type, index]
                        reupload_scaling=config.reupload_scaling[feature],  # type: ignore[arg-type, index]
                        feature_range=config.feature_range[feature],  # type: ignore[arg-type, index]
                        target_range=config.target_range[feature],  # type: ignore[arg-type, index]
                        param_prefix=param_prefixes[feature],
                    )
                )
            num_uploads[feature] -= 1

    return fm_blocks


def _encode_features_parallel_digital(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Encode the features in parallel using digital feature map.

    Args:
        register (int | Register): The number of qubits or the register object.
        config (FeatureMapConfig): The feature map configuration.

    Returns:
        list[AbstractBlock]: The list of digital feature map blocks.
    """
    num_qubits = register if isinstance(register, int) else register.n_qubits

    support_arrays_list = _create_support_arrays(
        num_qubits=num_qubits,
        num_features=config.num_features,
        multivariate_strategy=config.multivariate_strategy,
    )

    support_arrays = {
        key: support
        for key, support in zip(config.inputs, support_arrays_list)  # type: ignore[union-attr, arg-type]
    }

    num_uploads = {key: value + 1 for key, value in config.num_repeats.items()}  # type: ignore[union-attr]

    if config.param_prefix is None:
        param_prefixes = {feature: None for feature in config.inputs}  # type: ignore[union-attr]
    else:
        param_prefixes = {feature: f"{config.param_prefix}_{feature}" for feature in config.inputs}  # type: ignore

    fm_blocks: list[AbstractBlock] = []

    for i in range(max(num_uploads.values())):
        fm_layer = []
        for feature in config.inputs:  # type: ignore[union-attr]
            if num_uploads[feature] > 0:
                fm_layer.append(
                    feature_map(
                        len(support_arrays[feature]),
                        support=support_arrays[feature],
                        param=feature,
                        op=config.operation,  # type: ignore[arg-type]
                        fm_type=config.basis_set[feature],  # type: ignore[index]
                        reupload_scaling=config.reupload_scaling[feature],  # type: ignore[index]
                        feature_range=config.feature_range[feature],  # type: ignore[arg-type, index]
                        target_range=config.target_range[feature],  # type: ignore[arg-type, index]
                        param_prefix=param_prefixes[feature],
                    )
                )
            num_uploads[feature] -= 1

        fm_blocks.append(kron(*fm_layer))

    return fm_blocks


def _create_digital_fm(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Create the digital feature map.

    Args:
        register (int | Register): The number of qubits or the register object.
        config (FeatureMapConfig): The feature map configuration.

    Returns:
        list[AbstractBlock]: The list of digital feature map blocks.

    Raises:
        ValueError: If the encoding strategy is invalid. Only `MultivariateStrategy.SERIES` or
            `MultivariateStrategy.PARALLEL` are allowed.
    """
    if config.multivariate_strategy == MultivariateStrategy.SERIES:
        fm_blocks = _encode_features_series_digital(register, config)
    elif config.multivariate_strategy == MultivariateStrategy.PARALLEL:
        fm_blocks = _encode_features_parallel_digital(register, config)
    else:
        raise ValueError(
            f"Invalid encoding strategy {config.multivariate_strategy} provided. Only\
            `MultivariateStrategy.SERIES` or `MultivariateStrategy.PARALLEL` are allowed."
        )

    return fm_blocks


def _create_analog_fm(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Create the analog feature map.

    Args:
        register (int | Register): The number of qubits or the register object.
        config (FeatureMapConfig): The feature map configuration.

    Returns:
        list[AbstractBlock]: The list of analog feature map blocks.
    """

    num_uploads = {key: value + 1 for key, value in config.num_repeats.items()}  # type: ignore[union-attr]

    fm_blocks: list[AbstractBlock] = []

    for i in range(max(num_uploads.values())):
        for feature in config.inputs:  # type: ignore[union-attr]
            if num_uploads[feature] > 0:
                fm_blocks.append(
                    analog_feature_map(
                        param=feature,
                        op=config.operation,  # type: ignore[arg-type]
                        fm_type=config.basis_set[feature],  # type: ignore[arg-type, index]
                        reupload_scaling=config.reupload_scaling[feature],  # type: ignore[arg-type, index]
                        feature_range=config.feature_range[feature],  # type: ignore[arg-type, index]
                        target_range=config.target_range[feature],  # type: ignore[arg-type, index]
                    )
                )
            num_uploads[feature] -= 1

    return fm_blocks


def _encode_feature_rydberg(
    num_qubits: int,
    param: str,
    reupload_scaling: ReuploadScaling,
) -> AbstractBlock:
    """
    Encode features using a Rydberg feature map.

    Args:
        num_qubits (int): The number of qubits to encode the features on.
        param (str): The parameter prefix to use for the feature map parameter names.
            reupload_scaling (ReuploadScaling): The scaling strategy for reuploads.

    Returns:
        The Rydberg feature map.

    Raises:
        NotImplementedError: If the reupload scaling strategy is not implemented.
            Only `ReuploadScaling.CONSTANT` and `ReuploadScaling.TOWER` are supported.
    """
    if reupload_scaling == ReuploadScaling.CONSTANT:
        return rydberg_feature_map(n_qubits=num_qubits, param=param)

    elif reupload_scaling == ReuploadScaling.TOWER:
        return rydberg_tower_feature_map(n_qubits=num_qubits, param=param)

    else:
        raise NotImplementedError(f"Rydberg feature map not implemented for {reupload_scaling}")


def _create_rydberg_fm(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Create a Rydberg feature map for the given configuration.

    Args:
        register (int | Register): The number of qubits or the register to apply the feature map to.
        config (FeatureMapConfig): The configuration for the feature map.

    Returns:
        list: A list of Rydberg feature map blocks.
    """
    num_qubits = register if isinstance(register, int) else register.n_qubits

    num_uploads = {key: value + 1 for key, value in config.num_repeats.items()}  # type: ignore[union-attr]

    fm_blocks = []

    for i in range(max(num_uploads.values())):
        for feature in config.inputs:  # type: ignore[union-attr]
            if num_uploads[feature] > 0:
                fm_blocks.append(
                    _encode_feature_rydberg(
                        num_qubits=num_qubits,
                        param=feature,
                        reupload_scaling=config.reupload_scaling[feature],  # type: ignore[arg-type, index]
                    )
                )
            num_uploads[feature] -= 1

    return fm_blocks


def create_fm_blocks(
    register: int | Register,
    config: FeatureMapConfig,
) -> list[AbstractBlock]:
    """
    Create a list of feature map blocks based on the given configuration.

    In case of series encoding or even parallel encoding with data reuploads,
    the outputs is a list of blocks that still need to be interleaved with non
    commuting blocks.

    Args:
        register (int | Register): The number of qubits or the register.
        config (FeatureMapConfig): The configuration for the feature map.

    Returns:
        list[AbstractBlock]: A list of feature map blocks.

    Raises:
        ValueError: If the feature map strategy is not `Strategy.DIGITAL`, `Strategy.ANALOG` or
            `Strategy.RYDBERG`.
    """
    if config.feature_map_strategy == Strategy.DIGITAL:
        return _create_digital_fm(register=register, config=config)
    elif config.feature_map_strategy == Strategy.ANALOG:
        return _create_analog_fm(register=register, config=config)
    elif config.feature_map_strategy == Strategy.RYDBERG:
        return _create_rydberg_fm(register=register, config=config)
    else:
        raise NotImplementedError(
            f"Feature map not implemented for strategy {config.feature_map_strategy}. \
            Only `Strategy.DIGITAL`, `Strategy.ANALOG` or `Strategy.RYDBERG` allowed."
        )


def _ansatz_layer(
    register: int | Register,
    ansatz_config: AnsatzConfig,
    index: int,
) -> AbstractBlock:
    """
    Create a layer of the ansatz based on the configuration.

    Args:
        register (int | Register): The number of qubits or the register.
        ansatz_config (AnsatzConfig): The configuration for the ansatz.
        index (int): The index of the layer.

    Returns:
        AbstractBlock: The layer of the ansatz.
    """
    new_config = AnsatzConfig(
        depth=1,
        ansatz_type=ansatz_config.ansatz_type,
        ansatz_strategy=ansatz_config.ansatz_strategy,
        strategy_args=ansatz_config.strategy_args,
        param_prefix=f"fm_{index}",
    )

    return create_ansatz(register=register, config=new_config)


def _create_iia_digital(
    num_qubits: int,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Digital Identity Initialized Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Identity Initialized Ansatz.
    """
    operations = config.strategy_args.get("operations", [RX, RY])
    entangler = config.strategy_args.get("entangler", CNOT)
    periodic = config.strategy_args.get("periodic", False)

    return iia(
        n_qubits=num_qubits,
        depth=config.depth,
        param_prefix=config.param_prefix,
        strategy=Strategy.DIGITAL,
        rotations=operations,
        entangler=entangler,
        periodic=periodic,
    )


def _create_iia_sdaqc(
    num_qubits: int,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the SDAQC Identity Initialized Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The SDAQC Identity Initialized Ansatz.
    """
    operations = config.strategy_args.get("operations", [RX, RY])
    entangler = config.strategy_args.get("entangler", CNOT)
    periodic = config.strategy_args.get("periodic", False)

    return iia(
        n_qubits=num_qubits,
        depth=config.depth,
        param_prefix=config.param_prefix,
        strategy=Strategy.SDAQC,
        rotations=operations,
        entangler=entangler,
        periodic=periodic,
    )


def _create_iia(
    num_qubits: int,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Identity Initialized Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Identity Initialized Ansatz.

    Raises:
        ValueError: If the ansatz strategy is not supported. Only `Strategy.DIGITAL` and
            `Strategy.SDAQC` are allowed.
    """
    if config.ansatz_strategy == Strategy.DIGITAL:
        return _create_iia_digital(num_qubits=num_qubits, config=config)
    elif config.ansatz_strategy == Strategy.SDAQC:
        return _create_iia_sdaqc(num_qubits=num_qubits, config=config)
    else:
        raise ValueError(
            f"Invalid ansatz strategy {config.ansatz_strategy} provided. Only `Strategy.DIGITAL` \
                and `Strategy.SDAQC` allowed for IIA."
        )


def _create_hea_digital(num_qubits: int, config: AnsatzConfig) -> AbstractBlock:
    """
    Create the Digital Hardware Efficient Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Digital Hardware Efficient Ansatz.
    """
    operations = config.strategy_args.get("operations", [RX, RY, RX])
    entangler = config.strategy_args.get("entangler", CNOT)
    periodic = config.strategy_args.get("periodic", False)

    return hea_digital(
        n_qubits=num_qubits,
        depth=config.depth,
        param_prefix=config.param_prefix,
        operations=operations,
        entangler=entangler,
        periodic=periodic,
    )


def _create_hea_sdaqc(num_qubits: int, config: AnsatzConfig) -> AbstractBlock:
    """
    Create the SDAQC Hardware Efficient Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The SDAQC Hardware Efficient Ansatz.
    """
    operations = config.strategy_args.get("operations", [RX, RY, RX])
    entangler = config.strategy_args.get(
        "entangler", hamiltonian_factory(num_qubits, interaction=Interaction.NN)
    )

    return hea_sDAQC(
        n_qubits=num_qubits,
        depth=config.depth,
        param_prefix=config.param_prefix,
        operations=operations,
        entangler=entangler,
    )


def _create_hea_rydberg(
    register: int | Register,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Rydberg Hardware Efficient Ansatz based on the configuration.

    Args:
        register (int | Register): The number of qubits or the register object.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Rydberg Hardware Efficient Ansatz.
    """
    register = register if isinstance(register, Register) else Register.circle(n_qubits=register)

    addressable_detuning = config.strategy_args.get("addressable_detuning", True)
    addressable_drive = config.strategy_args.get("addressable_drive", False)
    tunable_phase = config.strategy_args.get("tunable_phase", False)

    return rydberg_hea(
        register=register,
        n_layers=config.depth,
        addressable_detuning=addressable_detuning,
        addressable_drive=addressable_drive,
        tunable_phase=tunable_phase,
        additional_prefix=config.param_prefix,
    )


def _create_hea(
    register: int | Register,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Hardware Efficient Ansatz based on the configuration.

    Args:
        register (int | Register): The number of qubits or the register to create the ansatz for.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The hardware efficient ansatz block.

    Raises:
        ValueError: If the ansatz strategy is not `Strategy.DIGITAL`, `Strategy.SDAQC`, or
            `Strategy.RYDBERG`.
    """
    num_qubits = register if isinstance(register, int) else register.n_qubits

    if config.ansatz_strategy == Strategy.DIGITAL:
        return _create_hea_digital(num_qubits=num_qubits, config=config)
    elif config.ansatz_strategy == Strategy.SDAQC:
        return _create_hea_sdaqc(num_qubits=num_qubits, config=config)
    elif config.ansatz_strategy == Strategy.RYDBERG:
        return _create_hea_rydberg(register=register, config=config)
    else:
        raise ValueError(
            f"Invalid ansatz strategy {config.ansatz_strategy} provided. Only `Strategy.DIGITAL`, \
                `Strategy.SDAQC`, and `Strategy.RYDBERG` allowed"
        )


def _create_ala_digital(
    num_qubits: int,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Digital Alternating Layer Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Digital Alternating Layer Ansatz.
    """
    operations = config.strategy_args.get("operation", [RX, RY, RX])
    entangler = config.strategy_args.get("entangler", CNOT)

    return ala_digital(
        n_qubits=num_qubits,
        m_block_qubits=config.m_block_qubits,  # type: ignore[arg-type]
        param_prefix=config.param_prefix,
        operations=operations,
        entangler=entangler,
    )


def _create_ala(
    num_qubits: int,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the Alternating Layer Ansatz based on the configuration.

    Args:
        num_qubits (int): The number of qubits.
        config (AnsatzConfig): The configuration for the ansatz.

    Returns:
        AbstractBlock: The Alternating Layer Ansatz.

    Raises:
        ValueError: If the ansatz strategy is not `Strategy.DIGITAL`.
    """
    if config.ansatz_strategy == Strategy.DIGITAL:
        return _create_ala_digital(num_qubits=num_qubits, config=config)
    else:
        raise ValueError(
            f"Invalid ansatz strategy {config.ansatz_strategy} provided. Only `Strategy.DIGITAL` \
                allowed"
        )


def create_ansatz(
    register: int | Register,
    config: AnsatzConfig,
) -> AbstractBlock:
    """
    Create the ansatz based on the configuration.

    Args:
        register (int | Register): Number of qubits or a register object.
        config (AnsatzConfig): Configuration for the ansatz.

    Returns:
        AbstractBlock: The ansatz block.

    Raises:
        NotImplementedError: If the ansatz type is not implemented.
    """
    num_qubits = register if isinstance(register, int) else register.n_qubits

    if config.ansatz_type == AnsatzType.IIA:
        return _create_iia(num_qubits=num_qubits, config=config)
    elif config.ansatz_type == AnsatzType.HEA:
        return _create_hea(register=register, config=config)
    elif config.ansatz_type == AnsatzType.ALA:
        return _create_ala(num_qubits=num_qubits, config=config)
    else:
        raise NotImplementedError(
            f"Ansatz of type {config.ansatz_type} not implemented yet. Only `AnsatzType.HEA` and\
                `AnsatzType.IIA` available."
        )


def _interleave_ansatz_in_fm(
    register: int | Register,
    fm_blocks: list[AbstractBlock],
    ansatz_config: AnsatzConfig,
) -> ChainBlock:
    """
    Interleave the ansatz layers in between the feature map layers.

    Args:
        register (int | Register): Number of qubits or a register object.
        fm_blocks (list[AbstractBlock]): List of feature map blocks.
        ansatz_config (AnsatzConfig): Ansatz configuration.

    Returns:
        ChainBlock: A block containing feature map layers with interleaved ansatz layers.
    """
    full_fm = []
    for idx, block in enumerate(fm_blocks):
        full_fm.append(block)
        if idx + 1 < len(fm_blocks):
            full_fm.append(_ansatz_layer(register, ansatz_config, idx))

    return chain(*full_fm)


def _global_identity(register: int | Register) -> KronBlock:
    """Create a global identity block."""
    return kron(
        *[I(i) for i in range(register if isinstance(register, int) else register.n_qubits)]
    )


def create_observable(
    register: int | Register,
    config: ObservableConfig,
) -> AbstractBlock:
    """
    Create an observable block.

    Args:
        register (int | Register): Number of qubits or a register object.
        config (ObservableConfig): Observable configuration.

    Returns:
        AbstractBlock: The observable block.
    """
    shifting_term: AbstractBlock = config.shift * _global_identity(register)  # type: ignore[operator]
    detuning_hamiltonian: AbstractBlock = config.scale * hamiltonian_factory(  # type: ignore[operator]
        register=register,
        interaction=config.interaction,
        detuning=config.detuning,
    )
    obs: AbstractBlock = add(shifting_term, detuning_hamiltonian)

    if isinstance(config.tag, str):
        tag(obs, config.tag)

    return obs


def build_qnn_from_configs(
    register: int | Register,
    observable_config: ObservableConfig | list[ObservableConfig],
    fm_config: FeatureMapConfig = FeatureMapConfig(),
    ansatz_config: AnsatzConfig = AnsatzConfig(),
    backend: BackendName = BackendName.PYQTORCH,
    diff_mode: DiffMode = DiffMode.AD,
    measurement: Measurements | None = None,
    noise: NoiseHandler | None = None,
    configuration: BackendConfiguration | dict | None = None,
    input_diff_mode: InputDiffMode | str = InputDiffMode.AD,
) -> QNN:
    """
    Build a QNN model.

    Args:
        register (int | Register): Number of qubits or a register object.
        observable_config (ObservableConfig | list[ObservableConfig]): Observable configuration(s).
        fm_config (FeatureMapConfig): Feature map configuration.
        ansatz_config (AnsatzConfig): Ansatz configuration.
        backend (BackendName): The chosen quantum backend.
        diff_mode (DiffMode): The differentiation engine to use. Choices are
            'gpsr' or 'ad'.
        measurement (Measurements): Optional measurement protocol. If None,
            use exact expectation value with a statevector simulator.
        noise (Noise): A noise model to use.
        configuration (BackendConfiguration | dict): Optional backend configuration.
        input_diff_mode (InputDiffMode): The differentiation mode for the input tensor.

    Returns:
        QNN: A QNN model.
    """
    blocks: list[AbstractBlock] = []
    inputs: list[Basic | str] | None = None

    if fm_config.num_features > 0:
        fm_blocks = create_fm_blocks(register=register, config=fm_config)
        full_fm = _interleave_ansatz_in_fm(
            register=register,
            fm_blocks=fm_blocks,
            ansatz_config=ansatz_config,
        )
        if isinstance(fm_config.tag, str):
            tag(full_fm, fm_config.tag)
        inputs = fm_config.inputs
        blocks.append(full_fm)

    ansatz = create_ansatz(register=register, config=ansatz_config)
    if isinstance(ansatz_config.tag, str):
        tag(ansatz, ansatz_config.tag)
    blocks.append(ansatz)

    circ = QuantumCircuit(register, *blocks)

    observable: AbstractBlock | list[AbstractBlock] = (
        [create_observable(register=register, config=cfg) for cfg in observable_config]
        if isinstance(observable_config, list)
        else create_observable(register=register, config=observable_config)
    )

    ufa = QNN(
        circ,
        observable,
        inputs=inputs,
        backend=backend,
        diff_mode=diff_mode,
        measurement=measurement,
        noise=noise,
        configuration=configuration,
        input_diff_mode=input_diff_mode,
    )

    return ufa


def _create_feature_map_qcnn(
    n_qubits: int,
    n_inputs: int,
    fm_type: str = "Fourier",
    op: Any = RX,
) -> Any:
    """
    Creates a feature map (FM) by dividing qubits among inputs and applying.

    the specified feature map type.

    Args:
        n_qubits (int): Total number of qubits.
        n_inputs (int): Number of inputs.
        fm_type (str): Type of feature map to use (e.g., "Fourier").
        op (Any): Quantum operation to use in the feature map (e.g., RX).

    Returns:
        Any: The combined feature map as a kronecker product
        of individual feature maps.
    """
    fm_temp = []
    qubits_per_input = n_qubits // n_inputs  # Base number of qubits per input
    exceeding_qubits = n_qubits % n_inputs  # Number of exceeding qubits
    start = 0  # Track current qubit index

    for i in range(n_inputs):
        # Assign base qubits + 1 extra if input has exceeding qubits
        num_qubits = qubits_per_input + 1 if i < exceeding_qubits else qubits_per_input
        end = start + num_qubits

        # Create FM for this input
        fm_temp.append(
            feature_map(
                n_qubits=num_qubits,
                param=f"\u03C6_{i}",  # Use phi_i as the parameter
                op=op,
                fm_type=fm_type,
                support=tuple(range(start, end)),
            )
        )
        start = end  # Update starting index for next FM

    # Combine all feature maps using kronecker product
    return kron(*fm_temp)


def _get_block_params(
    params: dict,
    layer: int,
    rep: int,
    pos: int,
    is_corr: bool = False,
) -> Any:
    """
    Retrieves the parameter for a given operation.

    Args:
        params (dict): Dictionary to store and retrieve parameters.
        layer (int): The index of the current layer.
        rep (int): The index of the current repetition in the layer.
        pos (int): Position of the qubit in the layer.
        is_corr (bool): If True, uses correlated parameters for corresponding gates in W^opt_ij.

    Returns:
        Parameter: the retrieved parameter.
    """
    if is_corr:
        # Cycle pos from 0 to 8
        key = f"θ_{layer}_{pos % 9}"
    else:
        key = f"θ_{layer}_{rep}_{pos}"

    if key not in params:
        params[key] = Parameter(key)
    return params[key]


def _create_single_W(
    params: dict,
    operations: list[Any],
    entangler: Any,
    layer: int,
    rep: int,
    max_reps: int,
    control: int,
    target: int,
    spacing: int,
    n_qubits: int,
    is_corr: bool = False,
) -> ChainBlock:
    """Creates a single convolutional cell W_ij."""
    pad = [
        I(q)
        for q in range(control - spacing, control + spacing + 1)
        if q != control and q != target and 0 <= q < n_qubits
    ]
    gates = []

    # Track per-layer repetition index for proper parameter continuity
    key_param_counter = f"param_index_{layer}_{rep}"
    if key_param_counter not in params:
        params[key_param_counter] = 0  # Initialize if first time

    param_index = params[key_param_counter]  # Load index
    single_params = {}  # Store params for single RZ/RY gates

    # Apply the first sequence of operations
    for _, op in enumerate(operations):
        param_control = _get_block_params(params, layer, rep, param_index, is_corr)
        param_index += 1
        param_target = _get_block_params(params, layer, rep, param_index, is_corr)
        param_index += 1
        gates.append(
            kron(
                *pad,
                op(control, param_control),
                op(target, param_target),
            )
        )
    # entangling gate
    gates.append(entangler(target, control))

    # Apply RZ, RY and entangling gates for intermediate step
    single_params["control_rz"] = _get_block_params(params, layer, rep, param_index, is_corr)
    param_index += 1
    single_params["target_ry"] = _get_block_params(params, layer, rep, param_index, is_corr)
    param_index += 1
    gates.append(
        kron(
            *pad,
            RZ(control, single_params["control_rz"]),
            RY(target, single_params["target_ry"]),
        )
    )
    # entangling gate
    gates.append(entangler(control, target))

    intermediate_ry = _get_block_params(params, layer, rep, param_index, is_corr)
    param_index += 1
    gates.append(
        kron(
            *pad,
            I(control),
            RY(target, intermediate_ry),
        )
    )
    # entangling gate
    gates.append(entangler(target, control))

    # Apply the first sequence of operations
    for _, op in enumerate(operations):
        param_control = _get_block_params(params, layer, rep, param_index, is_corr)
        param_index += 1
        param_target = _get_block_params(params, layer, rep, param_index, is_corr)
        param_index += 1
        gates.append(
            kron(
                *pad,
                op(control, param_control),
                op(target, param_target),
            )
        )
    # Add final entangling gate (control -> target)
    if rep == int(max_reps - 1):
        gates.append(entangler(control, target))

    # Update params dict with the last used index
    params[key_param_counter] = param_index

    return chain(*gates)


def _create_conv_layer(
    layer_index: int,
    max_reps: int,
    current_indices: list[int],
    params: dict,
    operations: list[Any],
    entangler: Any,
    n_qubits: int,
    is_corr: bool,
) -> tuple[AbstractBlock, list[int]]:
    """
    Function to create a single convolutional layer.

    Args:
        layer_index (int): The index of the current layer.
        reps (int): Number of repetitions for this layer.
        current_indices (List[int]): Indices of qubits for the current layer.
        params (dict): Dictionary to store and retrieve parameters.
        operations (List[Any]): List of quantum operations to apply in the gates.
        entangler (Any): Entangling operation, such as CZ.
        n_qubits (int): Total number of qubits.

    Returns:
        Tuple[AbstractBlock, List[int]]: A tuple containing the quantum block
        for the layer and the target indices for the next layer.
    """
    current_layer = []
    next_indices = []
    spacing = layer_index

    if layer_index in [0, 1]:  # Special behavior for first two layers
        layer_reps = []
        for rep in range(max_reps):
            rep_kron = []
            # Define qubit pairs based on odd/even repetition
            if rep % 2 == 0:  # Even d: regular behavior
                pairs = zip(current_indices[::2], current_indices[1::2])
            else:  # Odd d: shift downward, leaving qubits 0 and 7 free
                pairs = zip(current_indices[1:-1:2], current_indices[2:-1:2])

            # Build the gate sequence for each pair
            for control, target in pairs:
                W_pairs = _create_single_W(
                    params,
                    operations,
                    entangler,
                    layer_index,
                    rep,
                    max_reps,
                    control,
                    target,
                    spacing,
                    n_qubits,
                    is_corr,
                )
                tag(W_pairs, f"W{control,target}")
                rep_kron.append(W_pairs)

            layer_reps.append(kron(*rep_kron))

        # Combine all repetitions using `chain`
        current_layer.append(chain(*layer_reps))

    else:  # Original behavior for other layers
        for rep in range(max_reps):
            for control, target in zip(current_indices[::2], current_indices[1::2]):
                W_pairs = _create_single_W(
                    params,
                    operations,
                    entangler,
                    layer_index,
                    rep,
                    max_reps,
                    control,
                    target,
                    spacing,
                    n_qubits,
                    is_corr,
                )
                current_layer.append(W_pairs)

    # Update `next_indices` with the **targets** of the current layer
    next_indices = current_indices[1::2]
    return chain(*current_layer), next_indices
