# QCNN model

## Introduction
In this tutorial, we’ll explore how to train a Quantum Convolutional Neural Network (QCNN) using Qadence, demonstrating its application on a simple yet insightful data structure. The tutorial begins by detailing the construction of the quantum circuit and the synthetic data generation process. Convolutional architectures excel at capturing spatial relationships in grid-like structures, so we design our data to emulate a system of five interconnected grid elements, where the collective behavior produces a single continuous output value. The key idea is to aggregate features from four neighboring elements into a fifth central node, generating a compact embedding that encodes structural information. By training the QCNN to predict the global property from this localized representation, we effectively enable the model to infer system-wide behavior from a single node’s contextual features—showcasing the potential of quantum machine learning for relational reasoning tasks.

## QCNN circuitry
A generic QCNN is described as
$\Phi_{{\theta},{\lambda}} =\bigcirc_{l=1}^{L} \big(\text{P}_{l}^{{\lambda}_{l}} \circ \text{C}_{l}^{{\theta}_{l}}\big)$

where $\circ$ denotes a single function composition, and $\bigcirc_{l=1}^L$ represents the composition of $L$ functions applied sequentially. Each QGCN layer $l$ comprises a quatnum convolutional layer $\text{C}_l^{{\theta}_{l}}$ followed by a quantum pooling layer $\text{P}_{l}^{{\lambda}_{l}}$, with ${\theta}_{l}$ and ${\lambda}_{l}$ being the convolution and pooling parameters, respectively. The alternating structure of the QGNN circuit processes and simplifies the input quantum state, starting with $\text{C}_1$ and ending with $\text{P}_{L}$.

The convolutional layer $\text{C}_l^{{\theta}_l}: \mathcal{S}(\mathcal{H}_l) \rightarrow \mathcal{S}(\mathcal{H}_l)$ preserves the size of the quantum register. It reads
$\text{C}_l^{{\theta}_l}(\cdot) = \bigcirc_{j=1}^{r}  \left(\bigotimes_{i \in \text{S}(j)} W_l^{(i, i+1)}\left({\theta}_l\right)\right)(\cdot)$

Each convolutional layer acts on an even number of qubits, since the unitary $W_l$[^1] convolves a pairs of neighboring qubits $(i, i+1)$.  In general, the operator $W$ acts on pairs of adjacent qubits defined by the set $S(j) = \{(i, i+1) \mid i \equiv j \!\!\!\mod 2, \ 0 \leq i \leq N-2\}$, where $N$ denotes the total number of qubits[^2]. This construction ensures an alternating nearest-neighbor interaction pattern. The alternation is determined by the parity of $j$: for even $j$, $W$ acts on even-indexed pairs $(0,1)$, $(2,3)$, etc.; similarly, for odd values of $j$, the same operation is performed on odd-indexed pairs in an analogous manner. Each convolutional layer $\text{C}_l^{{\theta}_l}$ has an associated parameter $r$, representing its depth. For example, in a two-layer QGCN architecture, the depths of $\text{C}_1$ and $\text{C}_2$ are denoted as $r_1$ and $r_2$, respectively.

The pooling layer $\text{P}_{l}^{{\lambda}_l}: \mathcal{S}(\mathcal{H}_l) \rightarrow \mathcal{S}(\mathcal{H}_{l+1})$ reduces the size of the quantum register by tracing out specific qubits, such that $\dim(\mathcal{H}_{l+1}) < \dim(\mathcal{H}_l)$, and is defined as
$\text{P}_{l}^{{\lambda}_l}(\cdot) = Tr_{i}[(\cdot)]$

where the $i$-qubit is traced out in the $l$-th layer. The pooling layers typically discard half of the qubits at each step.

We adopt a simple architecture for $\text{C}_l^{{\theta}_l}$ and $\text{P}_{l}^{{\lambda}_l}$. The unitary $W$ is defined as in Ref. [vatan_2004_optimal](https://arxiv.org/abs/quant-ph/0308006), where the $A$ gates are defined similarly in terms of $R_G({\theta}_l) = e^{-iX\theta^1_{l}/2}e^{-iZ\theta^2_{l}/2}e^{-iX\theta^3_{l}/2}$. Entanglement is achieved applying non-parametrized CZ gates. In the pooling layers, entanglement is followed by local measurements on $\mathcal{O}$, enabled by the deferred measurement principle as in Ref. [Nielsen](https://books.google.com.br/books?hl=en&lr=&id=-s4DEy7o-a0C&oi=fnd&pg=PR17&ots=NJ5JfksuWw&sig=Un_kdl0BZ-eJzwRuS2JuoGY1KYI&redir_esc=y#v=onepage&q&f=false). The chosen design reduces the complexity of the QGCN circuit by
$\Phi_{{\theta}, {\lambda}} \to \Phi_{{\theta}}$

A schematic illustration of the full circuit can be found in [our work on QGNNs](https://www.arxiv.org/abs/2503.24111) which shares a similar circuit design


## Dummy Data Description

1. 4 grid elements (each with `n_qubits` features) influence a 5th element
2. The relationship is encoded in a single continuous target value

#### Input and Target Tensors (`dummy_input` and `dummy_target`)
1. Input
    - Shape: `(4, n_qubits)`: 4 samples representing 4 grid elements with `n_qubits` features per grid element. Random values between [0, 1)
2. Target
    - Shape: `(1, 1)`: Single scalar value representing the aggregated property of the 5th connected grid element. Random value between [0, 1)

```python exec="on" source="material-block" session="qcnn"
import torch
n_features = 8
dummy_input = torch.rand(4, n_features).float()
dummy_target = torch.rand(1, 1).float()
```

## Training a QCNN

Now we perform the training of the QCNN on the dummy data generated earlier.

### Define the QCNN circuit with Qadence
First we define a class with mean pooling on the QCNN output to match target value
```python exec="on" source="material-block" session="qcnn"
class qcnn_msg_passing(torch.nn.Module):
    def __init__(self, qcnn: torch.nn.Module, output_size: int = 1):
        super(qcnn_msg_passing, self).__init__()
        self.qcnn = qcnn
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.qcnn(x).float().view(-1, self.output_size)
        return torch.mean(x, dim=0, keepdim=True)
```

The model is hence defined as
```python exec="on" source="material-block" session="qcnn"
# QCNN model
from qadence import RX, RZ, CZ, QCNN

n_qubits = n_features
hidden_depth = [1,1,1]
operations = [RX, RZ, RX]
entangler = CZ
random_meas = True

# qgcn_circuit = QGCN(
qcnn_circuit = QCNN(
    n_inputs=n_qubits,
    n_qubits=n_qubits,
    depth=hidden_depth,
    operations=operations,
    entangler=entangler,
    random_meas=random_meas,
    is_corr=False,
)

model = qcnn_msg_passing(qcnn_circuit, dummy_target.shape[1])
```

### Training loop:

The training is performed as follows:

```python exec="on" source="material-block" session="qcnn"
n_epochs = 100
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(n_epochs + 1):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch={epoch:>4d} | Loss={loss.item():>10.6f}")
```

[^1]: We also refer to $W$ as convolutional cell.
[^2]: Qubit indexing is 0-based, i.e., $0, 1, \dots, N-1$.
