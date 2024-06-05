from qadence.backends.pyqtorch.convert_ops import PyQComposedBlock
from pyqtorch.primitive import X,Y,Z,CZ,CX,CY 
from pyqtorch.utils import product_state
from pyqtorch.apply import apply_operator
from torch import equal
from itertools import product
from functools import reduce
import pytest

# def test_PyQComposedBlock():
#     values = None
#     ops = [CX(target=0,control=1),CX(target=1,control=0),X(0),X(2),CX(target=0,control=2),CY(target=0,control=2)]
#     composed_block = PyQComposedBlock( ops = ops, qubits=[0,1,2],n_qubits=3)

#     qubits_strings = [str(i)+str(j)+str(k) for i in range(2) for j in range(2) for k in range(2)]

#     for qubits_string in qubits_strings:
#         state = product_state(qubits_string)
        
#         composed_state = composed_block.forward(state=state,values=values)

#         state_ev=state
#         for op in ops:
#             state_ev = apply_operator(state_ev,op.unitary(values=values),op.qubit_support)
        
        
#         assert equal(composed_state,state_ev)


@pytest.mark.parametrize('ops,state',product([ [X(1),Y(1),Z(1),X(2),Y(2),Z(0)],[CX(target=0,control=1),CX(target=1,control=0),X(0),X(2),CX(target=0,control=2),CY(target=0,control=2)]],
                                             [product_state(str(i)+str(j)+str(k)) for i in range(2) for j in range(2) for k in range(2)]
                                             )
                        )

def test_PyQComposedBlock(ops,state):
    values = None # test for different value of parameter values
    qubits_list = list(set(reduce(lambda x,y: x+list(y), [list(op.qubit_support) for op in ops] )))
    composed_block = PyQComposedBlock( ops = ops, qubits=qubits_list,n_qubits=len(qubits_list))
    composed_state = composed_block.forward(state=state,values=values)
    state_ev=state
    for op in ops:
        state_ev = apply_operator(state_ev,op.unitary(values=values),op.qubit_support)
    
    assert equal(composed_state,state_ev)


