import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
import matplotlib.pyplot as plt
import copy
from federated.client import Client, ModelParams

# --- MOCK CLIENT SETUP ---
class DummyModel:
    def __init__(self, param_shapes):
        self._params = [np.random.randn(*shape) * 5 for shape in param_shapes]
    def get_parameters(self):
        return [p.copy() for p in self._params]
    def set_parameters(self, params):
        self._params = [p.copy() for p in params]

class DummyClient(Client):
    def __init__(self, param_shapes):
        self.client_id = 'test_client'
        self.num_samples = 100
        self.local_model = DummyModel(param_shapes)
        self.initial_round_params = [np.zeros(shape) for shape in param_shapes]

# --- TESTING CLIPPING ---
def test_clipping_effect():
    param_shapes = [(10,), (1,)]  # Example: 10 weights + 1 bias
    client = DummyClient(param_shapes)

    # Simulate a local update (random for demo)
    local_params = client.local_model.get_parameters()
    global_params = [np.zeros_like(p) for p in local_params]
    client.initial_round_params = copy.deepcopy(global_params)

    # Compute raw update
    raw_update = [local - global_ for local, global_ in zip(local_params, global_params)]
    raw_norm = np.sqrt(sum(np.sum(np.square(u)) for u in raw_update))

    # Try a range of clip_norm values
    clip_norms = np.linspace(0.1, raw_norm * 1.2, 30)
    norms = []
    for c in clip_norms:
        clipped_update, _ = client.get_update(clip_norm=c)
        norm = np.sqrt(sum(np.sum(np.square(u)) for u in clipped_update))
        norms.append(norm)

    plt.figure(figsize=(8, 5))
    plt.plot(clip_norms, norms, label='Clipped Update Norm')
    plt.axhline(raw_norm, color='r', linestyle='--', label=f'Raw Update Norm ({raw_norm:.2f})')
    plt.xlabel('Clip Norm (C_clip)')
    plt.ylabel('L2 Norm of Update')
    plt.title('Effect of L2 Norm Clipping on Client Update')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/results/clipping_effect.png')
    plt.show()

if __name__ == '__main__':
    test_clipping_effect()
