# %%

from eeyore.constants import loss_functions
from eeyore.models import mlp

# %%

# hparams = mlp.Hyperparameters(dims=[2, 3, 3, 2], bias=3*[True], activations=3*[None])
hparams = mlp.Hyperparameters(dims=[2, 3, 3, 2], bias=[True, True, True], activations=3*[None])

model = mlp.MLP(loss=loss_functions['multiclass_classification'], hparams=hparams)

# %%

print(model.num_par_blocks())

print([model.starting_par_block_idx(i) for i in [0, 1, 2]])

print(model.starting_par_block_indices())

for b in range(8):
    l, n = model.layer_and_node_from_par_block(b)
    print("Block {} is in layer {} and node {} of that layer".format(b, l, n))

for b in range(8):
    indices, l, n = model.par_block_indices(b)
    print("Block {} is in layer {} and node {} of that layer and has indices {}".format(b, l, n, indices))
