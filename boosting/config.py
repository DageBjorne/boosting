RESPONSE = 'diameter'  #''grundyta'
TARGET = 'N.Norrland'

#resultat-fil res_{TARGET}_{RESPONSE}_{IDX}
## transfer config
v1s = [0.005, 0.007, 0.01, 0.03, 0.07]
target_tree_size_list = [1, 2, 3, 4]
source_tree_size_list = [1, 2, 3, 4]
decay_factor_list = [0.97, 0.99, 0.993, 0.995, 0.997]
train_size_list = [0.1, 0.25, 0.5, 0.75, 0.99]
test_size_list = [0.25]
epoch_list = [400, 600]
alpha_0_list = [1.0]
test_seed_list = [1]
train_seed_list = [1]

## non-transfer config
