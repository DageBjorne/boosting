RESPONSE = 'Dgv'  #Hgv  #'Dgv'
TARGET = 'Lettland'  #Lettland

# predictor_columns
predictor_columns = [
    'zq5', 'zq10', 'zq15', 'zq20', 'zq25', 'zq30', 'zq35', 'zq40', 'zq45',
    'zq50', 'zq55', 'zq60', 'zq65', 'zq70', 'zq75', 'zq80', 'zq85', 'zq90',
    'zq95', 'pzabovezmean', 'pzabove2', 'zpcum1', 'zpcum2', 'zpcum3', 'zpcum4',
    'zpcum5', 'zpcum6', 'zpcum7', 'zpcum8', 'zpcum9'
]

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
vs = [0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1]
tree_size_list = [1, 2, 3, 4, 5, 6]
epoch_list_no_transfer = [200, 400, 600, 800]
