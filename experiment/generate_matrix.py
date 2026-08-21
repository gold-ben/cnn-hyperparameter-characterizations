import itertools
import numpy as np

### blocks
# block 1: dataset (cifar10 vs cifar100)
# block 2: optimizer

### settings
# batch size, a, 1
# dropout, b, 2
# batch norm, c, 3
# max pooling, d, 4
# initialization, f, 5
# cnn width, g, 6
# convolutional layer dimensions, h, 7
# fully connected layer width, i, 8
# fully connected layer dimensions, j, 9

### defining relation
# 6 = 2345 -> g = bcdf
# 7 = 1345 -> h = acdf
# 9 = 348  -> j = cdi

### variables with center points
# batch size
# cnn width
# convolutional layer dimensions
# fully connected layer width
# fully connected layer dimensions

# Independent columns are a, b, c, d, f, and i. Generated columns are
# g=bcdf, h=acdf, and j=cdi. Product order is fixed for stable run IDs.
factorial_rows = []
for a, b, c, d, f, i in itertools.product([-1, 1], repeat=6):
    g = b * c * d * f
    h = a * c * d * f
    j = c * d * i
    factorial_rows.append([a, b, c, d, f, g, h, i, j])

design = np.array(factorial_rows)

# Indices for factors with center points (a, g, h, i, j)
center_indices = {0, 5, 6, 7, 8}
# Indices for factors without center points (b, c, d, f)
factorial_indices = [i for i in range(9) if i not in center_indices]

all_center_runs = []

# Generate all 2^4 combinations for the non-center factors
for combo in itertools.product([-1, 1], repeat=len(factorial_indices)):
    row = [0] * 9
    for i, val in enumerate(combo):
        row[factorial_indices[i]] = val
    all_center_runs.append(row)

center_points = np.array(all_center_runs)

print(f"Generated {len(center_points)} center point combinations.")
design_with_centers = np.vstack([design, center_points])
print(f"Total runs after adding center points: {len(design_with_centers)}")
print(design_with_centers)

column_names = ['batch_size_coded', 'dropout_flag_coded',
                'bn_flag_coded', 'max_pool_flag_coded',
                'initialization_coded', 'cnn_width_coded',
                'conv_dim_list_coded', 'fc_width_coded',
                'fc_dim_list_coded']

with open('./experiment_list.csv', 'w') as f:
    # Write header row
    f.write(','.join(column_names) + '\n')
    # Write data rows
    for row in design_with_centers:
        f.write(','.join(map(str, row)) + '\n')
