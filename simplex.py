import numpy as np        
np.set_printoptions(suppress = True, precision = 4)

def check(tableau):
    # Return true if there is a negative
    for element in tableau[0]:
        if element < 0:
            return True
    # Return false if none
    return False

def select_pivot_col_idx(tableau):
    # Find index of pivot col
    pivot_val = 0
    pivot_col_idx = 0
    for i in range(0, len(tableau[0])):
        if tableau[0][i] < pivot_val:
            pivot_val = tableau[0][i]
            pivot_col_idx = i
    # Return pivot col
    return pivot_col_idx

def select_pivot_row_idx(tableau, pivot_col_idx):
    # Find the row with
    pivot_val = 100000
    pivot_row_idx = 0
    for i in range(0, len(tableau)):
        if i == 0:
            continue 
        current_val = tableau[i][-1] / tableau[i][pivot_col_idx]
        if current_val < pivot_val:
            pivot_val = current_val
            pivot_row_idx = i
    # Return pivot row
    return pivot_row_idx

def normalize_cols(tableau, pivot_col_idx, pivot_row_idx):
    # Iterate over all rows
    for i in range(0, len(tableau)):
        # If row is pivot row, skip
        if i == pivot_row_idx:
            continue
        # If row is obj function, skip
        if tableau[i][pivot_col_idx] == 0:
            continue
        # Get coeff by dividing pivot col value by pivot val
        coeff = tableau[i][pivot_col_idx] / tableau[pivot_row_idx][pivot_col_idx]
        # Get new row by multiplying coeff to pivot row and add to current row
        new_row = (-1 * coeff * tableau[pivot_row_idx]) + tableau[i]
        # Put in tableau
        tableau[i] = new_row
    # Return new tableau
    return tableau

tableau = np.array([
    [  -40,  -30,    0,    0,    0  ],
    [   1,     1,    1,    0,   12  ],
    [   2,     1,    0,    1,   16  ]
], dtype=float)

iteration = 0

while (True):
    # Setup tableau
    print(f'=====Iteration {iteration}=====')
    print(tableau)

    # Select pivots
    pivot_col_idx = select_pivot_col_idx(tableau)
    print(f'  > Pivot Column is {pivot_col_idx + 1}')
    pivot_row_idx = select_pivot_row_idx(tableau, pivot_col_idx)
    print(f'  > Pivot Row is {pivot_row_idx + 1}')

    # Turn pivot element to 1
    pivot_element = tableau[pivot_row_idx][pivot_col_idx]
    pivot_row = tableau[pivot_row_idx]
    pivot_row = (1 / pivot_element) * pivot_row
    tableau[pivot_row_idx] = pivot_row

    # Normalize the cols
    tableau = normalize_cols(tableau, pivot_col_idx, pivot_row_idx)

    if check(tableau):
        print()
        iteration = iteration + 1
    else:
        print(f'\n=====Final Tableau=====')
        print(tableau)
        break