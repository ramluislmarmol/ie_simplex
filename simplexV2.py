import numpy as np
import pandas as pd
import sympy as sp

# Define big M
M = sp.Symbol('M')

# Functions starts here

def create_vector(index, size, type = 'slack'):
    if type == 'artificial':
        vector = np.array([[M]])
    else:
        vector = np.array([[0]])

    for i in range(0, size):
        if i == 0:
            continue
        if i == index:
            if type == 'surplus':
                vector = np.append(vector, [[-1]], axis = 0)
            else:
                vector = np.append(vector, [[1]], axis = 0)
        else:
            vector = np.append(vector, [[0]], axis = 0)
    return vector

def create_initial_tableau(A, eq, b):
    num_rows = A.shape[0]
    num_cols = A.shape[1]

    var_x = np.array([])
    for i in range(0, num_cols):
        var_string = f"x{i + 1}"
        var_x = np.append(var_x, [var_string])

    initial_tableau = A
    var_y = np.array([['z']])
    s = 1
    a = 1
    for i in range(0, num_rows):
        if eq[i] == 0:
            continue
        if eq[i] == 1:
            vector_to_append = create_vector(i, num_rows)
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"s{s}"])
            var_y = np.append(var_y, [f"s{s}"])
            s += 1 
        if eq[i] == 2:
            vector_to_append = create_vector(i, num_rows, type = 'artificial')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"a{a}"])
            var_y = np.append(var_y, [f"a{a}"])
            a += 1
        if eq[i] == 3:
            vector_to_append = create_vector(i, num_rows, type = 'artificial')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"a{a}"])
            var_y = np.append(var_y, [f"a{a}"])
            a += 1

            vector_to_append = create_vector(i, num_rows, type = 'surplus')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"s{s}"])
            var_y = np.append(var_y, [f"s{s}"])
            s += 1

    var_x = np.append(var_x, ['RHS'])
    initial_tableau = np.append(initial_tableau, b, axis = 1)

    return {'initial_tableau' : initial_tableau, 'var_x' : var_x, 'var_y' : var_y}

def remove_M_in_obj(tableau, var_y):
    if M not in tableau[0]:
        return tableau

    target_cols = []
    for i in range(0, len(var_y)):
        if 'a' in var_y[i]:
            target_cols.insert(0, i)

    for idx in target_cols:
        row_to_add = tableau[idx] * -1 * M
        tableau[0] = tableau[0] + row_to_add

    return tableau

def numerize_tableau(tableau):
    M_value = 1e9
    tableau_copy = tableau.copy()
    try:
        for i in range(0, len(tableau_copy[0])):
            expr = tableau_copy[0][i]
            tableau_copy[0][i] = expr.subs(M, M_value)
        return tableau_copy
    except:
        return tableau_copy

def select_pivot_col_idx(tableau):
    pivot_val = 0
    pivot_col_idx = 0
    for i in range(0, len(tableau[0]) - 1):
        if tableau[0][i] < pivot_val:
            pivot_val = tableau[0][i]
            pivot_col_idx = i
    return pivot_col_idx

def select_pivot_row_idx(tableau, pivot_col_idx):
    pivot_val = 1e9
    pivot_row_idx = 0
    for i in range(0, len(tableau)):
        if i == 0:
            continue 

        current_RHS = tableau[i][-1]
        current_col_element = tableau[i][pivot_col_idx]

        if current_col_element <= 0:
            continue

        current_val = current_RHS / current_col_element
        if current_val < pivot_val:
            pivot_val = current_val
            pivot_row_idx = i
    return pivot_row_idx

def normalize_cols(tableau, pivot_col_idx, pivot_row_idx):
    for i in range(0, len(tableau)):
        if i == pivot_row_idx:
            continue
        if tableau[i][pivot_col_idx] == 0:
            continue
        coeff = tableau[i][pivot_col_idx] / tableau[pivot_row_idx][pivot_col_idx]
        new_row = (-1 * coeff * tableau[pivot_row_idx]) + tableau[i]
        tableau[i] = new_row
    return tableau

def check(tableau):
    for element in tableau[0]:
        if element < 0:
            return True
    return False

def print_tableau(tableau, var_x, var_y):
    tableau_df = pd.DataFrame(tableau, index = var_y, columns = var_x)
    sp.pprint(tableau_df)

# Main Loop starts here

A = np.array([
    [ -3, -5 ],
    [  1,  0 ],
    [  0,  2 ],
    [  3,  2 ]
])

eq = np.array([
    [ 0 ],
    [ 1 ],
    [ 1 ],
    [ 2 ]
])

# 0 is for objective functions
# 1 is <=
# 2 is =
# 3 is >=

b = np.array([
    [  0 ],
    [  4 ],
    [ 12 ],
    [ 18 ]
])

creation_result = create_initial_tableau(A, eq, b)
var_x = creation_result['var_x']
var_y = creation_result['var_y']
tableau = creation_result['initial_tableau']

print("[Initial Tableau]")
print_tableau(tableau, var_x, var_y)
print(" > Removing Big M...")
tableau = remove_M_in_obj(tableau, var_y)
print()

iteration = 0
while (check(numerize_tableau(tableau))):
    print(f"[Iteration {iteration}]")
    print_tableau(tableau, var_x, var_y)

    pivot_col_idx = select_pivot_col_idx(numerize_tableau(tableau))
    pivot_row_idx = select_pivot_row_idx(numerize_tableau(tableau), pivot_col_idx)

    print(f" > Entering: {var_x[pivot_col_idx]}")
    print(f" >  Leaving: {var_y[pivot_row_idx]}")    

    var_y[pivot_row_idx] = var_x[pivot_col_idx]

    pivot_element = tableau[pivot_row_idx][pivot_col_idx]
    pivot_row = tableau[pivot_row_idx]
    pivot_row = (1 / pivot_element) * pivot_row
    tableau[pivot_row_idx] = pivot_row

    tableau = normalize_cols(tableau, pivot_col_idx, pivot_row_idx)

    if check(numerize_tableau(tableau)):
        print()
        iteration = iteration + 1
    else:
        print()
        print(f"[Final Tableau]")
        print_tableau(tableau, var_x, var_y)
        print()
        print(f"[Final Answers]")
        for i in range(0, len(var_y)):
            print(f" > {var_y[i]} = {tableau[i][-1]}")
        break