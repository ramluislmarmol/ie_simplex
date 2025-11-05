import numpy as np
import pandas as pd
import sympy as sp
from fractions import Fraction

# Define big M
M = sp.Symbol('M')

# Functions starts here

# --- [Function 1]: Returns a vector to be added based on equality symbols
def create_vector(index, size, type = 'slack'):
    # Check which type of value to append in objective function
    if type == 'artificial': # Returns M if we are adding artificial variable
        vector = np.array([[M]])
    else: # Otherwise return 0 
        vector = np.array([[0]])

    # Look through each row
    for i in range(0, size):
        if i == 0: # Skip row 1 (objective function)
            continue
        if i == index: # If row is the target row
            if type == 'surplus': # Return -1 if surplus
                vector = np.append(vector, [[-1]], axis = 0)
            else: # Otherwise return 1
                vector = np.append(vector, [[1]], axis = 0)
        else: # If row is not target row, just return 0
            vector = np.append(vector, [[0]], axis = 0)
    
    # Return the vector as column vector
    return vector

# --- [Function 2]: Creates the initial tableau, returns a dict containing tableau, var_x and var_y
def create_initial_tableau(A, eq, b):
    # Get the shape of the tableau for reference later
    num_rows = A.shape[0]
    num_cols = A.shape[1]

    # Initialize var_x
    var_x = np.array([])
    # Assign variables based on the amount of columns of the matrix
    for i in range(0, num_cols):
        var_string = f"x{i + 1}"
        var_x = np.append(var_x, [var_string])

    # Initialize initial tableau
    initial_tableau = A
    var_y = np.array([['z']])
    s = 1 # To track amount of slack / surplus variables
    a = 1 # To track amount of artificial variables
    # Loop through each row
    for i in range(0, num_rows):
        if eq[i] == 0: # Skip if equality is 0 (objective function)
            continue
        if eq[i] == 1: # For < case
            # Add slack variable
            vector_to_append = create_vector(i, num_rows)
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"s{s}"]) # Add variable to var_x
            var_y = np.append(var_y, [f"s{s}"]) # Add variable to var_y
            s += 1 
        if eq[i] == 2: # For = case
            # Add artificial variable
            vector_to_append = create_vector(i, num_rows, type = 'artificial')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"a{a}"]) # Add variable to var_x
            var_y = np.append(var_y, [f"a{a}"]) # Add variable to var_y
            a += 1
        if eq[i] == 3: # For >= case
            # Add artificial variable
            vector_to_append = create_vector(i, num_rows, type = 'artificial')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"a{a}"]) # Add variable to var_x
            var_y = np.append(var_y, [f"a{a}"]) # Add variable to var_y
            a += 1

            # Subtract surplus variable
            vector_to_append = create_vector(i, num_rows, type = 'surplus')
            initial_tableau = np.append(initial_tableau, vector_to_append, axis = 1)

            var_x = np.append(var_x, [f"s{s}"]) # Add variable to var_x, we don't add surplus to var_y
            s += 1

    var_x = np.append(var_x, ['RHS']) # Append RHS to var_x for header purposes
    initial_tableau = np.append(initial_tableau, b, axis = 1) # Append b to tableau for RHS

    # Return the dict
    return {'initial_tableau' : initial_tableau, 'var_x' : var_x, 'var_y' : var_y}

# --- [Function 3]: Remove big M in objective function
def remove_M_in_obj(tableau, var_y):
    if M not in tableau[0]: # If there is no big M to begin with, skip this entirely
        return tableau

    target_cols = [] # We will store the cols with M in this list
    # Loop through each element in var_y
    for i in range(0, len(var_y)):
        if 'a' in var_y[i]: # Check if the element contains the character 'a'
            target_cols.insert(0, i) # Add the index to the list

    # Loop through the indices in target_cols
    for idx in target_cols:
        row_to_add = tableau[idx] * -1 * M # We will add this row to obj function to cancel out the big M
        tableau[0] = tableau[0] + row_to_add # Cancel out the big M

    # Return the tableau
    return tableau

# --- [Function 4]: Remove sympy M to numerize the tableau
def numerize_tableau(tableau):
    M_value = 1e9
    tableau_copy = tableau.copy() # To not mutate the original tableau
    # Loop through each element in the objective function
    for i in range(0, len(tableau_copy[0])):
        expr = tableau_copy[0][i]
        if type(expr) == Fraction: # Skip non-sympy functions
            continue
        tableau_copy[0][i] = expr.subs(M, M_value) # Swap out the M sympy for a huge number (1e9)
    # Return the tableau
    return tableau_copy

# --- [Function 5]: Select the pivot column (lowest negative number)
def select_pivot_col_idx(tableau):
    # Initialize variables
    pivot_val = 0
    pivot_col_idx = 0

    # Loop through each element in objective function (except the RHS)
    for i in range(0, len(tableau[0]) - 1):
        if tableau[0][i] < pivot_val: # If this element is lower than last
            pivot_val = tableau[0][i] # Keep it as "lowest"
            pivot_col_idx = i # Store its index
        
    # Return pivot col
    return pivot_col_idx

# --- [Function 5]: Select the pivot row
def select_pivot_row_idx(tableau, pivot_col_idx):
    # Initialize variables
    pivot_val = 1e9 # Since we are finding the lowest pos num index
    pivot_row_idx = 0

    # Loop through each row
    for i in range(0, len(tableau)):
        if i == 0: # Skip row 1 (objective function)
            continue 

        current_RHS = tableau[i][-1] # RHS element of the current row
        current_col_element = tableau[i][pivot_col_idx] # Pivot col element of the current row

        if current_col_element <= 0 or current_RHS <= 0: # Skip if any of the element is 0 or negative
            continue

        current_val = current_RHS / current_col_element # The proposed pivot element
        if current_val < pivot_val: # If this element is lower than the last
            pivot_val = current_val # Keep it as "lowest"
            pivot_row_idx = i # Store its index
    
    # Return pivot row
    return pivot_row_idx

# --- [Function 6]: Normalizes the tableau
def update_tableau(tableau, pivot_col_idx, pivot_row_idx):
    # Step 1: Turn the pivot element to 1
    pivot_element = tableau[pivot_row_idx][pivot_col_idx]
    pivot_row = tableau[pivot_row_idx].copy()
    tableau[pivot_row_idx] = (Fraction(1) / pivot_element) * pivot_row

    # Step 2: Turn the rest of the column to 0
    # Loop through each row
    for i in range(0, len(tableau)):
        if i == pivot_row_idx: # Skip if row is pivot row
            continue
        if tableau[i][pivot_col_idx] == 0: # Skip if pivot row element is already 0
            continue
        
        numerator = tableau[i][pivot_col_idx] # The coefficient of the column element
        coeff = numerator / pivot_element # The number to be multiplied to one to cancel col ele
        tableau[i] = (-1 * coeff * pivot_row) + tableau[i] # Overwrite the column
    
    # Return the updated tableau
    return tableau

# --- [Function 7]: Validate if objective function has no more negatives
def check(tableau):
    # Loop through each element in objective function
    for element in tableau[0, 0:-1]:
        if element < 0: # If there is < 0, return True
            return True

    # Otherwise return false
    return False

# --- [Function 8]: Print the tableau as pandas cus it looks pretty
def print_tableau(tableau, var_x, var_y):
    tableau_df = pd.DataFrame(tableau, index = var_y, columns = var_x)
    sp.pprint(tableau_df)

# --- Main function that ties everything together
def optimize(A, eq, b):
    # Step 1: Initialize the tableau
    result = create_initial_tableau(A, eq, b)
    var_x = result['var_x']
    var_y = result['var_y']
    tableau = result['initial_tableau']

    # Step 1.5: Turn each element in the numpy array to Fraction operations
    tableau = tableau + Fraction()

    # Step 2: Print the Initial tableau
    print("[Initial Tableau]")
    print_tableau(tableau, var_x, var_y)

    # Step 3: Remove the big M from objective function 
    print(" > Removing Big M...\n")
    tableau = remove_M_in_obj(tableau, var_y)

    # Start iterations
    iteration = 0
    while (check(numerize_tableau(tableau))):
        print(f"[Iteration {iteration}]")
        print_tableau(tableau, var_x, var_y)

        pivot_col_idx = select_pivot_col_idx(numerize_tableau(tableau))
        pivot_row_idx = select_pivot_row_idx(numerize_tableau(tableau), pivot_col_idx)

        print(f" > Entering: {var_x[pivot_col_idx]}")
        print(f" >  Leaving: {var_y[pivot_row_idx]}\n")    

        var_y[pivot_row_idx] = var_x[pivot_col_idx]

        tableau = update_tableau(tableau, pivot_col_idx, pivot_row_idx)

        if check(numerize_tableau(tableau)):
            iteration = iteration + 1
        else:
            print(f"[Final Tableau]")
            print_tableau(tableau, var_x, var_y)
            print(f"\n[Final Answers]")
            for i in range(0, len(var_y)):
                print(f" > {var_y[i]} = {tableau[i][-1]}")
            break

# ====================== End of Module ======================
A = np.array([
    [ 3, 2, 7 ],
    [-1, 1, 0 ],
    [ 2,-1, 1 ]
])

eq = np.array([
    [ 0 ],
    [ 2 ],
    [ 3 ]
])

# 0 is for objective functions
# 1 is <=
# 2 is =
# 3 is >=

b = np.array([
    [  0 ],
    [ 10 ],
    [ 10 ]
])

optimize(A, eq, b)