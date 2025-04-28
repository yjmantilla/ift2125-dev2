  
# Nom(s) étudiant(s) / Name(s) of student(s):
# Yorguin José Mantilla Ramos
# Matricule: 20253616

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :

def dp(i, forest, cost, memo):
    # i is the current index (tree index) in the forest
    # memo is a dictionary to store previously computed results

    # Base case: if we are at the end of the forest
    if i >= len(forest):
        return (0, [])  # Profit 0, empty list
    
    # If we have already computed the result for this index, return it
    if i in memo:
        return memo[i]
    
    # Cases where compute is needed #
    # For each tree, we have two options:
    # 1. Skip the current tree and move to the next one
    # 2. Cut the current tree and move to the tree after the next one (i + 2)
    # Note, we also maintain list of indices of trees cut

    # Option 1: Skip current tree
    skip_profit, skip_list = dp(i + 1, forest, cost, memo)
    
    # Option 2: Cut current tree
    cut_profit, cut_list = dp(i + 2, forest, cost, memo)
    cut_profit += (forest[i] - cost) # Add the profit from cutting the current tree
    
    if cut_profit > skip_profit:
        memo[i] = (cut_profit, [i] + cut_list)
    else:
        memo[i] = (skip_profit, skip_list)
    
    return memo[i]

DEBUG = False

if DEBUG:
    forest = [3,5,11,9,4]
    cost = 2
    memo = {}
    max_profit, chosen_indices = dp(0,forest,cost,memo)
    print(f"Max profit: {max_profit}, Chosen indices: {chosen_indices}")

# Fonction à compléter / function to complete:
def solve(cost, forest) :
    memo = {}
    max_profit, chosen_indices = dp(0,forest,cost,memo)
    return max_profit


# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            cost = int(lines[0].strip())  # cout d'exploitation pour couper un arbre
            forest = list(map(int, lines[1].split()))  # valeur de chaque arbre    

        return solve(cost, forest)
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python foresterie.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()