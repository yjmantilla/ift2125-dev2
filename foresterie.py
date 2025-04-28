  
# Nom(s) étudiant(s) / Name(s) of student(s):

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :


def lis(array,start,end,memo):
    for i in range(start,end):
        #range(0,n):
        #range(len(array)-1,-1,-1):
        next_options = list(range(i+2, len(array))) #+2 to skip adjacent elements
        results =[]

        if next_options:
            for subproblem in next_options:
                if subproblem in memo:
                    results.append(memo[subproblem])
                else:
                    results.append(lis(array,subproblem,end,memo))
        else:
            results.append({'gain':array[i],'next_node':None})
        memo[start] = max(results, key=lambda x: x['gain'])
        return memo[i]


DEBUG = True

if DEBUG:
    forest = [3,5,11,9,4]
    cost = 2
    lis(forest,0,len(forest),{})

# Fonction à compléter / function to complete:
def solve(cost, forest) :
    return 


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