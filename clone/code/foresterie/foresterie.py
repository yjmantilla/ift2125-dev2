  
# Nom(s) étudiant(s) / Name(s) of student(s):

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :



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