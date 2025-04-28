
# Nom(s) étudiant(s) / Name(s) of student(s):
# Yorguin José Mantilla Ramos
# Matricule: 20253616
import sys



# Espace pour fonctions auxillaires :
# Space for auxilary functions :
def postorder_balance(node):
    # Strategy:
    # Postorder traversal is ideal as a child node can only give (or receive) tokens to/from its parent.
    # If we did it in preorder, the parent would need to decide how many tokens to give to its children before knowing how many tokens they need.
    # Thus is better to first calculate the balance of the children.


    # Base case: if the node is None, it means we have no tokens to balance, so we return (0, 0)
    # (0, 0) means that the node is balanced and no moves are needed
    if not node:
        return (0, 0)  # (balance, moves)

    # We first process the left and right children
    left_balance, left_moves = postorder_balance(node.left)
    right_balance, right_moves = postorder_balance(node.right)
    
    # Balance of the current node as a parent is the sum of the balances of its children plus its own value
    # But the balance of a node is the number of tokens the node needs
    # to send to parent (positive) or receive from parent (negative)
    # to achieve the target of 1 token per node.
    # So we subtract 1 from the node's value to get the balance=0 when we achieve the target.
    balance = node.val - 1 + left_balance + right_balance

    # moves is the total number of moves needed to balance the subtree rooted at this node
    # as the moves dont depend on the direction of the balance, we can just take the absolute value
    # moreover, we carry the moves from the children to the parent,
    # so that it accounts for the whole subtree.
    moves = abs(balance) + left_moves + right_moves

    return (balance, moves)

# Fonction à compléter / function to complete:
def solve(root):
    _, moves = postorder_balance(root)
    return moves

# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

# Ne pas modifier le code ci-dessous :

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            tree_list = list(map(int, lines[0].split()))  # valeur de chaque noeud
            root = build_tree(tree_list)

        return solve(root)
    
    except Exception as e:
        print(f"Error: {e}")

def build_tree(lst):
    if not lst:
        return None
    
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    
    while i < len(lst):
        node = queue.pop(0)
        
        if i < len(lst):
            node.left = TreeNode(lst[i])
            queue.append(node.left)
            i += 1
        
        if i < len(lst):
            node.right = TreeNode(lst[i])
            queue.append(node.right)
            i += 1
    
    return root

def print_tree(root):
    if not root:
        return
    current_level = [root]
    while current_level:
        next_level = []
        values = []
        for node in current_level:
            if node:
                values.append(str(node.val))
                if node.left != None :
                    next_level.append(node.left)
                if node.right != None :
                    next_level.append(node.right)
        print(" ".join(values))
        current_level = next_level

def main():
    if len(sys.argv) != 2:
        print("Usage: python distribution.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()
