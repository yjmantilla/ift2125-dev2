from distribution import process_numbers

results = [8,0,9]
passed = 0

if __name__ == "__main__":
    for i in range(1,len(results) + 1):
        filename = "tree"+str(i)+".txt"
        res = process_numbers(filename)
        if (res == results[i-1]) :
            passed +=1
            print(f"Test with {filename} passed")
        else :
            print(f"Test with {filename} failed")
            print(f"Expected {results[i-1]}, got {res}")