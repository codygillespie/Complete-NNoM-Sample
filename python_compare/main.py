import os

C_RESULTS_PATH = os.path.join(os.getcwd(), 'workspace/c/results.csv')
PY_RESULTS_PATH = os.path.join(os.getcwd(), 'workspace/python/results.csv')

def main():
    lines_total = 0
    lines_same = 0

    with open(C_RESULTS_PATH, 'r') as c_file:
        with open(PY_RESULTS_PATH, 'r') as py_file:
            for c_line in c_file:
                py_line = py_file.readline()
                lines_total += 1
                if c_line == py_line:
                    lines_same += 1
    
    print(f'SUMMARY: {lines_same} of {lines_total} lines are the same between the C and Python implementations.')

if __name__ == "__main__":
    main()