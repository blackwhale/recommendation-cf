import numpy as np


def load_data(filename, leading_row=1, leading_column=1, delimiter=','):
    with open(filename, 'rb') as f:
        input_matrix = f.read().splitlines()
        output_matrix = []
        for i, row in enumerate(input_matrix):
            if i < leading_row:
                continue
            row_data = row.split(delimiter)
            row_append = []
            for j, ele in enumerate(row_data):
                if j < leading_column:
                    continue
                row_append.append(int(ele.replace('"', '')))
            output_matrix.append(row_append)
    return np.array(output_matrix, dtype=np.float32)

def save_data(infile, outfile, out_matrix,
              leading_row=1, leading_column=1, delimiter=','):
    with open(infile, 'rb') as f:
        input_lines = f.read().splitlines()
        matrix = []
        for line in input_lines:
            row = line.split(delimiter)
            matrix.append(row)

    with open(outfile, 'wb') as f:
        width = len(matrix[0])
        height = len(matrix)
        for i in xrange(height):
            if i >= leading_row:
                for j in xrange(width):
                    if j < leading_column:
                        continue
                    matrix[i][j] = str(out_matrix[i-leading_row][j-leading_column])
            f.write(delimiter.join(matrix[i])+'\n')
