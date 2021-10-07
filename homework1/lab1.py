import math
import re
import numpy as np

def prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqrt = int(math.sqrt(n)) + 1

    for d in range(3, sqrt, 2):
        if n % d == 0:
            return False
    return True

#print(prime(31))

def sortWords(path):
    return sorted(i.lower() for i in re.split('\W+', open(path).read())[:-1])

#print(sortWords('./Latin-Lipsum.txt'))

def dotProduct(matrix, vector):

    if len(matrix[0]) != len(vector):
        print("Can't be multiplied")
        return None

    result = []
    for row in matrix:
        result.append(sum([a * b for a,b in zip(row, vector)]))

    return result

#print(dotProduct([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], [2, -5, 7, -10]))

########################################

def first():
    matrix = np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]])
    print(matrix[0:2, -2:])
    print(matrix[-1, -2:])

#first()

def second():
    vector1 = np.random.uniform(size=(3,))
    vector2 = np.random.uniform(size=(3,))
    
    if np.sum(vector1) > np.sum(vector2):
        print("sum(vector1) > sum(vector2)")
    else: 
        print("sum(vector2) > sum(vector1)")
    
    print(np.add(vector1, vector2))
    print(np.cross(vector1, vector2))
    print(np.dot(vector1, vector2))
    print(np.sqrt(vector1))
    print(np.sqrt(vector2))

#second()

def third():
    matrix = np.random.uniform(size=(5,5))

    print(matrix.T)
    print(np.linalg.inv(matrix))
    print(np.linalg.det(matrix))

    vector = np.random.uniform(size=(5,))

    print(np.dot(matrix, vector))

third()



