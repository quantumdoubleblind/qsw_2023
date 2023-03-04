import numpy as np


class NumberPartitioning:

    def __init__(self, problem_size):
        self.problem_size = problem_size

    # (c - 2* sum_(j=1)_m nj*xj)**2
    def generate_qubo(self, numbers):
        n = len(numbers)
        c = sum(numbers)

        q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    q[i][j] = numbers[i] * (numbers[i] - c)
                if i < j:
                    print(i,j)
                    q[i][j] = 2 *numbers[i] * numbers[j]

        return q

    @classmethod
    def generate_problems(self, n_problems, size):
        problems = np.random.randint(0, 100, (n_problems, size))
        return problems
