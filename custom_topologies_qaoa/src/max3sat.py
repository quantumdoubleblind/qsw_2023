# Copyright (c) 
# SPDX-License-Identifier: GPL-2.0
# coding: utf-8

import random
import itertools
import numpy as np


class Max3Sat:
    # min function example 3 variables sum_(i=1)_(m) ((1+wi)*(yi1 + yi2 + yi3) - yi1*yi2 - yi1*yi3 - yi2*yi3 - 2*w1)
    # variables: number of variables | clauses: list of clauses e.g [((0, True), (1, False)), ((0, False), (1, True))]
    def __init__(self, variables):
        self.variables = variables

    def generate_qubo(self, clauses):
        n = self.variables + len(clauses)
        q = np.zeros((n, n))

        for i, clause in enumerate(clauses):
            clause_idx = self.variables + i
            q[clause_idx][clause_idx] += 2

            # generating first part of the qubo (1+wi)*(yi1 + yi2 + yi3)
            # (1 + w1) * (x1 + x2 + x3) = x1 + x2 + x3 + w1x1 + w1x2 + w1x3
            # (1 + w1) * ((1 - x1) + x2 + x3) = -x1 + x2 + x3 + w1 - w1x1 + w1x2 + w1x3

            for item in clause:
                item_idx = item[0]
                val = item[1]
                if val:
                    q[item_idx][item_idx] -= 1
                    q[item_idx][clause_idx] -= 1
                if not val:
                    q[item_idx][item_idx] += 1
                    q[clause_idx][clause_idx] -= 1
                    q[item_idx][clause_idx] += 1

            for (item1, item2) in itertools.combinations(clause, 2):
                idx1 = item1[0]
                idx2 = item2[0]
                val1 = item1[1]
                val2 = item2[1]

                if val1 and val2:
                    q[idx1][idx2] += 1
                if not val1 and val2:
                    q[idx2][idx2] += 1.
                    q[idx1][idx2] -= 1.
                if val1 and not val2:
                    q[idx1][idx1] += 1.
                    q[idx1][idx2] -= 1
                if not val1 and not val2:
                    q[idx1][idx2] += 1.
                    q[idx1][idx1] -= 1.
                    q[idx2][idx2] -= 1.

        return q

    @classmethod
    def generate_problems(self, n_problems, size=(3, 10)):
        # size: First is variables, second is number of clauses.
        # Want something like this:
        # [((1, True), (2, True), (3, True)), ((1, True), (2, False), (4, False))]
        variables = size[0] - 1
        problems = []
        for _ in range(n_problems):
            problem = []

            # Eg: 10 clauses
            for _ in range(size[1]):
                # Each has THREE tuples!! But random vars and random value.
                # (True/False).
                idx1 = int(round(random.random() * variables))
                idx2 = int(round(random.random() * variables))
                idx3 = int(round(random.random() * variables))
                val1 = bool(random.getrandbits(1))
                val2 = bool(random.getrandbits(1))
                val3 = bool(random.getrandbits(1))

                clause = (
                    (abs(idx1), val1),
                    (abs(idx2), val2),
                    (abs(idx3), val3)
                )
                clause = sorted(clause)

                problem.append(clause)
            if len(problem) == size[1]:
                problems.append(sorted(problem))

        return problems
