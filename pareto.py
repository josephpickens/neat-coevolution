from collections import defaultdict

'''
Pareto ranking algorithm derived from "Best Order Sort: A New Algorithm to Non-dominated Sorting for
Evolutionary Multi-objective Optimization" by Roy, Islam, and Deb.

URL: https://www.egr.msu.edu/~kdeb/papers/c2016009.pdf
'''


class ParetoRanker():
    def __init__(self, population, objectives):
        '''
        :param population: a list of agents to pareto rank
        :param objectives: a list of dicts, where j-th dict maps each agent to its score on the j-th objective
        '''
        self.population = population
        self.objectives = objectives
        self.ranked_solutions = [[set()] * len(objectives) for _ in population]
        self.comparison_set = {s: set(range(len(objectives))) for s in population}
        self.is_ranked = defaultdict(bool)
        self.solution_count = 0
        self.rank_count = 0
        self.rank = defaultdict(int)
        self.intralayer_rank = defaultdict(int)
        self.sorted_by_objective = []
        for j in range(len(objectives)):
            self.sorted_by_objective.append(sorted(population, key=objectives[j].get, reverse=True))
        self.layers = None

    def get_rank(self):
        for i in range(len(self.population)):
            for j in range(len(self.objectives)):
                s = self.sorted_by_objective[j][i]
                self.comparison_set[s] -= {j}
                if self.is_ranked[s]:
                    self.ranked_solutions[self.rank[s]][j] |= {s}
                else:
                    self.find_rank(s, j)
                    self.is_ranked[s] = True
                    self.solution_count += 1
            if self.solution_count == len(self.population):
                break
        self.layers = [set().union(*r) for r in self.ranked_solutions[0:self.rank_count + 1]]
        return self.rank, self.intralayer_rank, self.layers

    def find_rank(self, s, j):
        done = False
        for k in range(self.rank_count + 1):
            check = False
            for t in self.ranked_solutions[k][j]:
                check = self.domination_check(s, t)
                if check:
                    self.intralayer_rank[t] += 1
                    break
            if not check:
                self.rank[s] = k
                self.intralayer_rank[s] += 1
                done = True
                self.ranked_solutions[self.rank[s]][j] |= {s}
                break
        if not done:
            self.rank_count += 1
            self.rank[s] = self.rank_count
            self.ranked_solutions[self.rank[s]][j] |= {s}

    def domination_check(self, s, t):
        for j in self.comparison_set[t]:
            if self.sorted_by_objective[j].index(s) < self.sorted_by_objective[j].index(t):
                return False
        return True

