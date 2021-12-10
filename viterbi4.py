import sys

class Viterbi:
    def __init__(self, test_word_ls, emission_matrix, transition_matrix_pair, transition_matrix_triplet, unique_tags_start_stop) -> None:
        self.pi = {}
        self.n = len(test_word_ls)  # number of words in word_seq
        self.test_word_ls = test_word_ls  # x
        self.emission_matrix = emission_matrix
        self.transition_matrix_pair = transition_matrix_pair
        self.transition_matrix_triplet = transition_matrix_triplet
        self.unique_tags_start_stop = unique_tags_start_stop

    def initialise(self):
        self.test_word_ls = ["start"] + self.test_word_ls + ["stop"]

        for i in range(self.n + 2):
            row = {}
            for tag in self.unique_tags_start_stop:
                row[tag] = -1.0

            self.pi[i] = row

        # initialisation: pi(0, u) 1 for START, 0 otherwise
        for u in self.unique_tags_start_stop:
            if u == "start":
                self.pi[0][u] = 1
            else:
                self.pi[0][u] = 0

    def recursive_step(self):
        for j in range(0, self.n):
            for u in self.unique_tags_start_stop:
                score_max = -sys.maxsize

                for v in self.unique_tags_start_stop: # source state
                    for w in self.unique_tags_start_stop: # state before source

                        test_word = self.test_word_ls[j + 1]
                        if test_word not in self.emission_matrix["start"].keys():
                            test_word = "#UNK#"
                        
                        if (j == 0) and (v == "start"):
                            #first state only have 1 state before it
                            try:
                                score = self.pi[j][v] * self.emission_matrix[u][test_word] * self.transition_matrix_pair[v][u]
                            except KeyError:
                                score = 0
                        else:
                            try:
                                score = self.pi[j][v] * self.emission_matrix[u][test_word] * self.transition_matrix_triplet[w][v][u]
                            except KeyError:
                                score = 0

                    if score > score_max:
                        score_max = score

                self.pi[j + 1][u] = score_max

    def final_step(self):
        score_max = -sys.maxsize

        for v in self.unique_tags_start_stop[1:-1]:
            for w in self.unique_tags_start_stop[:-1]:
                score = self.pi[self.n][v] * self.transition_matrix_triplet[w][v]["stop"]

            if score > score_max:
                score_max = score

        self.pi[self.n + 1]["stop"] = score_max

    def get_tag_seq(self):
        tag_seq = []

        y_n_star = ""

        y_n_score_max = -sys.maxsize
        for u in self.unique_tags_start_stop[1:-1]:
            for w in self.unique_tags_start_stop[1:-1]:
                score = self.pi[self.n][u] * self.transition_matrix_triplet[w][u]["stop"]
                if score > y_n_score_max:
                    y_n_score_max = score
                    y_n_star = u

        tag_seq.insert(0, y_n_star)

        for j in range(self.n - 1, 0, -1):
            y_j_star = ""
            y_j_score_max = -1

            for u in self.unique_tags_start_stop[1:-1]:
                for w in self.unique_tags_start_stop[1:-1]:
                    score = self.pi[j][u] * self.transition_matrix_triplet[w][u][tag_seq[0]]

                    if score > y_j_score_max:
                        y_j_score_max = score
                        y_j_star = u
                tag_seq.insert(0, y_j_star)
            
        return tag_seq
