import sys
import collections

def filter_top_5_tag(dict):
    return collections.OrderedDict(sorted(dict.items(), reverse=True)[:5])

class Viterbi:
    def __init__(self, test_word_ls, emission_matrix, transition_matrix,
                 unique_tags_start_stop) -> None:
        self.pi = {}
        self.n = len(test_word_ls)  # number of words in word_seq
        self.test_word_ls = test_word_ls  # x
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix
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

                for v in self.unique_tags_start_stop:
                    test_word = self.test_word_ls[j + 1]
                    if test_word not in self.emission_matrix["start"].keys():
                        test_word = "#UNK#"
                    try:
                        score = self.pi[j][v] * self.emission_matrix[u][
                            test_word] * self.transition_matrix[v][u]
                    except KeyError:
                        score = 0

                    if score > score_max:
                        score_max = score

                self.pi[j + 1][u] = score_max

    def final_step(self):
        score_max = -sys.maxsize

        for v in self.unique_tags_start_stop[:-1]:
            score = self.pi[self.n][v] * self.transition_matrix[v]["stop"]

            if score > score_max:
                score_max = score

        self.pi[self.n + 1]["stop"] = score_max

    def get_tag_seq(self):
        tag_seq = []
        top_5_tag = {}

        y_n_star = ""

        # y_n_score_max = -sys.maxsize
        for u in self.unique_tags_start_stop[1:-1]:
            score = self.pi[self.n][u] * self.transition_matrix[u]["stop"]
            top_5_tag[score] = [u]

        top_5_tag = filter_top_5_tag(top_5_tag)

        for j in range(self.n - 1, 0, -1):
            top_tag = {}

            for tag_seq in top_5_tag.values():
                for u in self.unique_tags_start_stop[1:-1]:
                    score = self.pi[j][u] * self.transition_matrix[u][tag_seq[0]]
                    top_tag[score] = [u] + tag_seq

            top_5_tag = filter_top_5_tag(top_tag)
        
        tag_seq = list(top_5_tag.values())[-1]

        return tag_seq
