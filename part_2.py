import sys

# languages = ["ES"]
# languages = ["RU"]
languages = ["ES", "RU"]

def read_data(lang):
    tag_total = []
    word_total = []
    test_word_total = []
    tag_seq_start_stop_total = []

    train_path = f'{lang}/train'
    test_path = f'{lang}/dev.in'

    with open(train_path, "r", encoding="UTF-8") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in sentences:
            word_seq = []
            tag_seq = []
            tag_seq_start_stop = []
            for word_tag in sentence.split("\n"):
                # word, tag = word_tag.split(" ")
                
                split_character = word_tag.split(" ")
                if len(split_character) > 2:
                    tag = split_character[-1]
                    # print(tag)
                    word = " ".join(split_character[0:2])
                    # print(word)
                else:
                    word, tag = split_character

                tag_seq.append(tag)
                word_seq.append(word)

                # need to add "start" and "stop" for each tag_seq
                # tag_seq_start_stop.insert(0, "start")
                # tag_seq_start_stop.insert(1, tag_seq)
                # tag_seq_start_stop.insert(-1, "stop")
                tag_seq_start_stop = ["start"] + tag_seq + ["stop"]
            
            tag_total.append(tag_seq)
            word_total.append(word_seq)
            tag_seq_start_stop_total.append(tag_seq_start_stop)
    
    with open(test_path, "r", encoding="UTF-8") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in sentences:
            word_seq = []
            for word in sentence.split("\n"):
                word_seq.append(word)
            test_word_total.append(word_seq)
    
    return tag_total, word_total, test_word_total, tag_seq_start_stop_total

# backbone code for getting unique tags & words
def get_unique_component(elements):
    # flatten the nested list
    flat_list = []
    for sublist in list(elements):
        for i in sublist:
            flat_list.append(i)

    # use the set properties to remove duplicate elements, then convert back to list
    flat_list = list(set(flat_list))
    return flat_list

# to get unique word with the above function defined
def get_unique_word(word_list):
    unique_word = get_unique_component(word_list)

    return unique_word

# to get unique tag with the above function defined
# now also return unique tags with "start" and "stop"
def get_unique_tag(tag_list):
    unique_tag = get_unique_component(tag_list)
    # unique_tag_start_stop = []
    # unique_tag_start_stop.insert(0, "start")
    # unique_tag_start_stop.insert(1, unique_tag)
    # unique_tag_start_stop.insert(-1, "stop")
    unique_tag_start_stop = ["start"] + unique_tag + ["stop"]


    return unique_tag, unique_tag_start_stop

def get_emission_pair(word_list, tag_list):
    emission_pair = []

    # unwrap the nested list
    for tag, word in [(tags, words) for tags in tag_list for words in word_list]:
        emission_pair.append([tag, word])

    return emission_pair

def get_all_emission_pair(unique_word_list, unique_tag_list):
    all_emission_pair = [(tags, words) for tags in unique_tag_list for words in unique_word_list]

    return all_emission_pair

def get_emission_matrix(unique_tag, unique_word, tag_total, word_total, k):
    # use dictionary instead of list to create the matrix
    emission_matrix = {}

    # instantiate the matrix
    for tag in unique_tag:
        row = {}
        for word in unique_word:
            row[word] = 0.0
        row["#UNK#"] = 0.0
        emission_matrix[tag] = row

    # adding count to the matrix with the actual emission pair
    for tags, words in zip(tag_total, word_total):
        for tag, word in zip(tags, words):
            emission_matrix[tag][word] += 1
    
    # get the probability by dividing the tag count
    for tag, matrix_row in emission_matrix.items():
        tag_count = get_tag_count(tag, tag_total) + k

        #row.popitem()
        for word, word_count in matrix_row.items():
            emission_matrix[tag][word] = word_count / tag_count

        emission_matrix[tag]["#UNK#"] = k / tag_count

    return emission_matrix

def get_transition_pair(tag_list):
    transition_pair = []

    # tags[:-1] removes all the "stop"s
    # tags[1:] removes all the "start"s
    for tags in tag_list:
        for tag_no_stop in tags[:-1]:
            for tag_no_start in tags[1:]:
                transition_pair.append([tag_no_stop, tag_no_start])

    return transition_pair

def get_all_transition_pair(unique_tag_list):
    # unique_tag_list[:-1] removes all the "stop"s
    # unique_tag_list[1:] removes all the "start"s
    all_transition_pair = [(tag_no_stop, tag_no_start) for tag_no_stop in unique_tag_list[:-1] for tag_no_start in unique_tag_list[1:]]

    return all_transition_pair

def get_transition_matrix(unique_tag_start_stop, tag_seq_start_stop_total, transition_pair):
    transition_matrix = {}

    for tag1 in unique_tag_start_stop[:-1]:
        row = {}
        for tag2 in unique_tag_start_stop[1:]:
            row[tag2] = 0.0
        transition_matrix[tag1] = row

    # adding count to the matrix with the actual transition pair
    for tag1, tag2 in transition_pair:
        transition_matrix[tag1][tag2] += 1
    
    # get the probability by dividing the tag count
    for tag1, matrix_row in transition_matrix.items():
        tag_count = get_tag_count(tag1, tag_seq_start_stop_total)

        for tag2, word_count in matrix_row.items():
            transition_matrix[tag1][tag2] = word_count / tag_count

    return transition_matrix

def get_tag_count(tag, tag_list):
    get_tag_list = []
    for sublist in tag_list:
        for i in sublist:
            get_tag_list.append(i)

    # get count
    count = get_tag_list.count(tag)

    return count

def get_tag(word, emission_matrix):
    # arbitrary large number
    max_score = -sys.maxsize
    opti_tag = ""

    for tag, matrix_row in emission_matrix.items():
        score = matrix_row[word]
        if score > max_score:
            max_score = score
            opti_tag = tag

    return opti_tag

def predict(test_word_list, emission_matrix, new_words, language):
    result = ""

    for words in test_word_list:
        for word in words:
            opti_tag = ""
            if word in new_words:
                opti_tag = get_tag("#UNK#", emission_matrix)
            else:
                opti_tag = get_tag(word, emission_matrix)

            result += f"{word} {opti_tag}"
            result += "\n"
        result += "\n"
    
    with open(f"{language}/dev.p2.out", "w", encoding="UTF-8") as f:
        f.write(result)

def predict_viterbi(test_word_total, emission_matrix, transition_matrix, unique_tags_start_stop, new_words, language):
    result = ""
    pi = {}
    n = len(test_word_total)
    emission_matrix = get_emission_matrix(unique_tag_start_stop, unique_word, tag_total, word_total, k)


    for word in test_word_total:
        # initialise
        # word.insert(0, "start")
        # word.insert(1, word)
        # word.insert(-1, "stop")
        word = ["start"] + word + ["stop"]


        for i in range(n + 2):
            pi_row = {}
            for tag in unique_tags_start_stop:
                pi_row[tag] = 1.0
            
            pi[i] = pi_row
        
        # initialise pi(0, u) 1 for start, 0 otherwise
        for u in unique_tags_start_stop:
            if u == "start":
                pi[0][u] = 1
            
            else:
                pi[0][u] = 0
        
        for i in range (0, n):
            for u in unique_tags_start_stop:
                max_score = -sys.maxsize

                for v in unique_tags_start_stop:
                    word = test_word_total[i + 1]
                    
                    if word not in emission_matrix["start"].keys():
                        word = "#UNK#"
                    
                    try:
                        score = pi[i][v] * emission_matrix[u][word] * transition_matrix[v][u]
                    
                    except KeyError:
                        score = 0
                    
                    if score > max_score:
                        max_score = score
        
        max_score = -sys.maxsize

        for v in unique_tags_start_stop[:-1]:
            score = pi[n][v] * transition_matrix[v]["stop"]

            if score > max_score:
                max_score = score
            
        pi[n + 1]["stop"] = max_score

        y_seq = []

        y_n_star = ""

        y_n_score_max = -1
        for u in unique_tags_start_stop[1:-1]:
            score = pi[n][u] * transition_matrix[u]["stop"]
            if score > y_n_score_max:
                y_n_score_max = score
                y_n_star = u

        y_seq.insert(0, y_n_star)

        for j in range(n - 1, 0, -1):
            y_j_star = ""
            y_j_score_max = -1
            for u in unique_tags_start_stop[1:-1]:
                score = pi[j][u] * transition_matrix[u][y_seq[0]]
                if score > y_j_score_max:
                    y_j_score_max = score
                    y_j_star = u

            y_seq.insert(0, y_j_star)

    for word in test_word_total:
        for opti_tag in y_seq:
            result += f"{word} {opti_tag}"
            result += "\n"
        result += "\n"

    with open(f"{language}/dev.p2.out", "w", encoding="UTF-8") as f:
        f.write(result)

# use log scale to prevent numerical underflow
if __name__ == "__main__":
    for lang in languages:
        tag_total, word_total, test_word_total, tag_seq_start_stop_total = read_data(lang)

        unique_tag, unique_tag_start_stop = get_unique_tag(tag_total)
        
        # print(unique_tag)
        # print(len(unique_tag))
        # print(len(word_total))
        # print(len(test_word_total))

        unique_word = get_unique_word(word_total)
        unique_test_word = get_unique_word(test_word_total)

        # actual emission observation
        emission_pair = get_emission_pair(word_total, tag_total)
        # possible emission
        all_emission_pair = get_all_emission_pair(unique_word, unique_tag)

        # actual transition observation
        transition_pair = get_transition_pair(tag_seq_start_stop_total)
        # possible transition
        all_transition_pair = get_all_transition_pair(unique_tag_start_stop)

        k = 1
        emission_matrix = get_emission_matrix(unique_tag_start_stop, unique_word, tag_total, word_total, k)
        transition_matrix = get_transition_matrix(unique_tag_start_stop, tag_seq_start_stop_total, transition_pair)

        # use set difference
        new_words = set(unique_test_word).difference(set(unique_word))
        
        predict_viterbi(test_word_total, emission_matrix, transition_matrix, unique_tag_start_stop, new_words, lang)