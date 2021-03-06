import sys
import itertools

languages = ["ES", "RU"]

def read_data(lang):
    tag_total = []
    word_total = []
    test_word_total = []

    train_path = f'{lang}/train'
    test_path = f'{lang}/dev.in'

    with open(train_path, "r", encoding="UTF-8") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in sentences:
            word_seq = []
            tag_seq = []
            for word_tag in sentence.split("\n"):
                
                split_character = word_tag.split(" ")
                if len(split_character) > 2:
                    tag = split_character[-1]
                    word = " ".join(split_character[0:2])
                else:
                    word, tag = split_character

                tag_seq.append(tag)
                word_seq.append(word)
            
            tag_total.append(tag_seq)
            word_total.append(word_seq)
    
    with open(test_path, "r", encoding="UTF-8") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in sentences:
            word_seq = []
            for word in sentence.split("\n"):
                word_seq.append(word)
            test_word_total.append(word_seq)
    
    return tag_total, word_total, test_word_total

# backbone code for getting unique tags & words
def get_unique_component(elements):
    # flatten the nested list
    # flat_list = []
    # for sublist in elements:
    #     for i in sublist:
    #         flat_list.append(i)

    # # use the set properties to remove duplicate elements, then convert back to list
    # flat_list = list(set(flat_list))
    flat_list = list(set(list(itertools.chain.from_iterable(elements))))
    flat_list.sort()
    return flat_list

def get_emission_pair(word_list, tag_list):
    emission_pair = []

    # unwrap the nested list
    for tag, word in [(tags, words) for tags in tag_list for words in word_list]:
        emission_pair.append([tag, word])

    return emission_pair

def get_all_emission_pair(unique_word_list, unique_tag_list):
    # all_emission_pair = [(tags, words) for tags in unique_tag_list for words in unique_word_list]

    # return all_emission_pair

    return list(itertools.product(unique_tag_list, unique_word_list))

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

        for word, word_count in matrix_row.items():
            emission_matrix[tag][word] = word_count / tag_count

        emission_matrix[tag]["#UNK#"] = k / tag_count

    return emission_matrix

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
    
    with open(f"{language}/dev.p1.out", "w", encoding="UTF-8") as f:
        f.write(result)

if __name__ == "__main__":
    for lang in languages:
        tag_total, word_total, test_word_total = read_data(lang)

        unique_tag = get_unique_component(tag_total)

        unique_word = get_unique_component(word_total)
        unique_test_word = get_unique_component(test_word_total)

        # actual emission observation
        emission_pair = get_emission_pair(word_total, tag_total)
        # possible emission
        all_emission_pair = get_all_emission_pair(unique_word, unique_tag)

        k = 1
        emission_matrix = get_emission_matrix(unique_tag, unique_word, tag_total, word_total, k)

        # use set difference
        new_words = set(unique_test_word).difference(set(unique_word))
        
        predict(test_word_total, emission_matrix, new_words, lang)