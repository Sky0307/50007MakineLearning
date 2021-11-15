from enum import unique
from re import split
from tqdm import tqdm
import itertools

# languages = ["RU"]
languages = ["RU"]
# languages = ["ES", "RU"]

def read_data(lang):
    tag_total = []
    word_total = []
    test_word_total = []

    train_path = f'{lang}/train'
    test_path = f'{lang}/dev.in'

    with open(train_path, "r") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in tqdm(sentences):
            word_seq = []
            tag_seq = []
            for word_tag in sentence.split("\n"):
                # word, tag = word_tag.split(" ")
                
                split_character = word_tag.split(" ")
                if len(split_character) > 2:
                    tag = split_character[-1]
                    word = " ".join(split_character[0:2])
                    print(word)
                else:
                    word, tag = split_character

                tag_seq.append(tag)
                word_seq.append(word)
            
            tag_total.append(tag_seq)
            word_total.append(word_seq)
    
    with open(test_path, "r") as f:
        document = f.read().rstrip()
        sentences = document.split("\n\n")

        for sentence in tqdm(sentences):
            word_seq = []
            for word in sentence.split("\n"):
                word_seq.append(word)
            test_word_total.append(word_seq)
    
    return tag_total, word_total, test_word_total

# backbone code for getting unique tags & words
def get_unique_component(elements):
    #flatten the nested list, then using the set properties to remove duplicate elements
    return list(set(list(itertools.chain.from_iterable(elements))))

#to get unique word with the above function defined
def get_unique_word(word_list):
    unique_word = get_unique_component(word_list)
    unique_word.sort()
    return unique_word

#to get unique tag with the above function defined
def get_unique_tag(tag_list):
    unique_tag = get_unique_component(tag_list)
    unique_tag.sort()
    tags_with_start_stop = ["START"] + unique_tag + ["STOP"]
    return unique_tag, tags_with_start_stop


for lang in languages:
    tag_total, word_total, test_word_total = read_data(lang)
    print(len(tag_total))
    print(len(word_total))
    print(len(test_word_total))
    unique_tag, tag_with_start_stop = get_unique_tag(tag_total)
    unique_word = get_unique_word(word_total)
    print(len(unique_tag))
    print(len(unique_word))