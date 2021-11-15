from tqdm import tqdm
import itertools

languages = ["RU"]#["ES", "RU"]

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
            print(sentence)
            word_seq = []
            tag_seq = []
            for word_tag in sentence.split("\n"):
                word, tag = word_tag.split(" ")
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


for lang in languages:
    print(lang)
    tag_total, word_total, test_word_total = read_data(lang)