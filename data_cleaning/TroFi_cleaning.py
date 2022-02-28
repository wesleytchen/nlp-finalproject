import util_cleaning
from nltk.stem import PorterStemmer
from nltk import WordPunctTokenizer
import nltk
import spacy


def get_trofi_tuples():
    tokenizer = WordPunctTokenizer()
    stemmer = PorterStemmer()
    nlp = spacy.load('en_core_web_md')
    trofi_file = open('data/trofi/TroFiExampleBase.txt')
    lines = []
    label = 0
    target_word = ''

    # Labeling was done by clusters, meaning that even if a line was unannotated, if it was clustered under "literal", it was
    # classified as literal
    for line in trofi_file:
        # 6 astericks means the beginning of a target verb's associated sentences
        num_asterics_in_line = len(line) - len(line.replace("*", ""))
        if num_asterics_in_line == 6:
            target_word = line.replace("*", "").strip()

        # 2 astericks means beginning of literal or nonliteral cluster
        elif num_asterics_in_line == 2:
            if "nonliteral" in line:
                label = 1
            else:
                label = 0
        # Sentences containing wsj at beginning are the sentences to use as training data
        elif 'wsj' in line:
            line = line.strip()
            line_by_space = line.split()
            words_in_line = line_by_space[2:]
            words_in_line[len(words_in_line)-1] = words_in_line[len(words_in_line)-1].split('/')[0]

            # We need to find the index of the target word. We will first try doing this by looking for words that have
            # the same stem as the target word. If no words have the same stem, then find the word with the minimal edit
            # distance
            cur_index = 0
            target_index = -1
            for word in words_in_line:
                word_tokens = tokenizer.tokenize(word)
                for tok in word_tokens:
                    if stemmer.stem(tok) == stemmer.stem(target_word):
                        target_index = cur_index
                cur_index += 1

            if target_index == -1:
                list_of_differences = []
                for word in words_in_line:
                    word_tokens = tokenizer.tokenize(word)
                    list_of_tok_differences = []
                    for tok in word_tokens:
                        list_of_tok_differences.append(nlp(tok).similarity(nlp(target_word))/nltk.edit_distance(tok, target_word))
                    list_of_differences.append(max(list_of_tok_differences))
                target_index = list_of_differences.index(max(list_of_differences))
                print("The target word is: " + str(target_word))
                print("The identified word is: " + str(words_in_line[target_index]))
            lines.append((words_in_line, target_index, label))

    sentence_tokens_target_label = util_cleaning.tokenize_and_print_metrics(lines)
    return sentence_tokens_target_label

if __name__ == '__main__':
    get_trofi_tuples()
