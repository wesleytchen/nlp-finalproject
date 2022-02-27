from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

trofi_file = open('data/trofi/TroFiExampleBase.txt')
lines = []
label = 0
cur_word = ''

# Labeling was done by clusters, meaning that even if a line was unannotated, if it was clustered under "literal", it was
# classified as literal
for line in trofi_file:
    num_asterics_in_line = len(line) - len(line.replace("*", ""))
    if num_asterics_in_line == 6:
        cur_word = line.replace("*", "").strip()
    elif num_asterics_in_line == 2:
        if "nonliteral" in line:
            label = 1
        else:
            label = 0
    elif 'wsj' in line:
        cur_label = label
        line = line.strip()
        line_by_space = line.split()

        words_in_line = line_by_space[2:]
        words_in_line[len(words_in_line)-1] = words_in_line[len(words_in_line)-1].split('/')[0]
        lines.append((words_in_line, cur_word, label))
    else:
        continue

vocab = set()

num_metaphor = 0
total = 0
sentence_tokens_target_label = []
for sentence, target, label in lines:
    if label == 1:
        num_metaphor += 1
    sentence = ' '.join(sentence)
    sentence_tokens = tokenizer.tokenize(sentence)
    for token in sentence_tokens:
        vocab.add(token)
    sentence_tokens_target_label.append((sentence_tokens, target, label))
    total += 1


print(len(vocab))
print(num_metaphor/total)


