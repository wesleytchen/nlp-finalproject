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
total_sentences = 0
num_tokens = 0
sentence_tokens_target_label = []

for sentence, target, label in lines:
    if label == 1:
        num_metaphor += 1
    sentence = ' '.join(sentence)
    sentence_tokens = tokenizer.tokenize(sentence)
    for token in sentence_tokens:
        num_tokens += 1
        vocab.add(token)
    sentence_tokens_target_label.append((sentence_tokens, target, label))
    total_sentences += 1

vocab_size = len(vocab)
percent_metaphors = num_metaphor / total_sentences
type_to_token = vocab_size / num_tokens
average_sentence_length = num_tokens / total_sentences


print("The length of the vocab is: " + str(vocab_size))
print("The number of sentences is: " + str(total_sentences))
print("The percentage of sentences that are metaphors is: " + str(percent_metaphors))
print("The number of tokens is: " + str(num_tokens))
print("The type to token ratio is: " + str(type_to_token))
print("The average sentence length is: " + str(average_sentence_length))


