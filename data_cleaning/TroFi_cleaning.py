import util_cleaning

trofi_file = open('../data/trofi/TroFiExampleBase.txt')
lines = []
label = 0
cur_word = ''

# Labeling was done by clusters, meaning that even if a line was unannotated, if it was clustered under "literal", it was
# classified as literal
for line in trofi_file:

    # 6 astericks means the beginning of a target verb's associated sentences
    num_asterics_in_line = len(line) - len(line.replace("*", ""))
    if num_asterics_in_line == 6:
        cur_word = line.replace("*", "").strip()

    # 2 astericks means beginning of literal or nonliteral cluster
    elif num_asterics_in_line == 2:
        if "nonliteral" in line:
            label = 1
        else:
            label = 0
    # Sentences containing wsj at beginning are the sentences to use as training data
    elif 'wsj' in line:
        cur_label = label
        line = line.strip()
        line_by_space = line.split()

        words_in_line = line_by_space[2:]
        words_in_line[len(words_in_line)-1] = words_in_line[len(words_in_line)-1].split('/')[0]
        lines.append((words_in_line, cur_word, label))


sentence_tokens_target_label = util_cleaning.tokenize_and_print_metrics(lines)
