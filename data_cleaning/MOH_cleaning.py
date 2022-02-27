import util_cleaning

MOH_file = open('data/trofi/TroFiExampleBase.txt')

lines = MOH_file.readlines()
number_of_lines = len(lines)
sentence_target_label = []

# Go through each line other than the first line and the last two as those aren't labeled sentences
for line in MOH_file.readlines()[1:number_of_lines-2]:
    line = line.split()
    # Target word is defined at the beginning of each sentence
    target_word = line[0]
    length_of_line = len(line)
    # Sentence starts at third word (delineated by space) and ends at third to last word.
    sentence = line[2:length_of_line-2]
    # Label is at 2nd to last word (delinated by space)
    label_word = sentence[length_of_line-2]
    if label_word == 'literal':
        label = 0
    else:
        label = 1
    sentence_target_label.append((sentence, target_word, label))

sentence_tokens_target_label = util_cleaning.tokenize_and_print_metrics(sentence_target_label)
