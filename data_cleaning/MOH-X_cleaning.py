import csv
import util_cleaning
sentence_target_index_label = []

csv_file = open('../../nlp-finalproject/data/MOH-X/MOH-X_Data.csv')
lines = csv.reader(csv_file)
# Skip over first line
next(lines)
for line in lines:
    sentence_target_index_label.append([line[3][1:].split(), int(line[4]), int(line[5])])

util_cleaning.tokenize_and_print_metrics(sentence_target_index_label)
