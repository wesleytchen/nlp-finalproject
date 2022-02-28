import csv
import util_cleaning


def get_vua_tuples():
    sentence_target_index_label = []
    vua_file = open('../../nlp-finalproject/data/vua/VUA_formatted.csv', encoding='latin-1')
    lines = csv.reader(vua_file)
    next(lines)
    for line in lines:
        sentence_target_index_label.append([line[3].split(), int(line[4]), int(line[5])])
    return util_cleaning.tokenize_and_print_metrics(sentence_target_index_label)


if __name__ == '__main__':
    get_vua_tuples()
