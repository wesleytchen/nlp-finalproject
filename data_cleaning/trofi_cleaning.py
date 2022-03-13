import util_cleaning
import csv


def get_trofi_tuples():
    sentence_target_index_label = []
    csv_file = open('../nlp-finalproject/data/trofi/TroFi_formatted_all3737.csv')
    lines = csv.reader(csv_file)
    next(lines)
    for line in lines:
        sentence_target_index_label.append([line[1].strip(), int(line[2]), int(line[3])])
    return sentence_target_index_label

if __name__ == '__main__':
    get_trofi_tuples()
