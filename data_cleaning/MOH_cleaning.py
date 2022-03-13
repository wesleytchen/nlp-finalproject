import csv
import util_cleaning

def get_moh_tuples():
    sentence_target_index_label = []
    csv_file = open('../../nlp-finalproject/data/moh/MOH_formatted.csv')
    lines = csv.reader(csv_file)
    # Skip over first line
    next(lines)
    for line in lines:
        # Below line is partially taken from the github for the original project
        print(line)
        sentence_target_index_label.append([line[1].split(), int(line[2]), int(line[3])])

    return util_cleaning.tokenize_and_print_metrics(sentence_target_index_label)


if __name__ == '__main__':
    get_moh_tuples()
