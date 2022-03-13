import csv
import util_cleaning


def get_mohx_tuples():
    sentence_target_index_label = []
    csv_file = open('../../nlp-finalproject/data/MOH-X/MOH-X_formatted_svo.csv')
    lines = csv.reader(csv_file)
    # Skip over first line
    next(lines)
    for line in lines:
        # Below line is partially taken from the github for the original project
        sentence_target_index_label.append([line[3][1:].split(), int(line[4]), int(line[5])])
    return util_cleaning.tokenize_and_print_metrics(sentence_target_index_label)


if __name__ == '__main__':
    get_mohx_tuples()
