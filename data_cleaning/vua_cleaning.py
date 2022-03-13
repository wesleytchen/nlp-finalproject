import csv
import util_cleaning

def get_vua_tuples():
    train_val_test = []
    for path in ['train', 'test', 'val']:
        sentence_target_index_label = []
        vua_file = open('../../nlp-finalproject/data/vua/VUA_formatted_' + path + '.csv', encoding='latin-1')
        lines = csv.reader(vua_file)
        next(lines)
        for line in lines:
            sentence_target_index_label.append([line[3].split(), int(line[4]), int(line[5])])
        train_val_test.append(util_cleaning.tokenize_and_print_metrics(sentence_target_index_label))
    return train_val_test

if __name__ == '__main__':
    get_vua_tuples()
