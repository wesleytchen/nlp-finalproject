from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()


# Takes in an array of tuples, with each tuple containing words in a list, the target word, and the associated label
# Prints vocab length, number of sentences, percentage of sentences that are metaphors, number of tokens, type to
# token ratio and average sentence length
# Returns an array of tuples, with each tuple containing tokens, the target word and the associated label
def tokenize_and_print_metrics(lines):
    vocab = set()
    num_metaphor = 0
    total_sentences = 0
    num_tokens = 0
    sentence_tokens_target_label = []
    for sentence, target, label in lines:
        if label == 1:
            num_metaphor += 1
        sentence_tokens = sentence
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
    return sentence_tokens_target_label


