import pandas as pd

def build_bow_dataframe(filtered_results):
    df = pd.DataFrame(filtered_results)
    vocab = set()
    for bow in df['Bag_of_Words']:
        vocab.update(bow.keys())
    vocab = list(vocab)

    matrix = []
    for bow in df['Bag_of_Words']:
        row = [bow.get(word, 0) for word in vocab]
        matrix.append(row)

    signer_bow = pd.DataFrame(matrix, columns=vocab)
    signer_bow.insert(0, 'Signer', df['Signer'])
    return signer_bow
