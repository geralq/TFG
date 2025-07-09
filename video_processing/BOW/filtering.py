import pandas as pd

def filter_signers(summary_df, invalid_signers=None):
    """
    Limpia los firmantes no deseados y agrupa por firmante.
    """
    if invalid_signers is None:
        invalid_signers = ['3,2', '4']
    
    signer_elan = summary_df.loc[1:, ['ELAN file', 'Signer']]
    signer_elan['Signer'] = signer_elan['Signer'].astype(str).str.strip()
    signer_elan = signer_elan[~signer_elan['Signer'].isin(invalid_signers)]
    return signer_elan.groupby('Signer')['ELAN file'].apply(list).reset_index()

def filter_rare_words_per_signer(results, min_signers=3):
    """
    Elimina palabras que aparecen en menos de `min_signers` firmantes.
    """
    word_signer_count = {}
    for result in results:
        for word in result['Bag_of_Words']:
            word_signer_count[word] = word_signer_count.get(word, 0) + 1

    filtered = []
    for result in results:
        filtered_bow = {
            word: count
            for word, count in result['Bag_of_Words'].items()
            if word_signer_count[word] >= min_signers
        }
        filtered.append({'Signer': result['Signer'], 'Bag_of_Words': filtered_bow})
    return filtered

def filter_frequent_words(df, min_total_count=60):
    """
    Elimina columnas (palabras) con frecuencia total menor a `min_total_count`.
    """
    word_totals = df.iloc[:, 1:].sum()
    keep_words = word_totals[word_totals >= min_total_count].index
    return df[['Signer'] + list(keep_words)]
