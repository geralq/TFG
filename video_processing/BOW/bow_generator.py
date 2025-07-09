def generate_bow(glosses, filename):
    row = glosses[glosses.iloc[:, 0] == filename].iloc[0]
    return {col: row[col] for col in glosses.columns[1:] if row[col] > 1}

def sum_bow(*bows):
    result = {}
    for bow in bows:
        for word, count in bow.items():
            result[word] = result.get(word, 0) + count
    return {word: count for word, count in result.items() if count > 5}

def build_bows_per_signer(glosses, signer_elan):
    results = []
    for signer, elan_files in signer_elan.values:
        bows = [generate_bow(glosses, filename) for filename in elan_files]
        total_bow = sum_bow(*bows)
        results.append({'Signer': signer, 'Bag_of_Words': total_bow})
    return results
