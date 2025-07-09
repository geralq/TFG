from BOW.data_loader import load_excel_sheets, clean_glosses
from BOW.bow_generator import build_bows_per_signer
from BOW.filtering import filter_signers, filter_rare_words_per_signer, filter_frequent_words
from BOW.matrix_builder import build_bow_dataframe

def main():

    dfs = load_excel_sheets('/home/gerardo/LSE_HEALTH/LSE-Health-UVigo.xlsx')

    glosses_df = clean_glosses(dfs['#Glosses'])

    signer_elan_df = filter_signers(dfs['Summary'])

    signer_elan_df.to_excel("signer_video.xlsx", index=False)

    results = build_bows_per_signer(glosses_df, signer_elan_df)

    filtered_results = filter_rare_words_per_signer(results)

    bow_df = build_bow_dataframe(filtered_results)
    
    bow_df = filter_frequent_words(bow_df)

    bow_df.to_excel("signer_bow.xlsx", index=False)

if __name__ == "__main__":
    main()
