import os
import time
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from practice4_main import create_index_with_config, compute_statistics, generate_inex_run

def exercise5_tokenization(data_file_path, queries, start_run_id):
    """Teste différentes méthodes de tokenization - Version rapide"""
    
    tokenizations = ["extended", "hyphen", "apostrophe"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for tokenization in tokenizations:
        print(f"\n--- Testing tokenization: {tokenization} ---")
        
        index_data = create_index_with_config(data_file_path, tokenization, "nostem", "nostop")
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        for weighting in weightings:
            print(f"  - Génération run {weighting.upper()}")
            generate_inex_run(current_run_id, ranker, queries, weighting, "article", "nostem", "nostop", tokenization)
            current_run_id += 1
    
    return current_run_id

def exercise5_stemmers(data_file_path, queries, start_run_id):
    """Teste différents algorithmes de stemming - Version rapide"""
    stemmers = ["snowball"] # ajouter possiblement d'autres stemmers
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stemmer in stemmers:
        print(f"\n--- Testing stemmer: {stemmer} ---")
        try:
            index_data = create_index_with_config(data_file_path, "basic", stemmer, "nostop")
            index = index_data['index']
            ranker = RankedRetrieval(index)
            
            for weighting in weightings:
                print(f"  - Génération run {weighting.upper()}")
                generate_inex_run(current_run_id, ranker, queries, weighting, "article", stemmer, "nostop", "basic")
                current_run_id += 1
                
        except Exception as e:
            print(f"Erreur avec le stemmer {stemmer}: {e}")
    
    return current_run_id

def exercise5_stop_words(data_file_path, queries, start_run_id):
    """Teste différentes listes de stop-words - Version rapide"""
    
    stop_lists = ["stop319", "stop733"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stop_list in stop_lists:
        print(f"\n--- Testing stop-words: {stop_list} ---")
        
        index_data = create_index_with_config(data_file_path, "basic", "nostem", stop_list)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        for weighting in weightings:
            print(f"  - Génération run {weighting.upper()}")
            generate_inex_run(current_run_id, ranker, queries, weighting, "article", "nostem", stop_list, "basic")
            current_run_id += 1
    
    return current_run_id