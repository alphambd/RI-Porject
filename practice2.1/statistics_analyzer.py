
from advanced_indexer import AdvancedInvertedIndex
from stop_words_indexer import StopWordsIndexer
from steming_indexer import StemmingIndexer
import nltk
from nltk.stem import PorterStemmer


def compare_with_stop_words(collection_file, stop_words_file):
    """Comparer les statistiques avec/sans stop words"""
    
    # Indexation sans stop words (baseline)
    baseline_index = AdvancedInvertedIndex()
    baseline_index.build_from_file(collection_file)
    baseline_stats = baseline_index.compute_statistics()
    
    # Indexation avec stop words
    stop_words_index = StopWordsIndexer(stop_words_file)
    stop_words_index.build_from_file(collection_file)
    stop_words_stats = stop_words_index.compute_statistics()
    
    print("=== COMPARAISON AVEC/SANS STOP WORDS ===")
    print(f"Taille vocabulaire: {baseline_stats['vocabulary_size']} → {stop_words_stats['vocabulary_size']}")
    print(f"Réduction: {((baseline_stats['vocabulary_size'] - stop_words_stats['vocabulary_size']) / baseline_stats['vocabulary_size'] * 100):.1f}%")
    
    return baseline_stats, stop_words_stats

def comprehensive_analysis(collection_file, stop_words_file):
    """Analyse comparative des trois techniques"""
    
    # 1. Baseline (sans traitement)
    baseline_index = AdvancedInvertedIndex()
    baseline_index.build_from_file(collection_file)
    
    # 2. Avec stop words
    stop_words_index = StopWordsIndexer(stop_words_file)
    stop_words_index.build_from_file(collection_file)
    
    # 3. Avec stop words + stemming
    stemming_index = StemmingIndexer(stop_words_file)
    stemming_index.build_from_file(collection_file)
    
    # Collecter les statistiques
    results = {
        'baseline': baseline_index.compute_statistics(),
        'stop_words': stop_words_index.compute_statistics(),
        'stemming': stemming_index.compute_statistics()
    }
    
    return results