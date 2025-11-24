"""
Lovins Stemmer - version simplifiée
"""

def stem_lovins(word):
    """Stemmer Lovins basique"""
    if len(word) <= 2:
        return word.lower()
    
    word_lower = word.lower()
    
    suffixes = [
        'alistically', 'arizability', 'izationally', 'antialness', 'arisations',
        'arizations', 'entialness', 'fulnesses', 'ivenesses', 'izations',
        'lessnesses', 'nesses', 'ously', 'able', 'ible', 'al', 'ance', 'ence',
        'ed', 'en', 'er', 'es', 'ful', 'ic', 'ing', 'ism', 'ist', 'ity', 'ive',
        'ize', 'ly', 'ment', 'ness', 'ous', 's', 'y'
    ]
    
    # Trier par longueur décroissante
    suffixes.sort(key=len, reverse=True)
    
    for suffix in suffixes:
        if word_lower.endswith(suffix):
            stem = word_lower[:-len(suffix)]
            if len(stem) >= 2:
                return stem
    
    return word_lower