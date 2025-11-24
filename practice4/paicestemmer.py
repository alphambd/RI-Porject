"""
Paice/Husk Stemmer - version simplifiée
"""

def stem_paice(word):
    """Stemmer Paice/Husk basique"""
    if len(word) <= 2:
        return word.lower()
    
    word_lower = word.lower()
    
    rules = [
        ('sses', 'ss'), ('ies', 'y'), ('ss', 'ss'), ('s', ''),
        ('eed', 'ee'), ('ed', ''), ('ing', ''),
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
        ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
        ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'),
        ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
        ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
        ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
        ('iviti', 'ive'), ('biliti', 'ble'), ('logi', 'log'),
        ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
        ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''),
        ('ness', '')
    ]
    
    def ends_with_double_consonant(w):
        if len(w) < 2:
            return False
        return w[-1] == w[-2] and w[-1] in 'bcdfghjklmnpqrstvwxz'
    
    for suffix, replacement in rules:
        if word_lower.endswith(suffix):
            stem = word_lower[:-len(suffix)] + replacement
            if len(stem) >= 2:
                return stem
    
    if word_lower.endswith('e') and len(word_lower) > 2:
        return word_lower[:-1]
    elif word_lower.endswith('l') and len(word_lower) > 2 and ends_with_double_consonant(word_lower[:-1]):
        return word_lower[:-1]
    
    return word_lower