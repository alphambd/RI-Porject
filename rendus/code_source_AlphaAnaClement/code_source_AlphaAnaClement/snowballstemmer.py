"""
Snowball English Stemmer (Porter2) - Complete Implementation
Fusion des fichiers essentiels : Among + BaseStemmer + EnglishStemmer
"""

# ==================== PARTIE 1: Among Class ====================
class Among:
    """Class for searching patterns in the stemmer"""
    def __init__(self, s, substring_i, result, method=None):
        self.s = s                    # search string
        self.substring_i = substring_i # index to longest matching substring  
        self.result = result          # result of the lookup
        self.method = method          # method to use if substring matches

# ==================== PARTIE 2: BaseStemmer Class ====================
class BaseStemmer:
    """Base class for all Snowball stemmers"""
    
    def __init__(self):
        self.set_current("")

    def set_current(self, value):
        self.current = value
        self.cursor = 0
        self.limit = len(self.current)
        self.limit_backward = 0
        self.bra = self.cursor
        self.ket = self.limit

    def get_current(self):
        return self.current

    def copy_from(self, other):
        self.current = other.current
        self.cursor = other.cursor
        self.limit = other.limit
        self.limit_backward = other.limit_backward
        self.bra = other.bra
        self.ket = other.ket

    # === Méthodes de groupe de caractères ===
    def in_grouping(self, s, min, max):
        if self.cursor >= self.limit:
            return False
        ch = ord(self.current[self.cursor])
        if ch > max or ch < min:
            return False
        ch -= min
        if (s[ch >> 3] & (0x1 << (ch & 0x7))) == 0:
            return False
        self.cursor += 1
        return True

    def in_grouping_b(self, s, min, max):
        if self.cursor <= self.limit_backward:
            return False
        ch = ord(self.current[self.cursor - 1])
        if ch > max or ch < min:
            return False
        ch -= min
        if (s[ch >> 3] & (0x1 << (ch & 0x7))) == 0:
            return False
        self.cursor -= 1
        return True

    def out_grouping(self, s, min, max):
        if self.cursor >= self.limit:
            return False
        ch = ord(self.current[self.cursor])
        if ch > max or ch < min:
            self.cursor += 1
            return True
        ch -= min
        if (s[ch >> 3] & (0x1 << (ch & 0x7))) == 0:
            self.cursor += 1
            return True
        return False

    def out_grouping_b(self, s, min, max):
        if self.cursor <= self.limit_backward:
            return False
        ch = ord(self.current[self.cursor - 1])
        if ch > max or ch < min:
            self.cursor -= 1
            return True
        ch -= min
        if (s[ch >> 3] & (0x1 << (ch & 0x7))) == 0:
            self.cursor -= 1
            return True
        return False

    # === Méthodes de recherche ===
    def eq_s(self, s):
        if self.limit - self.cursor < len(s):
            return False
        if self.current[self.cursor:self.cursor+len(s)] != s:
            return False
        self.cursor += len(s)
        return True

    def eq_s_b(self, s):
        if self.cursor - self.limit_backward < len(s):
            return False
        if self.current[self.cursor-len(s):self.cursor] != s:
            return False
        self.cursor -= len(s)
        return True

    def find_among(self, v):
        i = 0
        j = len(v)
        c = self.cursor
        l = self.limit
        common_i = 0
        common_j = 0
        first_key_inspected = False

        while True:
            k = i + ((j - i) >> 1)
            diff = 0
            common = min(common_i, common_j)
            w = v[k]
            for i2 in range(common, len(w.s)):
                if c + common == l:
                    diff = -1
                    break
                diff = ord(self.current[c + common]) - ord(w.s[i2])
                if diff != 0:
                    break
                common += 1
            if diff < 0:
                j = k
                common_j = common
            else:
                i = k
                common_i = common
            if j - i <= 1:
                if i > 0:
                    break
                if j == i:
                    break
                if first_key_inspected:
                    break
                first_key_inspected = True
        
        while True:
            w = v[i]
            if common_i >= len(w.s):
                self.cursor = c + len(w.s)
                if w.method is None:
                    return w.result
                method = getattr(self, w.method)
                if method():
                    self.cursor = c + len(w.s)
                    return w.result
            i = w.substring_i
            if i < 0:
                return 0
        return -1

    def find_among_b(self, v):
        i = 0
        j = len(v)
        c = self.cursor
        lb = self.limit_backward
        common_i = 0
        common_j = 0
        first_key_inspected = False

        while True:
            k = i + ((j - i) >> 1)
            diff = 0
            common = min(common_i, common_j)
            w = v[k]
            for i2 in range(len(w.s) - 1 - common, -1, -1):
                if c - common == lb:
                    diff = -1
                    break
                diff = ord(self.current[c - 1 - common]) - ord(w.s[i2])
                if diff != 0:
                    break
                common += 1
            if diff < 0:
                j = k
                common_j = common
            else:
                i = k
                common_i = common
            if j - i <= 1:
                if i > 0:
                    break
                if j == i:
                    break
                if first_key_inspected:
                    break
                first_key_inspected = True
        
        while True:
            w = v[i]
            if common_i >= len(w.s):
                self.cursor = c - len(w.s)
                if w.method is None:
                    return w.result
                method = getattr(self, w.method)
                if method():
                    self.cursor = c - len(w.s)
                    return w.result
            i = w.substring_i
            if i < 0:
                return 0
        return -1

    # === Méthodes de manipulation de texte ===
    def replace_s(self, c_bra, c_ket, s):
        adjustment = len(s) - (c_ket - c_bra)
        self.current = self.current[0:c_bra] + s + self.current[c_ket:]
        self.limit += adjustment
        if self.cursor >= c_ket:
            self.cursor += adjustment
        elif self.cursor > c_bra:
            self.cursor = c_bra
        return adjustment

    def slice_from(self, s):
        self.replace_s(self.bra, self.ket, s)
        self.ket = self.bra + len(s)

    def slice_del(self):
        return self.slice_from("")

    def slice_to(self):
        return self.current[self.bra:self.ket]

    def assign_to(self):
        return self.current[0:self.limit]

    # === Interface publique ===
    def stemWord(self, word):
        self.set_current(word)
        self._stem()
        return self.get_current()

    def stemWords(self, words):
        return [self.stemWord(word) for word in words]

    def _stem(self):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _stem method")

# ==================== PARTIE 3: EnglishStemmer ====================
class EnglishStemmer(BaseStemmer):
    """
    English Stemmer implementation based on Snowball Porter2 algorithm
    """
    
    def __init__(self):
        super().__init__()
        self.p1 = 0
        self.p2 = 0
        self.Y_found = False
        
        # Character groups
        self.vowels = "aeiouy"
        self.v_WXY = "aeiouywxY"
        self.valid_LI = "cdeghkmnrt"
    
    def _stem(self):
        """Main stemming algorithm"""
        # Handle exceptions first
        if self._exception1():
            return
            
        # Skip if word is too short
        if len(self.current) < 3:
            return
            
        self._prelude()
        self._mark_regions()
        
        # Apply stemming steps
        self._step_1a()
        self._step_1b()
        self._step_1c()
        self._step_2()
        self._step_3()
        self._step_4()
        self._step_5()
        
        self._postlude()
    
    def _prelude(self):
        """Initial processing"""
        self.Y_found = False
        
        # Handle apostrophes
        if self.current.startswith("'"):
            self.current = self.current[1:]
            
        # Convert initial 'y' to 'Y'
        if self.current.startswith('y'):
            self.current = 'Y' + self.current[1:]
            self.Y_found = True
            
        # Convert 'y' to 'Y' after vowels
        result = []
        for i, char in enumerate(self.current):
            if char == 'y' and i > 0 and self.current[i-1] in self.vowels:
                result.append('Y')
                self.Y_found = True
            else:
                result.append(char)
        self.current = ''.join(result)
    
    def _mark_regions(self):
        """Mark regions R1 and R2"""
        self.p1 = len(self.current)
        self.p2 = len(self.current)
        
        # Look for special prefixes
        special_prefixes = [
            'gener', 'commun', 'arsen', 'past', 'univers', 
            'later', 'emerg', 'organ', 'inter'
        ]
        
        for prefix in special_prefixes:
            if self.current.startswith(prefix):
                self.p1 = len(prefix)
                self.p2 = self._find_p2(self.p1)
                return
                
        # Default region marking
        cursor = 0
        found_vowel = False
        while cursor < len(self.current):
            if self.current[cursor] in self.vowels:
                found_vowel = True
            elif found_vowel:
                self.p1 = cursor
                self.p2 = self._find_p2(self.p1)
                return
            cursor += 1
    
    def _find_p2(self, start):
        """Find p2 region starting from given position"""
        cursor = start
        found_vowel = False
        while cursor < len(self.current):
            if self.current[cursor] in self.vowels:
                found_vowel = True
            elif found_vowel:
                return cursor
            cursor += 1
        return len(self.current)
    
    def _R1(self):
        """Check if cursor is in R1 region"""
        return self.cursor >= self.p1
    
    def _R2(self):
        """Check if cursor is in R2 region"""
        return self.cursor >= self.p2
    
    def _shortv(self):
        """Check for short vowel pattern"""
        if len(self.current) < 3:
            return False
        return (self.current[-3] not in self.v_WXY and 
                self.current[-2] in self.vowels and 
                self.current[-1] not in self.vowels)
    
    def _step_1a(self):
        """Step 1a: Handle plurals and past participles"""
        if self.current.endswith('sses'):
            self.current = self.current[:-2]  # sses -> ss
        elif self.current.endswith('ies'):
            self.current = self.current[:-2]  # ies -> i
        elif self.current.endswith('s') and not self.current.endswith('ss'):
            stem = self.current[:-1]
            if self._contains_vowel(stem):
                self.current = stem
    
    def _step_1b(self):
        """Step 1b: Handle various verb endings"""
        if self.current.endswith('eed'):
            stem = self.current[:-3]
            if self._measure(stem) > 0:
                self.current = stem + 'ee'
                return
        elif self.current.endswith(('ed', 'edly', 'ing', 'ingly')):
            if self.current.endswith('ed'):
                stem = self.current[:-2]
            elif self.current.endswith('edly'):
                stem = self.current[:-4]
            elif self.current.endswith('ing'):
                stem = self.current[:-3]
            else:  # 'ingly'
                stem = self.current[:-5]
                
            if self._contains_vowel(stem):
                self.current = stem
                
                # Additional processing
                if self.current.endswith(('at', 'bl', 'iz')):
                    self.current += 'e'
                elif self._ends_with_double_consonant() and not self.current.endswith(('l', 's', 'z')):
                    self.current = self.current[:-1]
                elif (self._measure(self.current) == 1 and 
                      self._ends_cvc() and 
                      not self.current.endswith('w')):
                    self.current += 'e'
    
    def _step_1c(self):
        """Step 1c: Change y to i"""
        if (self.current.endswith('y') or self.current.endswith('Y')) and len(self.current) > 1:
            stem = self.current[:-1]
            if self._contains_vowel(stem):
                self.current = stem + 'i'
    
    def _step_2(self):
        """Step 2: Double-letter endings"""
        suffixes = {
            'tional': 'tion', 'enci': 'ence', 'anci': 'ance',
            'abli': 'able', 'entli': 'ent', 'izer': 'ize',
            'ization': 'ize', 'ational': 'ate', 'ation': 'ate',
            'ator': 'ate', 'alism': 'al', 'aliti': 'al',
            'alli': 'al', 'fulness': 'ful', 'ousli': 'ous',
            'ousness': 'ous', 'iveness': 'ive', 'iviti': 'ive',
            'biliti': 'ble', 'bli': 'ble', 'ogi': 'og',
            'fulli': 'ful', 'lessli': 'less', 'li': ''
        }
        
        for suffix, replacement in suffixes.items():
            if self.current.endswith(suffix):
                stem = self.current[:-len(suffix)]
                if self._measure(stem) > 0:
                    # Special cases
                    if suffix == 'ogi' and not stem.endswith('l'):
                        continue
                    if suffix == 'li' and stem[-1] not in self.valid_LI:
                        continue
                    self.current = stem + replacement
                    return
    
    def _step_3(self):
        """Step 3: Various replacements"""
        suffixes = {
            'tional': 'tion', 'ational': 'ate', 'alize': 'al',
            'icate': 'ic', 'iciti': 'ic', 'ical': 'ic',
            'ful': '', 'ness': '', 'ative': ''
        }
        
        for suffix, replacement in suffixes.items():
            if self.current.endswith(suffix):
                stem = self.current[:-len(suffix)]
                if self._measure(stem) > 0:
                    # Special case for 'ative'
                    if suffix == 'ative' and not (self.p2 > 0 and len(stem) >= self.p2):
                        continue
                    self.current = stem + replacement
                    return
    
    def _step_4(self):
        """Step 4: Final cleanup"""
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible',
            'ant', 'ement', 'ment', 'ent', 'ism', 'ate', 'iti',
            'ous', 'ive', 'ize'
        ]
        
        for suffix in suffixes:
            if self.current.endswith(suffix):
                stem = self.current[:-len(suffix)]
                if self._measure(stem) > 1:
                    self.current = stem
                    return
        
        # Special case for 'ion'
        if self.current.endswith('ion') and len(self.current) > 3:
            stem = self.current[:-3]
            if self._measure(stem) > 1 and stem[-1] in 'st':
                self.current = stem
    
    def _step_5(self):
        """Step 5: Final adjustments"""
        # Step 5a
        if self.current.endswith('e'):
            stem = self.current[:-1]
            measure = self._measure(stem)
            if measure > 1 or (measure == 1 and not self._ends_cvc(stem)):
                self.current = stem
        
        # Step 5b  
        if (self._measure(self.current) > 1 and 
            self._ends_with_double_consonant() and 
            self.current.endswith('l')):
            self.current = self.current[:-1]
    
    def _exception1(self):
        """Handle exceptional cases"""
        exceptions = {
            'skis': 'ski', 'skies': 'sky', 'dying': 'die',
            'lying': 'lie', 'tying': 'tie', 'idly': 'idl',
            'gently': 'gentl', 'ugly': 'ugli', 'early': 'earli',
            'only': 'onli', 'singly': 'singl',
            'sky': 'sky', 'news': 'news', 'howe': 'howe',
            'atlas': 'atlas', 'cosmos': 'cosmos', 'bias': 'bias', 'andes': 'andes'
        }
        
        if self.current in exceptions:
            self.current = exceptions[self.current]
            return True
        return False
    
    def _postlude(self):
        """Final processing - convert Y back to y"""
        if self.Y_found:
            self.current = self.current.replace('Y', 'y')
    
    # Helper methods
    def _measure(self, stem):
        """Calculate VC measure"""
        count = 0
        in_vowel = False
        for char in stem:
            if char in self.vowels:
                if not in_vowel:
                    in_vowel = True
            else:
                if in_vowel:
                    count += 1
                    in_vowel = False
        return count
    
    def _contains_vowel(self, stem):
        return any(char in self.vowels for char in stem)
    
    def _ends_with_double_consonant(self):
        if len(self.current) < 2:
            return False
        return (self.current[-1] == self.current[-2] and 
                self.current[-1] not in self.vowels)
    
    def _ends_cvc(self, word=None):
        if word is None:
            word = self.current
        if len(word) < 3:
            return False
        return (word[-1] not in self.v_WXY and
                word[-2] in self.vowels and
                word[-3] not in self.vowels)

# ==================== INTERFACE SIMPLE ====================
def stem_word(word):
    """Stem a single word using Porter2 algorithm"""
    stemmer = EnglishStemmer()
    return stemmer.stemWord(word)

def stem_words(words):
    """Stem a list of words"""
    stemmer = EnglishStemmer()
    return [stemmer.stemWord(word) for word in words]

def stem_text(text):
    """Stem all words in a text"""
    stemmer = EnglishStemmer()
    words = text.split()
    stemmed_words = [stemmer.stemWord(word) for word in words]
    return ' '.join(stemmed_words)

# ==================== TESTS ====================
if __name__ == "__main__":
    # Test the stemmer
    test_words = [
        "running", "cats", "happily", "conventional", "agreement",
        "skiing", "dying", "lying", "early", "usually",
        "generation", "communication", "organization"
    ]
    
    print("Snowball English Stemmer (Porter2) - Tests")
    print("=" * 50)
    
    for word in test_words:
        stemmed = stem_word(word)
        print(f"{word:15} -> {stemmed}")
    
    # Test avec une phrase
    print("\n" + "=" * 50)
    text = "The running cats are happily playing with conventional agreements"
    print(f"Original: {text}")
    print(f"Stemmed:  {stem_text(text)}")