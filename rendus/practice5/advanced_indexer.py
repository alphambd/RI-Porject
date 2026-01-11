import re
import os
import time
import hashlib
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Union
import unicodedata
import html

from inex_document import INEXDocument

class WeightedInvertedIndex:
    """Index inversé unifié pour tous les exercices"""
    
    def __init__(self):
        self.dictionary = defaultdict(dict)  # term -> {doc_id: freq}
        self.doc_ids = []                    # Liste de tous les docs
        self.doc_lengths = {}                # doc_id -> longueur en terms
        self.doc_count = 0
        self.total_terms = 0
        
        # Statistiques tokens/terms
        self.total_tokens_bp = 0
        self.distinct_tokens_bp = set()
        self.total_chars_tokens = 0
        self.avg_doc_length = 0  
        
        # Configuration
        self.stop_words_set = set()
        self.stemmer_func = None
        self.tokenization_method = "basic"
        self.stop_list_name = "nostop"
        self.stemmer_name = "nostem"
        
        # Métadonnées
        self.metadata_store = {}
        self.doc_type = "article"
        self.target_tags = []
        
        # Cache des documents XML
        self.xml_cache_keys = set()
        
        # Support pour les deux méthodes de parsing
        self.use_lxml = LXML_AVAILABLE
    
    def configure(self, tokenization="basic", stemmer="nostem", 
                  stop_words="nostop", **kwargs):
        """Configure tous les paramètres"""
        self.configure_tokenization(tokenization)
        self.configure_stemmer(stemmer)
        self.configure_stop_words(stop_words)
        
        if 'target_tags' in kwargs:
            self.target_tags = kwargs['target_tags']
        
        # mettre use_lxml par défaut : 
        use_lxml = True
        if 'use_lxml' in kwargs:
            self.use_lxml = kwargs['use_lxml'] and LXML_AVAILABLE
    
    def configure_tokenization(self, method="basic"):
        """Configure la méthode de tokenization"""
        self.tokenization_method = method
    
    def configure_stemmer(self, stemmer_name="nostem"):
        """Configure l'algorithme de stemming"""
        self.stemmer_name = stemmer_name
        
        if stemmer_name == "porter":
            stemmer = PorterStemmer()
            self.stemmer_func = lambda word: stemmer.stem(word, 0, len(word) - 1)
        elif stemmer_name == "snowball":
            self.stemmer_func = stem_word
        else:
            self.stemmer_func = None
    
    def configure_stop_words(self, stop_list_name="nostop"):
        """Configure la liste de stop-words"""
        self.stop_list_name = stop_list_name
        if stop_list_name != "nostop":
            self._load_stop_words(stop_list_name)
    
    def _load_stop_words(self, stop_list_name="stop671"):
        """Charge différentes listes de stop-words"""
        stop_files = {
            "stop671": "data/stopwords/stop-words-english4.txt",
            "stop319": "data/stopwords/stop-words-english5.txt",
            "stop733": "data/stopwords/stop-words-kaggle.txt"
        }
        
        file_path = stop_files.get(stop_list_name)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() 
                                        for line in file if line.strip())
    
    # Méthodes de tokenization
    def _tokenize_basic(self, text):
        """Tokenization basique: seulement lettres"""
        text = re.sub(r'[^A-Za-z\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]
    
    def _tokenize_extended(self, text):
        """Tokenization étendue: lettres et chiffres"""
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]
    
    def _tokenize_hyphen(self, text):
        """Tokenization qui garde les traits d'union"""
        text = re.sub(r'[^A-Za-z\-\s]', ' ', text)
        tokens = []
        for token in text.split():
            if '-' in token and len(token) > 1:
                tokens.extend([token] + token.split('-'))
            else:
                tokens.append(token)
        return [t for t in tokens if len(t) > 0]
    
    def _tokenize_apostrophe(self, text):
        """Tokenization qui garde les apostrophes"""
        text = re.sub(r'[^A-Za-z\'\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]
    
    def apply_tokenization(self, text):
        """Applique la tokenization configurée"""
        methods = {
            "basic": self._tokenize_basic,
            "extended": self._tokenize_extended,
            "hyphen": self._tokenize_hyphen,
            "apostrophe": self._tokenize_apostrophe
        }
        tokenizer = methods.get(self.tokenization_method, self._tokenize_basic)
        return tokenizer(text)
    
    def process_tokens(self, tokens):
        """Transforme les tokens en terms avec la configuration actuelle"""
        # Case folding
        tokens = [t.lower() for t in tokens]
        
        # Stop words
        if self.stop_list_name != "nostop":
            tokens = [token for token in tokens 
                     if token not in self.stop_words_set]
        
        # Stemming
        if self.stemmer_func:
            tokens = [self.stemmer_func(token) for token in tokens]
        
        return tokens
    
    def store_metadata(self, doc_id, metadata_dict):
        """Stocke les métadonnées pour un document/élément"""
        self.metadata_store[doc_id] = metadata_dict
    
    def get_metadata(self, doc_id):
        """Récupère les métadonnées d'un document/élément"""
        metadata = self.metadata_store.get(doc_id, {
            'doc_id': doc_id,
            'xml_path': '/article[1]',
            'type': self.doc_type,
            'tag': 'article' if self.doc_type == 'article' else 'unknown'
        })
        
        # S'assurer que parent_doc_id existe pour les éléments
        if 'parent_doc_id' not in metadata:
            if self.doc_type == 'element' and '_' in doc_id:
                metadata['parent_doc_id'] = doc_id.split('_')[0]
            else:
                metadata['parent_doc_id'] = doc_id
        
        return metadata
    
    def get_parent_article_id(self, element_id):
        """Pour un élément, retourne l'ID de l'article parent"""
        metadata = self.get_metadata(element_id)
        return metadata.get('parent_doc_id', element_id.split('_')[0] if '_' in element_id else element_id)
    
    def get_xml_path(self, doc_id):
        """Retourne le chemin XML complet"""
        metadata = self.get_metadata(doc_id)
        return metadata.get('xml_path', '/article[1]')
    
    def _index_document_content(self, doc_id: str, text: str, 
                               metadata: Dict = None) -> bool:
        """Indexe un contenu texte (commun aux articles et éléments)"""
        # Tokenization
        tokens = self.apply_tokenization(text)
        
        # Mise à jour statistiques tokens
        self.total_tokens_bp += len(tokens)
        self.distinct_tokens_bp.update(tokens)
        self.total_chars_tokens += sum(len(t) for t in tokens)
        
        # Transformation tokens -> terms
        terms = self.process_tokens(tokens)
        
        if not terms:
            return False
        
        # Mise à jour de l'index
        doc_length = len(terms)
        self.doc_ids.append(doc_id)
        self.doc_lengths[doc_id] = doc_length
        self.total_terms += doc_length
        
        # Ajout au dictionnaire
        term_freq = Counter(terms)
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
        
        # Métadonnées
        if metadata:
            self.store_metadata(doc_id, metadata)
        
        return True
    
    def build_index_from_xml_collection(self, xml_dir: str, 
                                    max_files: Optional[int] = None) -> float:
        """Indexe les articles complets (phase FETCH ou exercices 1-2)"""
        print(f"Indexation des articles depuis {xml_dir}...")
        start_time = time.time()
        
        self.doc_type = "article"
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        success_count = 0
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"\r  Traitement article {i}/{len(xml_files)}...", end="", flush=True)
            
            doc = INEXDocument(xml_file)
            if not doc.parse(self.use_lxml):
                continue
            
            # Utiliser extract_full_article_text() qui fait le nettoyage complet
            text = doc.extract_full_article_text()
            
            if text and len(text) > 100:
                doc_id = doc.doc_id
                
                if self._index_document_content(doc_id, text):
                    self.xml_cache_keys.add(doc_id)
                    
                    self.store_metadata(doc_id, {
                        'doc_id': doc_id,
                        'parent_doc_id': doc_id,
                        'xml_path': '/article[1]',
                        'tag': 'article',
                        'type': 'article',
                        'source_file': xml_file
                    })
                    success_count += 1
        
        self.doc_count = success_count
        if self.doc_count > 0:
            self.avg_doc_length = self.total_terms / self.doc_count
        
        indexing_time = time.time() - start_time
        print(f"Indexation terminée: {self.doc_count} articles en {indexing_time:.2f}s")
        return indexing_time
    
    def build_index_from_articles(self, xml_dir: str, 
                                 max_files: Optional[int] = None) -> float:
        """Alias pour compatibilité avec le code existant"""
        return self.build_index_from_xml_collection(xml_dir, max_files)
    
    def build_index_from_xml_elements(self, xml_dir: str, 
                                     target_tags: List[str] = ['sec', 'p', 'bdy'],
                                     max_files: Optional[int] = None) -> float:
        """Indexe les éléments individuels (phase BROWSE ou exercices 3-4)"""
        print(f"Indexation des éléments {target_tags}...")
        start_time = time.time()
        
        self.doc_type = "element"
        self.target_tags = set(target_tags)
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        total_elements = 0
        
        for i, xml_file in enumerate(xml_files):
            if i % 50 == 0:
                print(f"\r  Traitement fichier {i}/{len(xml_files)}...", end="", flush=True)
            
            doc = INEXDocument(xml_file)
            if not doc.parse(self.use_lxml):
                continue
            
            # Extraire tous les éléments cibles
            elements = doc.get_inex_elements(set(target_tags))
            
            for elem_data in elements:
                elem_id = elem_data['elem_id']
                elem_text = elem_data['text']
                
                metadata = {
                    'doc_id': elem_data['doc_id'],
                    'parent_doc_id': elem_data['doc_id'],
                    'element_id': elem_id,
                    'xml_path': elem_data.get('xml_path', elem_data.get('full_path', '/article[1]')),
                    'tag': elem_data['tag'],
                    'type': 'element',
                    'source_file': xml_file,
                    'depth': elem_data.get('depth', 0),
                    'priority': elem_data.get('priority', 0)
                }
                
                if self._index_document_content(elem_id, elem_text, metadata):
                    total_elements += 1
        
        self.doc_count = total_elements
        if self.doc_count > 0:
            self.avg_doc_length = self.total_terms / self.doc_count
        
        indexing_time = time.time() - start_time
        print(f"Indexation terminée: {total_elements} éléments en {indexing_time:.2f}s")
        return indexing_time
    
    def build_index_from_elements(self, xml_dir: str, 
                                 target_tags: List[str] = ['sec', 'p', 'bdy'],
                                 max_files: Optional[int] = None) -> float:
        """Alias pour compatibilité"""
        return self.build_index_from_xml_elements(xml_dir, target_tags, max_files)
    
    def _get_xml_files(self, xml_dir: str, max_files: Optional[int] = None) -> List[str]:
        """Liste récursivement les fichiers XML"""
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        if max_files:
            xml_files = xml_files[:max_files]
        
        return xml_files
    
    def get_collection_statistics(self, indexing_time=None):
        #Calcule TOUTES les statistiques demandées
        total_tokens = self.total_tokens_bp
        distinct_tokens = len(self.distinct_tokens_bp)

        avg_token_length = (
            sum(len(token) for token in self.distinct_tokens_bp) / distinct_tokens
            if distinct_tokens > 0 else 0
        )

        total_terms = self.total_terms
        distinct_terms = len(self.dictionary)
        total_chars_terms = sum(len(term) for term in self.dictionary.keys())
        avg_term_length = total_chars_terms / distinct_terms if distinct_terms > 0 else 0

        avg_doc_length = self.avg_doc_length

        stats = {
            'total_tokens': total_tokens,
            'distinct_tokens': distinct_tokens,
            'avg_token_length': avg_token_length,
            'total_terms': total_terms,
            'distinct_terms': distinct_terms,
            'avg_doc_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }
        
        if indexing_time is not None:
            stats['indexing_time'] = indexing_time
        
        return stats
    
    def get_cache_data(self):
        """Retourne les données pouvant être sérialisées pour le cache"""
        return {
            'dictionary': dict(self.dictionary),
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'doc_count': self.doc_count,
            'total_terms': self.total_terms,
            'metadata_store': self.metadata_store,
            'config': {
                'stop_list_name': self.stop_list_name,
                'stemmer_name': self.stemmer_name,
                'tokenization_method': self.tokenization_method,
                'target_tags': list(self.target_tags) if hasattr(self, 'target_tags') else [],
                'use_lxml': self.use_lxml
            },
            'xml_cache_keys': list(self.xml_cache_keys)
        }
    
    def save_to_file(self, filename: str):
        """Sauvegarde l'index dans un fichier"""
        data_to_save = self.get_cache_data()
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load_from_file(cls, filename: str):
        """Charge un index depuis un fichier"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        index = cls()
        index.dictionary = defaultdict(dict, data['dictionary'])
        index.doc_ids = data['doc_ids']
        index.doc_lengths = data['doc_lengths']
        index.doc_count = data['doc_count']
        index.total_terms = data['total_terms']
        index.metadata_store = data['metadata_store']
        
        # Restaurer la configuration
        config = data['config']
        index.stop_list_name = config['stop_list_name']
        index.stemmer_name = config['stemmer_name']
        index.tokenization_method = config['tokenization_method']
        
        if 'target_tags' in config:
            index.target_tags = set(config['target_tags'])
        
        index.use_lxml = config.get('use_lxml', LXML_AVAILABLE)
        
        if 'xml_cache_keys' in data:
            index.xml_cache_keys = set(data['xml_cache_keys'])
        
        return index
    
    def build_index_with_stats(self, xml_dir: str, max_files: Optional[int] = None) -> Dict:
        """
        Indexe les articles et retourne les données complètes pour compute_statistics
        """
        print(f"Indexation FETCH des articles depuis {xml_dir}...")
        start_time = time.time()
        
        # Au lieu de réécrire la logique, utiliser build_index_from_xml_collection
        indexing_time = self.build_index_from_xml_collection(xml_dir, max_files)
        
        # Calculer les statistiques
        stats = self.get_collection_statistics(indexing_time)
        
        print(f"Indexation FETCH terminée: {self.doc_count} articles en {indexing_time:.2f}s")
        
        return {
            'index': self,  # Retourne l'index lui-même
            'indexing_time': indexing_time,
            'stats': stats,
            'config': {
                'tokenization': self.tokenization_method,
                'stemmer': self.stemmer_name,
                'stop_words': self.stop_list_name,
                'use_lxml': self.use_lxml
            }
        }

    def _compute_basic_statistics(self, indexing_time: float) -> Dict:
        #Calcule les statistiques de base pour compute_statistics
        # Statistiques TOKENS
        total_tokens = self.total_tokens_bp
        distinct_tokens = len(self.distinct_tokens_bp)
        avg_token_length = (
            sum(len(token) for token in self.distinct_tokens_bp) / distinct_tokens
            if distinct_tokens > 0 else 0
        )
        
        # Statistiques TERMS
        total_terms = self.total_terms
        distinct_terms = len(self.dictionary)
        
        # Longueur moyenne des terms
        total_chars_terms = sum(len(term) for term in self.dictionary.keys())
        avg_term_length = total_chars_terms / distinct_terms if distinct_terms > 0 else 0
        
        # Longueur moyenne des documents
        avg_doc_length = self.avg_doc_length
        
        return {
            'total_tokens': total_tokens,
            'distinct_tokens': distinct_tokens,
            'avg_token_length': avg_token_length,
            'total_terms': total_terms,
            'distinct_terms': distinct_terms,
            'avg_doc_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }
    