import re
import os
import time
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional
import pickle

# IMPORT LXML pour parsing XML correct
try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    print("ATTENTION: lxml non installé. Installation recommandée: pip install lxml")
    import xml.etree.ElementTree as ET
    LXML_AVAILABLE = False

from porterstemmer import PorterStemmer
from snowballstemmer import stem_word


class INEXDocument:
    """Classe pour représenter un document INEX avec son arbre XML"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.doc_id = None
        self.tree = None
        self.root = None
        
    def parse(self):
        """Parse le document XML avec lxml"""
        try:
            if LXML_AVAILABLE:
                parser = etree.XMLParser(recover=True, remove_comments=True)
                self.tree = etree.parse(self.xml_path, parser)
            else:
                self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
            
            # Extraire l'ID du document
            self.doc_id = self._extract_doc_id()
            return True
        except Exception as e:
            print(f"Erreur parsing {self.xml_path}: {e}")
            return False
    
    def _extract_doc_id(self) -> str:
        """Extrait l'ID du document depuis l'XML de manière robuste"""
        
        # 1. Chercher <id> dans le header
        try:
            if LXML_AVAILABLE:
                header = self.root.find('.//header')
                if header is not None:
                    id_elem = header.find('id')
                    if id_elem is not None and id_elem.text:
                        id_text = id_elem.text.strip()
                        if id_text.isdigit():
                            return id_text
        except:
            pass
        
        # 2. Chercher n'importe quel <id> dans le document
        try:
            for elem in self.root.iter():
                tag = self._clean_tag(elem.tag)
                if tag == 'id' and elem.text:
                    id_text = elem.text.strip()
                    if id_text.isdigit():
                        return id_text
        except:
            pass
        
        # 3. Extraire du nom de fichier
        filename = os.path.basename(self.xml_path)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return max(numbers, key=len)
        
        # 4. Fallback: hash du chemin
        return hashlib.md5(self.xml_path.encode()).hexdigest()[:8]
    
    def get_inex_elements(self, target_tags: Set[str]) -> List[Dict]:
        """
        Extrait les éléments XML pertinents pour INEX avec des seuils optimisés
        pour éviter les small_err_nodes
        """
        elements = []
        
        if self.root is None:
            return elements
        
        # Tags principaux pour INEX
        main_tags = {'article', 'bdy', 'sec', 'p'}
        
        def build_inex_path(elem, include_article=True):
            """Construit un chemin XPath INEX-compatible"""
            path_parts = []
            current = elem
            
            while current is not None:
                tag = self._clean_tag(current.tag)
                
                if tag in main_tags:
                    parent = current.getparent()
                    if parent is not None:
                        siblings = [c for c in parent if self._clean_tag(c.tag) == tag]
                        try:
                            index = siblings.index(current) + 1
                        except ValueError:
                            index = 1
                        path_parts.insert(0, f"{tag}[{index}]")
                    else:
                        path_parts.insert(0, f"{tag}[1]")
                
                current = current.getparent()
                if tag == 'article' and not include_article:
                    break
            
            if not path_parts:
                return "/article[1]"
            
            return '/' + '/'.join(path_parts)
        
        def extract_meaningful_text(elem) -> str:
            """Extrait le texte en évitant les small_err_nodes"""
            try:
                if LXML_AVAILABLE:
                    text_parts = []
                    for t in elem.itertext():
                        if t:
                            cleaned = t.strip()
                            if cleaned and len(cleaned) > 3:  # Ignorer mots isolés
                                text_parts.append(cleaned)
                    result = ' '.join(text_parts)
                    
                    # Nettoyer les espaces multiples
                    result = re.sub(r'\s+', ' ', result)
                    return result.strip()
                else:
                    # Version ElementTree
                    text = (elem.text or "").strip()
                    for child in elem:
                        child_text = extract_meaningful_text(child)
                        if child_text:
                            text += " " + child_text
                    if elem.tail and elem.tail.strip():
                        text += " " + elem.tail.strip()
                    return ' '.join(text.split())
            except Exception:
                return ""
        
        def is_valid_content(text: str, tag: str) -> bool:
            """Valide le contenu selon le type d'élément"""
            if not text or not text.strip():
                return False
            
            clean_text = text.strip()
            length = len(clean_text)
            words = len(clean_text.split())
            
            # Seuils minimaux par tag (augmentés pour éviter small_err_nodes)
            min_requirements = {
                'article': {'chars': 200, 'words': 30},
                'bdy': {'chars': 100, 'words': 20},
                'sec': {'chars': 50, 'words': 10},
                'p': {'chars': 40, 'words': 8}  # Augmenté de 10 à 40 caractères
            }
            
            req = min_requirements.get(tag, {'chars': 30, 'words': 5})
            if length < req['chars'] or words < req['words']:
                return False
            
            # Éviter les éléments non informatifs
            lower_text = clean_text.lower()
            non_informative = [
                'see also', 'references', 'external links',
                'further reading', 'bibliography', 'contents',
                'navigation menu', 'jump to'
            ]
            
            for pattern in non_informative:
                if pattern in lower_text and words < 15:
                    return False
            
            return True
        
        def process_element(elem, level=0):
            """Traite un élément XML récursivement"""
            tag = self._clean_tag(elem.tag)
            
            if tag in target_tags or tag == 'article':
                text = extract_meaningful_text(elem)
                
                if is_valid_content(text, tag):
                    xml_path = build_inex_path(elem, include_article=(tag != 'article'))
                    
                    # Limiter la profondeur
                    if xml_path.count('/') <= 8:
                        element_id = f"{self.doc_id}_{hashlib.md5(xml_path.encode()).hexdigest()[:8]}"
                        
                        elements.append({
                            'elem_id': element_id,
                            'doc_id': self.doc_id,
                            'tag': tag,
                            'text': text,
                            'xml_path': xml_path,
                            'source_file': self.xml_path,
                            'depth': level,
                            'is_article': (tag == 'article'),
                            'priority': self._get_tag_priority(tag),
                            'char_count': len(text)
                        })
            
            # Explorer les enfants avec limite de profondeur
            if level < 12:
                for child in elem:
                    process_element(child, level + 1)
        
        process_element(self.root)
        
        # Éliminer les doublons et trier
        unique_elements = []
        seen_paths = set()
        
        for elem in elements:
            if elem['xml_path'] not in seen_paths:
                seen_paths.add(elem['xml_path'])
                unique_elements.append(elem)
        
        # Trier par priorité et profondeur
        unique_elements.sort(key=lambda x: (
            0 if x['is_article'] else 1,
            -x['priority'],
            x['depth'],
            x['xml_path']
        ))
        
        return unique_elements
    
    def _clean_tag(self, tag):
        """Nettoie le tag du namespace"""
        if isinstance(tag, str):
            if '}' in tag:
                return tag.split('}', 1)[1]
        return str(tag)
    
    def _get_tag_priority(self, tag):
        """Retourne la priorité du tag (plus haut = plus prioritaire)"""
        priorities = {
            'p': 4,
            'sec': 3, 
            'bdy': 2,
            'article': 1
        }
        return priorities.get(tag, 0)


class WeightedInvertedIndex:
    """Index inversé optimisé pour INEX"""
    
    def __init__(self):
        self.dictionary = defaultdict(dict)  # term -> {doc_id: freq}
        self.doc_ids = []                    # Liste de tous les docs
        self.doc_lengths = {}                # doc_id -> longueur en terms
        self.doc_count = 0
        self.total_terms = 0
        
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
        
        # Cache des documents XML (clés seulement)
        self.xml_cache_keys = set()
    
    def configure(self, tokenization="basic", stemmer="nostem", 
                  stop_words="nostop", **kwargs):
        """Configure tous les paramètres"""
        self.configure_tokenization(tokenization)
        self.configure_stemmer(stemmer)
        self.configure_stop_words(stop_words)
        
        if 'target_tags' in kwargs:
            self.target_tags = kwargs['target_tags']
    
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
    
    # Méthodes de tokenization (une seule version de chaque)
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
    
    def _index_document_content(self, doc_id: str, text: str, 
                               metadata: Dict = None) -> bool:
        """Indexe un contenu texte (commun aux articles et éléments)"""
        # Tokenization
        tokens = self.apply_tokenization(text)
        
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
    
    def build_index_from_articles(self, xml_dir: str, 
                                 max_files: Optional[int] = None) -> float:
        """Indexe les articles complets (phase FETCH)"""
        print(f"Indexation FETCH des articles depuis {xml_dir}...")
        start_time = time.time()
        
        self.doc_type = "article"
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        success_count = 0
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"  Traitement article {i}/{len(xml_files)}...")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
                continue
            
            # Extraire tout le texte de l'article
            text = self._extract_full_article_text(doc.root)
            
            if text and len(text) > 100:  # Au moins 100 caractères
                doc_id = doc.doc_id
                
                if self._index_document_content(doc_id, text):
                    self.xml_cache_keys.add(doc_id)
                    
                    self.store_metadata(doc_id, {
                        'doc_id': doc_id,
                        'xml_path': '/article[1]',
                        'tag': 'article',
                        'type': 'article',
                        'source_file': xml_file
                    })
                    success_count += 1
        
        self.doc_count = success_count
        indexing_time = time.time() - start_time
        
        print(f"Indexation FETCH terminée: {self.doc_count} articles "
              f"en {indexing_time:.2f}s")
        return indexing_time
    
    def _extract_full_article_text(self, root) -> str:
        """Extrait tout le texte pertinent d'un article"""
        text_parts = []
        
        def collect_text(elem):
            if hasattr(elem, 'tag'):
                tag = elem.tag if hasattr(elem.tag, 'strip') else str(elem.tag)
                if '}' in tag:
                    tag = tag.split('}', 1)[1]
                
                # Ignorer certaines balises non textuelles
                if tag in ['link', 'image', 'caption', 'ref']:
                    return
                
                if elem.text:
                    text_parts.append(elem.text.strip())
                
                for child in elem:
                    collect_text(child)
                
                if elem.tail:
                    text_parts.append(elem.tail.strip())
        
        collect_text(root)
        return ' '.join(text_parts)
    
    def build_index_from_elements(self, xml_dir: str, 
                                 target_tags: List[str] = ['sec', 'p', 'bdy', 'article'],
                                 max_files: Optional[int] = None) -> float:
        """Indexe les éléments individuels pour phase BROWSE"""
        print(f"Indexation BROWSE des éléments {target_tags}...")
        start_time = time.time()
        
        self.doc_type = "element"
        self.target_tags = set(target_tags)
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        total_elements = 0
        
        for i, xml_file in enumerate(xml_files):
            if i % 50 == 0:
                print(f"  Traitement fichier {i}/{len(xml_files)}...")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
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
                    'xml_path': elem_data['xml_path'],
                    'tag': elem_data['tag'],
                    'type': 'element',
                    'source_file': xml_file,
                    'depth': elem_data.get('depth', 0),
                    'priority': elem_data.get('priority', 0)
                }
                
                if self._index_document_content(elem_id, elem_text, metadata):
                    total_elements += 1
        
        self.doc_count = total_elements
        indexing_time = time.time() - start_time
        
        print(f"Indexation BROWSE terminée: {total_elements} éléments "
              f"en {indexing_time:.2f}s")
        return indexing_time
    
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
                'target_tags': list(self.target_tags) if hasattr(self, 'target_tags') else []
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
        
        if 'xml_cache_keys' in data:
            index.xml_cache_keys = set(data['xml_cache_keys'])
        
        return index