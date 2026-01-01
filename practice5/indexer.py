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
        self.element_cache = {}
        
    def parse(self):
        """Parse le document XML avec lxml"""
        try:
            # Lire le contenu brut d'abord
            with open(self.xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                self._raw_content = f.read()
            
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
        
        # STRATÉGIE: Chercher l'ID dans cet ordre:
        
        # 1. Chercher <id> dans le header (cas normal)
        try:
            if LXML_AVAILABLE:
                # Chercher spécifiquement dans le header
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
            # Prendre le plus grand nombre (généralement l'ID)
            return max(numbers, key=len)
        
        # 4. Fallback: hash du chemin
        return hashlib.md5(self.xml_path.encode()).hexdigest()[:8]
    
    def get_all_elements(self, target_tags: Set[str]) -> List[Dict]:
        #Extrait TOUS les éléments avec leurs chemins XPath exacts
        elements = []
        
        if not self.root:
            return elements
        
        # Fonction récursive pour extraire les éléments
        def extract_recursive(elem, level=0):
            tag = self._clean_tag(elem.tag)
            
            # Si c'est un tag cible, extraire le texte
            if tag in target_tags:
                text = self._extract_element_text(elem)
                if text and len(text.strip()) > 10:
                    # GÉNÉRER LE XPATH AVEC LA NOUVELLE MÉTHODE
                    xpath = self._build_xpath(elem)
                    
                    # Créer un ID unique pour l'élément
                    element_id = f"{self.doc_id}_{hashlib.md5(xpath.encode()).hexdigest()[:8]}"
                    
                    elements.append({
                        'elem_id': element_id,
                        'doc_id': self.doc_id,
                        'tag': tag,
                        'text': text.strip(),
                        'xml_path': xpath,  # <-- ICI LE CHEMIN CORRECT
                        'source_file': self.xml_path,
                        'depth': level
                    })
            
            # Explorer les enfants
            for child in elem:
                extract_recursive(child, level + 1)
        
        extract_recursive(self.root)
        return elements
    
    def get_all_elements_simple_corrected(self, target_tags: Set[str]) -> List[Dict]:
        """Version simplifiée CORRIGÉE pour l'exercice 3"""
        elements = []
        
        if self.root is None:
            return elements
        
        # Dictionnaire pour compter les occurrences de chaque tag par parent
        tag_counters = defaultdict(lambda: defaultdict(int))
        
        def extract_elements(elem, parent_path="", level=0):
            tag = self._clean_tag(elem.tag)
            
            # Construire le chemin actuel
            if not parent_path:
                current_path = f"/{tag}[1]"
            else:
                # Incrémenter le compteur pour ce tag à ce niveau
                parent_key = f"{parent_path}_{tag}"
                tag_counters[parent_key]['count'] += 1
                index = tag_counters[parent_key]['count']
                
                # S'assurer qu'on a le bon index
                # Vérifier combien de frères avec le même tag existent
                parent = elem.getparent()
                if parent is not None:
                    siblings = [c for c in parent if self._clean_tag(c.tag) == tag]
                    if elem in siblings:
                        index = siblings.index(elem) + 1
                    # Sinon utiliser le compteur
            
                current_path = f"{parent_path}/{tag}[{index}]"
            
            # Si c'est un tag cible
            if tag in target_tags:
                text = self._extract_element_text(elem)
                if text and len(text.strip()) > 10:
                    element_id = f"{self.doc_id}_{hashlib.md5(current_path.encode()).hexdigest()[:8]}"
                    
                    elements.append({
                        'elem_id': element_id,
                        'doc_id': self.doc_id,
                        'tag': tag,
                        'text': text.strip(),
                        'xml_path': current_path,
                        'source_file': self.xml_path,
                        'depth': level
                    })
            
            # Explorer les enfants (limité)
            if level < 10:
                for child in elem:
                    extract_elements(child, current_path, level + 1)
        
        extract_elements(self.root)
        return elements

    def _build_simple_xpath(self, elem):
        """Construit un XPath simple, en ignorant les niveaux intermédiaires bizarres"""
        path_parts = []
        current = elem
        
        while current is not None:
            tag = self._clean_tag(current.tag)
            
            # Ne garder que les balises importantes
            if tag in ['article', 'bdy', 'sec', 'p']:
                # Compter l'index
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
        
        if not path_parts:
            return "/article[1]"
        
        return '/' + '/'.join(path_parts)

    def get_basic_elements(self, target_tags: Set[str]) -> List[Dict]:
        """Version basique mais fonctionnelle pour l'exercice 3"""
        elements = []
        
        if self.root is None:
            return elements
        
        # Compter les éléments par tag pour générer des indices
        element_counters = {'bdy': 0, 'sec': 0, 'p': 0}
        
        def collect_elements(elem, level=0):
            tag = self._clean_tag(elem.tag)
            
            # Ne traiter que les tags cibles
            if tag in target_tags:
                text = self._extract_element_text(elem)
                if text and len(text.strip()) > 20:
                    # Générer un chemin simple
                    element_counters[tag] += 1
                    
                    if tag == 'bdy':
                        xml_path = "/article[1]/bdy[1]"
                    elif tag == 'sec':
                        xml_path = f"/article[1]/bdy[1]/sec[{element_counters['sec']}]"
                    elif tag == 'p':
                        # Essayer de déterminer si c'est dans une section ou directement dans bdy
                        parent = elem.getparent()
                        parent_tag = self._clean_tag(parent.tag) if parent is not None else ""
                        
                        if parent_tag == 'sec':
                            # Trouver l'index de la section parent
                            grandparent = parent.getparent() if parent is not None else None
                            if grandparent is not None:
                                sec_siblings = [c for c in grandparent if self._clean_tag(c.tag) == 'sec']
                                try:
                                    sec_index = sec_siblings.index(parent) + 1
                                except ValueError:
                                    sec_index = 1
                                xml_path = f"/article[1]/bdy[1]/sec[{sec_index}]/p[{element_counters['p']}]"
                            else:
                                xml_path = f"/article[1]/bdy[1]/p[{element_counters['p']}]"
                        else:
                            xml_path = f"/article[1]/bdy[1]/p[{element_counters['p']}]"
                    else:
                        xml_path = f"/article[1]/{tag}[1]"
                    
                    element_id = f"{self.doc_id}_{hashlib.md5(xml_path.encode()).hexdigest()[:8]}"
                    
                    elements.append({
                        'elem_id': element_id,
                        'doc_id': self.doc_id,
                        'tag': tag,
                        'text': text.strip(),
                        'xml_path': xml_path,
                        'source_file': self.xml_path,
                        'depth': level
                    })
            
            # Explorer les enfants (limité)
            if level < 8:
                for child in elem:
                    collect_elements(child, level + 1)
        
        collect_elements(self.root)
        return elements

    def get_inex_elements(self, target_tags: Set[str]) -> List[Dict]:
        """
        Version améliorée qui inclut les ARTICLES ENTIERS
        et gère mieux la granularité INEX
        """
        elements = []
        
        if self.root is None:
            return elements
        
        # Tags principaux pour le chemin XPath
        main_tags = {'article', 'bdy', 'sec', 'p'}
        
        # Compter les éléments par article pour les indices
        element_counter = {}
        
        def build_inex_path(elem, include_article=True):
            """Construit un chemin INEX-compatible"""
            path_parts = []
            current = elem
            
            while current is not None:
                tag = self._clean_tag(current.tag)
                
                # On garde seulement les balises principales
                if tag in main_tags:
                    # Trouver l'index parmi les frères
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
                    # Ne pas remonter plus haut que l'article
                    break
            
            if not path_parts:
                return "/article[1]"
            
            return '/' + '/'.join(path_parts)
        
        def process_element(elem, level=0):
            """Traite un élément XML récursivement"""
            tag = self._clean_tag(elem.tag)
            
            # Vérifier si c'est une balise cible OU un article
            if tag in target_tags or tag == 'article':
                text = self._extract_element_text(elem)
                
                # Critères d'acceptation différents selon le tag
                should_include = False
                
                if tag == 'article':
                    # Article : accepter s'il a du texte significatif
                    should_include = (text and len(text.strip()) > 300) # 100
                elif tag == 'bdy':
                    # Body : au moins 50 caractères
                    should_include = (text and len(text.strip()) > 150) # 50
                elif tag == 'sec':
                    # Section : au moins 30 caractères
                    should_include = (text and len(text.strip()) > 90) # 30
                elif tag == 'p':
                    # Paragraphe : au moins 10 caractères
                    should_include = (text and len(text.strip()) > 90) # 10
                
                if should_include:
                    xml_path = build_inex_path(elem, include_article=(tag != 'article'))
                    
                    # Vérifier que le chemin est valide (pas trop profond)
                    if xml_path.count('/') <= 8:
                        # Générer un ID unique
                        element_id = f"{self.doc_id}_{hashlib.md5(xml_path.encode()).hexdigest()[:8]}"
                        """
                        elements.append({
                            'elem_id': element_id,
                            'doc_id': self.doc_id,
                            'tag': tag,
                            'text': text.strip(),
                            'xml_path': xml_path,
                            'source_file': self.xml_path,
                            'depth': level,
                            'is_article': (tag == 'article')
                        })"""
                        elements.append({
                            'elem_id': element_id,
                            'doc_id': self.doc_id,
                            'tag': tag,  # IMPORTANT: 'p', 'sec', 'bdy', ou 'article'
                            'text': text.strip(),
                            'xml_path': xml_path,
                            'source_file': self.xml_path,
                            'depth': level,
                            'is_article': (tag == 'article'),
                            'priority': self._get_tag_priority(tag)  # Nouveau champ
                        })
            
            # Explorer les enfants (limité en profondeur)
            if level < 12:
                for child in elem:
                    process_element(child, level + 1)
        
        # Commencer le traitement
        process_element(self.root)
        
        # Éliminer les doublons (même chemin)
        unique_elements = []
        seen_paths = set()
        
        for elem in elements:
            if elem['xml_path'] not in seen_paths:
                seen_paths.add(elem['xml_path'])
                unique_elements.append(elem)
        
        # Trier : articles d'abord, puis par profondeur
        unique_elements.sort(key=lambda x: (0 if x['is_article'] else 1, x['depth'], x['xml_path']))
        
        return unique_elements

    # Ajouter cette méthode :
    def _get_tag_priority(self, tag):
        """Retourne la priorité du tag (plus haut = plus prioritaire)"""
        priorities = {
            'p': 4,
            'sec': 3, 
            'bdy': 2,
            'article': 1
        }
        return priorities.get(tag, 0)

    def _build_xpath(self, elem):
        """Construit un XPath avec indices corrects [1], [2], etc."""
        if elem is self.root:
            return f"/{self._clean_tag(elem.tag)}[1]"
        
        # Obtenir le chemin du parent
        parent = elem.getparent()
        parent_path = self._build_xpath(parent) if parent is not None else ""
        
        # Nettoyer le tag
        tag = self._clean_tag(elem.tag)
        
        # CORRECTION: Compter correctement les indices
        # Trouver tous les enfants du parent avec le même tag
        if parent is not None:
            # Récupérer tous les enfants directs du parent
            siblings = []
            for child in parent:
                if child is elem:
                    # On ajoute l'élément lui-même
                    siblings.append(child)
                    continue
                if self._clean_tag(child.tag) == tag:
                    siblings.append(child)
            
            # Maintenant on compte: l'élément actuel + ses frères déjà ajoutés
            # Mais attention: l'ordre dans siblings n'est pas forcément l'ordre du document
            
            # MÉTHODE PLUS ROBUSTE: compter en parcourant les enfants dans l'ordre
            count = 0
            index = 1
            for child in parent:
                if self._clean_tag(child.tag) == tag:
                    count += 1
                    if child is elem:
                        index = count
            
            xpath = f"{parent_path}/{tag}[{index}]"
        else:
            xpath = f"/{tag}[1]"
        
        return xpath

    def _clean_tag(self, tag):
        """Nettoie le tag du namespace"""
        if isinstance(tag, str):
            if '}' in tag:
                return tag.split('}', 1)[1]
        return str(tag)

    def _extract_element_text(self, elem) -> str:
        """Extrait tout le texte d'un élément (y compris sous-éléments)"""
        try:
            if LXML_AVAILABLE:
                # Avec lxml: utiliser itertext() pour tout le texte
                text_parts = []
                for t in elem.itertext():
                    if t and t.strip():
                        text_parts.append(t.strip())
                return ' '.join(text_parts)
            else:
                # Avec ElementTree: méthode basique
                text = elem.text or ''
                for child in elem:
                    text += ' ' + (self._extract_element_text(child) or '')
                # Nettoyage
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
        except Exception as e:
            print(f"Erreur extraction texte: {e}")
            return ""
        

class WeightedInvertedIndex:
    """Index inversé optimisé pour INEX"""
    
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
        self.xml_cache = {}
        

    # === MÉTHODES DE CONFIGURATION MANQUANTES ===

    def configure_tokenization(self, method="basic"):
        """Configure la méthode de tokenization"""
        self.tokenization_method = method
        print(f"- Tokenization configurée: {method}")

    def configure_stemmer(self, stemmer_name="nostem"):
        """Configure l'algorithme de stemming"""
        self.stemmer_name = stemmer_name
        
        if stemmer_name == "nostem":
            self.stemmer_func = None
        elif stemmer_name == "porter":
            stemmer = PorterStemmer()
            self.stemmer_func = lambda word: stemmer.stem(word, 0, len(word) - 1)
        elif stemmer_name == "snowball":
            self.stemmer_func = stem_word
        else:
            print(f"Stemmer '{stemmer_name}' non supporté, utilisation de 'nostem'")
            self.stemmer_func = None
        print(f"- Stemmer configuré: {stemmer_name}")

    def configure_stop_words(self, stop_list_name="nostop"):
        """Configure la liste de stop-words"""
        self.stop_list_name = stop_list_name
        if stop_list_name != "nostop":
            self._load_stop_words(stop_list_name)
        print(f"- Stop-words configurés: {stop_list_name}")

    def _load_stop_words(self, stop_list_name="stop671"):
        """Charge différentes listes de stop-words"""
        stop_files = {
            "stop671": "data/stopwords/stop-words-english4.txt",
            "stop319": "data/stopwords/stop-words-english5.txt",
            "stop733": "data/stopwords/stop-words-kaggle.txt"
        }
        
        file_path = stop_files.get(stop_list_name, "data/stopwords/stop-words-english4.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"- {len(self.stop_words_set)} stop words chargés depuis {stop_list_name}")
        except FileNotFoundError:
            print(f"- Fichier {file_path} non trouvé, utilisation liste vide")
            self.stop_words_set = set()

    # === MÉTHODES DE TOKENIZATION (déjà partiellement présentes) ===

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
            tokens = [token for token in tokens if token not in self.stop_words_set]

        # Stemming
        if self.stemmer_func:
            tokens = [self.stemmer_func(token) for token in tokens]

        return tokens
    
    def store_metadata(self, doc_id, metadata_dict):
        """Stocke les métadonnées pour un document/élément"""
        # S'assurer que parent_doc_id existe pour les éléments
        if self.doc_type == "element" and 'parent_doc_id' not in metadata_dict:
            if '_' in doc_id:
                metadata_dict['parent_doc_id'] = doc_id.split('_')[0]
            else:
                metadata_dict['parent_doc_id'] = doc_id
        
        self.metadata_store[doc_id] = metadata_dict

    def get_metadata(self, doc_id):
        """Récupère les métadonnées d'un document/élément"""
        metadata = self.metadata_store.get(doc_id, {
            'doc_id': doc_id,
            'xml_path': '/article[1]',
            'type': self.doc_type,
            'tag': 'article' if self.doc_type == 'article' else 'unknown'
        })
        
        # CORRECTION: Toujours s'assurer que parent_doc_id existe
        if 'parent_doc_id' not in metadata:
            if self.doc_type == 'element' and '_' in doc_id:
                metadata['parent_doc_id'] = doc_id.split('_')[0]
            else:
                metadata['parent_doc_id'] = doc_id
        
        # S'assurer que type est défini
        if 'type' not in metadata:
            metadata['type'] = 'element' if '_' in doc_id else 'article'
        
        return metadata





    def configure(self, tokenization="basic", stemmer="nostem", stop_words="nostop", **kwargs):
        """Configure tous les paramètres d'un coup avec arguments optionnels"""
        self.configure_tokenization(tokenization)
        self.configure_stemmer(stemmer)
        self.configure_stop_words(stop_words)
        
        # Gérer les arguments optionnels
        if 'target_tags' in kwargs:
            self.target_tags = kwargs['target_tags']
    
    def _index_document_content(self, doc_id: str, text: str, metadata: Dict = None) -> bool:
        """Indexe un contenu texte (commun aux articles et éléments)"""
        # Tokenization
        tokens = self.apply_tokenization(text)
        
        # Statistiques tokens
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
    
    def build_index_from_articles(self, xml_dir: str, max_files: Optional[int] = None) -> float:
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
            
            if text and len(text) > 50:
                doc_id = doc.doc_id
                
                # Indexer
                if self._index_document_content(doc_id, text):
                    # Stocker le document XML dans le cache pour phase BROWSE
                    self.xml_cache[doc_id] = doc
                    
                    # Métadonnées
                    self.store_metadata(doc_id, {
                        'doc_id': doc_id,
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
        print(f"Indexation FETCH terminée: {self.doc_count} articles en {indexing_time:.2f}s")
        return indexing_time
    
    def _extract_full_article_text(self, root) -> str:
        """Extrait tout le texte pertinent d'un article"""
        text_parts = []
        
        def collect_text(elem):
            if hasattr(elem, 'tag'):
                tag = elem.tag if hasattr(elem.tag, 'strip') else str(elem.tag)
                if '}' in tag:
                    tag = tag.split('}', 1)[1]
                
                # Ignorer certaines balises
                if tag in ['link', 'image', 'caption']:
                    return
                
                # Ajouter le texte
                if elem.text:
                    text_parts.append(elem.text.strip())
                
                # Explorer enfants
                for child in elem:
                    collect_text(child)
                
                # Ajouter tail
                if elem.tail:
                    text_parts.append(elem.tail.strip())
        
        collect_text(root)
        return ' '.join(text_parts)
    

    def build_index_from_elements(self, xml_dir: str, 
                                target_tags: List[str] = ['sec', 'p', 'bdy'],
                                max_files: Optional[int] = None) -> float:
        """Indexe les éléments individuels pour phase BROWSE"""
        print(f"Indexation BROWSE des éléments {target_tags}...")
        start_time = time.time()
        
        self.doc_type = "element"
        self.target_tags = set(target_tags)
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        total_elements = 0
        tag_set = set(target_tags)
        
        for i, xml_file in enumerate(xml_files):
            if i % 50 == 0:
                print(f"  Traitement fichier {i}/{len(xml_files)}...")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
                continue
            
            # Extraire tous les éléments cibles
            #elements = doc.get_all_elements(tag_set)
            elements = doc.get_inex_elements(list(tag_set))

            for elem_data in elements:
                elem_id = elem_data['elem_id']
                elem_text = elem_data['text']
                
                # CRITIQUE: S'assurer que parent_doc_id est correctement stocké
                metadata = {
                    'doc_id': elem_data['doc_id'],  # ID de l'article
                    'parent_doc_id': elem_data['doc_id'],  # MÊME QUE doc_id
                    'element_id': elem_id,
                    'xml_path': elem_data['xml_path'],
                    'tag': elem_data['tag'],
                    'type': 'element',
                    'source_file': xml_file,
                    'depth': elem_data.get('depth', 0)
                }
                
                if self._index_document_content(elem_id, elem_text, metadata):
                    total_elements += 1
        
        self.doc_count = total_elements
        if self.doc_count > 0:
            self.avg_doc_length = self.total_terms / self.doc_count
        
        indexing_time = time.time() - start_time
        print(f"Indexation BROWSE terminée: {total_elements} éléments en {indexing_time:.2f}s")
        return indexing_time

    def get_article_elements(self, article_id: str, 
                            target_tags: List[str] = None) -> List[Dict]:
        """Récupère tous les éléments d'un article (pour phase BROWSE dynamique)"""
        if target_tags is None:
            target_tags = ['sec', 'p']
        
        elements = []
        if article_id in self.xml_cache:
            doc = self.xml_cache[article_id]
            elements = doc.get_all_elements(set(target_tags))
        
        return elements
    
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
    
    def save_to_file(self, filename: str):
        """Sauvegarde l'index dans un fichier (sans les objets lxml)"""
        # Créer une copie des données sans le cache XML
        data_to_save = {
            'dictionary': dict(self.dictionary),
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'doc_count': self.doc_count,
            'total_terms': self.total_terms,
            'metadata_store': self.metadata_store,
            'config': {
                'stop_list_name': self.stop_list_name,
                'stemmer_name': self.stemmer_name,
                'tokenization_method': self.tokenization_method
            },
            'xml_cache_keys': list(self.xml_cache.keys())  # Sauvegarder seulement les clés
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                
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
                'tokenization_method': self.tokenization_method
            },
            # Ne pas inclure xml_cache car contient des objets lxml non sérialisables
        }


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
        
        return index