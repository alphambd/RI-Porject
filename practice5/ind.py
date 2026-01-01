import re
from collections import defaultdict, Counter
import gzip
import time
import xml.etree.ElementTree as ET
import os
import html
import hashlib
import unicodedata
from unidecode import unidecode

from porterstemmer import PorterStemmer
from snowballstemmer import stem_word

class WeightedInvertedIndex:
    def __init__(self):
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.doc_lengths = {}
        self.doc_count = 0
        self.total_terms = 0
        self.total_tokens_bp = 0
        self.distinct_tokens_bp = set()
        self.total_chars_tokens = 0
        self.avg_doc_length = 0

        # Configuration simplifiée - SUPPRESSION des attributs de compatibilité
        self.stop_words_set = set()
        self.stemmer_func = None
        self.tokenization_method = "basic"
        self.stop_list_name = "nostop"
        self.stemmer_name = "nostem"

        # Métadonnées pour la recherche structurée
        self.metadata_store = {}  # {doc_id: metadata_dict}
        self.doc_type = "article"  # "article" ou "element"
        self.target_tags = []  # Tags cibles pour éléments

    def store_metadata(self, doc_id, metadata_dict):
        """Stocke les métadonnées pour un document/élément"""
        self.metadata_store[doc_id] = metadata_dict

    def get_metadata(self, doc_id):
        """Récupère les métadonnées d'un document/élément"""
        return self.metadata_store.get(doc_id, {
            'doc_id': doc_id,
            'xml_path': '/article[1]',
            'type': self.doc_type,
            'tag': 'article' if self.doc_type == 'article' else 'unknown'
        })

    def get_parent_article_id(self, element_id):
        """Pour un élément, retourne l'ID de l'article parent"""
        metadata = self.get_metadata(element_id)
        return metadata.get('parent_doc_id', element_id.split('_')[0] if '_' in element_id else element_id)

    def get_xml_path(self, doc_id):
        """Retourne le chemin XML complet"""
        metadata = self.get_metadata(doc_id)
        return metadata.get('xml_path', '/article[1]')

    # === FONCTIONS DE TOKENIZATION ===

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

    # === FONCTIONS DE STEMMING ===

    def configure_stemmer(self, stemmer_name="nostem"):
        """Configure l'algorithme de stemming - VERSION SIMPLIFIÉE"""
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

    # === FONCTIONS DE STOP-WORDS ===

    def _load_stop_words(self, stop_list_name="stop671"):
        """Charge différentes listes de stop-words"""
        stop_files = {
            "stop671": "data/stopwords/stop-words-english4.txt",
            "stop319": "data/stopwords/stop-words-english5.txt",
            "stop733": "data/stopwords/stop-words-kaggle.txt"
        }
        # TO HANDLE
        file_path = stop_files.get(stop_list_name, "data/stop-words-english4.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"- {len(self.stop_words_set)} stop words chargés depuis {stop_list_name}")
        except FileNotFoundError:
            print(f"- Fichier {file_path} non trouvé, utilisation liste vide")
            self.stop_words_set = set()

    def configure_stop_words(self, stop_list_name="nostop"):
        """Configure la liste de stop-words"""
        self.stop_list_name = stop_list_name
        if stop_list_name != "nostop":
            self._load_stop_words(stop_list_name)
        print(f"- Stop-words configurés: {stop_list_name}")

    def configure_tokenization(self, method="basic"):
        """Configure la méthode de tokenization"""
        self.tokenization_method = method
        print(f"- Méthode de tokenization configurée: {method}")

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

    def read_file(self, filename, is_zipped):
        """Renvoie le contenu d'un fichier zippé ou non"""
        if is_zipped:
            try:
                with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
            except Exception as e:
                print(f"- Erreur lecture: {e}")
                return None
        else:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        return content

    def parse_xml_file(self, xml_file_path):
        """Convertit un XML au format texte"""
        try:
            with open(xml_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extraire l'ID
            doc_id = None
            #id_match = re.search(r'<id>(\d+)</id>', content)
            id_match = re.search(r'<title>.*?</title>\s*<id>(\d+)</id>', content)

            if id_match:
                doc_id = id_match.group(1)
            else:
                doc_id = os.path.basename(xml_file_path).replace('.xml', '')

            # L'ancien code avait juste le texte entre <doc> et </doc>
            # Ici on doit supprimer toutes les balises
            text = self.remove_balise(content)

            # Normaliser les espaces (identique)
            text = re.sub(r'\s+', ' ', text).strip()

            # Nettoyer les entités
            text = self.clean_html_entities(text)

            # Retourner au format similaire à l'ancien
            return {
                'doc_id': doc_id,
                'doc_text': text  # Note: nom 'doc_text' pour correspondre à l'ancien code
            }

        except Exception as e:
            print(f"Erreur conversion format {xml_file_path}: {e}")
            return None

    @staticmethod
    def remove_balise(content):
        text_content = content

        removed_balises_without_space = ["link","/link","it","/it","/weblink"]
        text_content = re.sub(rf'<({"|".join(removed_balises_without_space)})>', '', text_content)
        text = re.sub(r'<[^>]+>', ' ', text_content)

        return unidecode(text)

    @staticmethod
    def clean_html_entities(text):
        """Version corrigée qui gère TOUTES les entités correctement"""
        
        # D'abord html.unescape
        #text = html.unescape(text)
        
        # Liste COMPLÈTE des entités 
        complete_entity_map = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&apos;': "'",
            '&quot;': '"',
            '&lt;': '<',
            '&gt;': '>',
            '&ndash;': '–',
            '&mdash;': '—',
            '&hellip;': '...',
            '&middot;': '·',
            '&bull;': '•',
            '&ldquo;': '"',
            '&rdquo;': '"',
            '&lsquo;': "'",
            '&rsquo;': "'",
            '&laquo;': '"',
            '&raquo;': '"',
            '&ensp;': ' ',
            '&emsp;': '    ',
            '&thinsp;': ' ',
            '&zwnj;': '',
            '&zwj;': '',
            '&lrm;': '',
            '&rlm;': '',
            '&lsaquo;': '‹',
            '&rsaquo;': '›'
        }
        
        # Remplacer toutes les entités nommées
        for entity, replacement in complete_entity_map.items():
            text = text.replace(entity, replacement)
        
        # Gérer les entités numériques RESTANTES
        # Certaines entités numériques ne sont pas décodées par html.unescape
        # parce qu'elles sont dans du texte qui a déjà été transformé
        
        # Entités décimales
        def replace_decimal(match):
            try:
                char_code = int(match.group(1))
                # Filtrer les caractères de contrôle (0-31) et DEL (127)
                if char_code < 32 or char_code == 127:
                    return ' '  # Remplacer par espace
                return chr(char_code)
            except:
                return ' '
        
        text = re.sub(r'&#(\d+);', replace_decimal, text)
        
        # Entités hexadécimales
        def replace_hex(match):
            try:
                char_code = int(match.group(1), 16)
                if char_code < 32 or char_code == 127:
                    return ' '
                return chr(char_code)
            except:
                return ' '
        
        text = re.sub(r'&#x([0-9a-fA-F]+);', replace_hex, text)
        
        # 5. IMPORTANT : Nettoyer les caractères de contrôle
        # Même après décodage, il peut rester des caractères de contrôle
        control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
        for char in control_chars:
            text = text.replace(char, ' ')
        
        return text.strip()
    





    def extract_xml_elements(self, xml_file_path, target_tags=None):
        """
        Phase BROWSE (fetch & browse – cours)
        Extrait les éléments XML candidats comme points d’accès INEX
        """
        # to handle
        if target_tags is None:
            target_tags = ["p", "sec", "bdy"]
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # --- Identifier l'article ---
            article = root.find(".//article")
            if article is None:
                return []

            # ID article
            id_node = article.find(".//id")
            doc_id = id_node.text.strip() if id_node is not None else os.path.basename(xml_file_path)

            results = []

            # --- Helper : texte direct (sans descendants) ---
            def get_own_text(elem):
                texts = []
                if elem.text:
                    texts.append(elem.text)
                for child in elem:
                    if child.tail:
                        texts.append(child.tail)
                text = " ".join(texts)
                text = self.clean_html_entities(self.remove_balise(text))
                return re.sub(r"\s+", " ", text).strip()

            # --- ARTICLE (fallback) ---
            results.append({
                "elem_id": f"{doc_id}_article",
                "doc_id": doc_id,
                "tag": "article",
                "role": "root",
                "text": "",  # pas indexé directement
                "xml_path": "/article[1]",
                "file_path": xml_file_path
            })

            # --- BODY ---
            bdy = article.find(".//bdy")
            if bdy is None:
                return results

            results.append({
                "elem_id": f"{doc_id}_bdy",
                "doc_id": doc_id,
                "tag": "bdy",
                "role": "context-global",
                "text": "",  # jamais indexé directement
                "xml_path": "/article[1]/bdy[1]",
                "file_path": xml_file_path
            })

            # --- SECTIONS ---
            for sec_idx, sec in enumerate(bdy.findall(".//sec"), start=1):
                sec_text = get_own_text(sec)

                sec_id = f"{doc_id}_sec_{sec_idx}"
                sec_path = f"/article[1]/bdy[1]/sec[{sec_idx}]"

                results.append({
                    "elem_id": sec_id,
                    "doc_id": doc_id,
                    "tag": "sec",
                    "role": "context",
                    "text": sec_text,
                    "xml_path": sec_path,
                    "file_path": xml_file_path
                })

                # --- PARAGRAPHS ---
                for p_idx, p in enumerate(sec.findall("./p"), start=1):
                    p_text = get_own_text(p)
                    if len(p_text) < 20:
                        continue

                    p_id = f"{doc_id}_sec{sec_idx}_p{p_idx}"
                    p_path = f"{sec_path}/p[{p_idx}]"

                    results.append({
                        "elem_id": p_id,
                        "doc_id": doc_id,
                        "tag": "p",
                        "role": "content",
                        "text": p_text,
                        "xml_path": p_path,
                        "file_path": xml_file_path
                    })

            return results

        except Exception as e:
            print(f"Erreur XML {xml_file_path}: {e}")
            return []







    def _get_xml_files(self, xml_dir, max_files=None):
        """Retourne la liste des fichiers XML d'un répertoire"""
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        if max_files is not None:
            xml_files = xml_files[:max_files]
        
        return xml_files

    def _index_document_content(self, doc_id, text_content, metadata=None):
        """Méthode générique pour indexer un contenu texte"""
        # Tokenization
        tokens = self.apply_tokenization(text_content)
        
        # Mise à jour des statistiques pour les TOKENS
        self.total_tokens_bp += len(tokens)
        self.distinct_tokens_bp.update(tokens)
        self.total_chars_tokens += sum(len(token) for token in tokens)
        
        # Traitement TOKENS pour obtenir les TERMS
        terms = self.process_tokens(tokens)
        
        if not terms:  # Document vide
            return False
        
        # Mise à jour des statistiques TERMS
        doc_length = len(terms)
        self.doc_ids.append(doc_id)
        self.doc_lengths[doc_id] = doc_length
        self.total_terms += doc_length
        
        # Construction du dictionnaire
        term_freq = Counter(terms)
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
        
        # Stocker les métadonnées si fournies
        if metadata:
            self.store_metadata(doc_id, metadata)
        
        return True
        
    def build_index_from_xml_collection(self, xml_dir, max_files=None):
        """Construit l'index à partir d'un répertoire de fichiers XML"""
        start_time = time.time()
        
        # Configurer le type de document
        self.doc_type = "article"
        
        # Lister les fichiers XML
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        success_count = 0
        
        for i, xml_file in enumerate(xml_files):
            # Parser le fichier XML
            doc_data = self.parse_xml_file(xml_file)
            if not doc_data:
                continue
            
            doc_id = doc_data['doc_id']
            doc_text = doc_data['doc_text']
            
            # Indexer avec la méthode générique
            if self._index_document_content(doc_id, doc_text):
                # Métadonnées pour les articles
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
        self.avg_doc_length = self.total_terms / self.doc_count if self.doc_count > 0 else 0
        
        end_time = time.time()
        return end_time - start_time
            
    def build_index_from_xml_elements(self, xml_dir, target_tags=("p", "sec", "bdy"), max_files=None):
        """
        Phase d’indexation des unités XML (browse units)
        - Chaque élément XML est une unité documentaire
        - Les éléments sont regroupés par article (fetch & browse)
        """

        import time
        from collections import defaultdict

        start_time = time.time()

        print(f"Indexation des éléments XML: {target_tags}")

        # Configuration
        self.doc_type = "element"
        self.target_tags = target_tags

        # STRUCTURE CLÉ POUR FETCH & BROWSE
        self.elements_by_article = defaultdict(list)

        xml_files = self._get_xml_files(xml_dir, max_files)
        total_elements = 0

        for xml_file in xml_files:
            elements = self.extract_xml_elements(xml_file, target_tags)

            for elem in elements:
                elem_id = elem["elem_id"]
                article_id = elem["doc_id"]
                text = elem["text"]

                # Indexation textuelle (inverted index)
                indexed = self._index_document_content(elem_id, text)

                if not indexed:
                    continue

                # --- MÉTADONNÉES (unité documentaire XML) ---
                self.store_metadata(elem_id, {
                    "doc_id": article_id,
                    "parent_doc_id": article_id,
                    "element_id": elem_id,
                    "xml_path": elem["full_path"],
                    "tag": elem["tag"],
                    "type": "element",
                    "source_file": elem["file_path"]
                })

                # --- STRUCTURE FETCH & BROWSE ---
                self.elements_by_article[article_id].append(elem_id)

                total_elements += 1

        self.doc_count = total_elements
        self.avg_doc_length = (
            self.total_terms / self.doc_count if self.doc_count > 0 else 0
        )

        elapsed = time.time() - start_time

        print(f"Indexation terminée : {total_elements} éléments")
        print(f"Temps d’indexation : {elapsed:.2f}s")

        return elapsed

    def get_collection_statistics(self, indexing_time):
        """Calcule TOUTES les statistiques demandées"""
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

        return {
            'indexing_time': indexing_time,
            'total_tokens': total_tokens,
            'distinct_tokens': distinct_tokens,
            'avg_token_length': avg_token_length,
            'total_terms': total_terms,
            'distinct_terms': distinct_terms,
            'avg_doc_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }