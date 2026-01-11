import re
import os
import time
import hashlib
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Optional, Union
import unicodedata
import html

# IMPORT LXML pour parsing XML correct (optionnel)
try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as ET

from unidecode import unidecode
from porterstemmer import PorterStemmer
from snowballstemmer import stem_word


class INEXDocument:
    """Classe pour représenter un document INEX avec son arbre XML"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.doc_id = None
        self.tree = None
        self.root = None

    def parse(self, use_lxml: bool = True):
        """Parse le document XML avec correction des entités avant parsing"""
        try:
            # 1. Lire le fichier
            with open(self.xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 2. CORRECTION DES ENTITÉS AVANT PARSING
            content = self._preprocess_xml_entities(content)

            # 3. Parser le contenu corrigé
            if use_lxml and LXML_AVAILABLE:
                parser = etree.XMLParser(recover=True, remove_comments=True)
                # Parser depuis la string, pas depuis le fichier
                self.tree = etree.fromstring(content.encode('utf-8'), parser)
            else:
                # Pour ElementTree
                self.tree = ET.fromstring(content)

            self.root = self.tree  # Note: avec fromstring(), tree EST root
            self.doc_id = self._extract_doc_id()
            return True

        except Exception as e:
            print(f"Erreur parsing {self.xml_path}: {e}")
            # Fallback: essayer sans correction
            try:
                if use_lxml and LXML_AVAILABLE:
                    parser = etree.XMLParser(recover=True, remove_comments=True)
                    self.tree = etree.parse(self.xml_path, parser)
                else:
                    self.tree = ET.parse(self.xml_path)

                self.root = self.tree.getroot()
                self.doc_id = self._extract_doc_id()
                return True
            except:
                return False

    @staticmethod
    def _preprocess_xml_entities(content: str) -> str:
        """Corrige les entités XML problématiques avant parsing"""
        # Liste des entités qui causent des erreurs
        corrections = {
            '&rsaquo;': '&#8250;',  # guillemet simple fermant
            '&lsaquo;': '&#8249;',  # guillemet simple ouvrant
            '&middot;': '&#183;',  # point médian
            # Ajoute d'autres si besoin
            '&rdquo;': '&#8221;',
            '&ldquo;': '&#8220;',
            '&rsquo;': '&#8217;',
            '&lsquo;': '&#8216;',
        }

        for wrong, correct in corrections.items():
            content = content.replace(wrong, correct)

        return content
    
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
                
        # 2. Extraire du nom de fichier
        filename = os.path.basename(self.xml_path)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return max(numbers, key=len)
        
        # 3. Fallback: hash du chemin
        return hashlib.md5(self.xml_path.encode()).hexdigest()[:8]
    
    def get_inex_elements(self, target_tags: Set[str]) -> List[Dict]:
        """
        Extrait les éléments XML pertinents pour INEX avec des seuils optimisés
        Deux méthodes disponibles : DOM (lxml) ou regex (fallback)
        """
        if LXML_AVAILABLE:
            return self._get_elements_dom(target_tags)
        else:
            return self._get_elements_regex(target_tags)

    def _get_elements_dom(self, target_tags: Set[str]) -> List[Dict]:
        """Version DOM (lxml) pour extraction précise - CORRIGÉE"""
        elements = []
        
        if self.root is None:
            return elements
        
        def build_inex_path(elem):
            """Construit un chemin XPath INEX-compatible - CORRIGÉE"""
            path_parts = []
            current = elem
            
            while current is not None:
                tag = self._clean_tag(current.tag)
                
                #  SEULEMENT les tags qui nous intéressent
                if tag in {'article', 'bdy', 'sec', 'p'}:
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
        
        def extract_meaningful_text(elem) -> str:
            #Extrait le texte en évitant les small_err_nodes
            try:
                text_parts = []
                for t in elem.itertext():
                    if t:
                        cleaned = t.strip()
                        if cleaned and len(cleaned) > 3:  # Ignorer mots isolés
                            text_parts.append(cleaned)
                result = ' '.join(text_parts)
                result = re.sub(r'\s+', ' ', result)

                result = INEXDocument.clean_and_normalize_text(result)
                return result.strip()
            except Exception:
                return ""
        
        def is_valid_content(text: str, tag: str) -> bool:
            #Valide le contenu selon le type d'élément
            if not text or not text.strip():
                return False
            
            cleaned = text.strip()
            
            # Seuils minimaux par tag (augmentés pour éviter small_err_nodes)
            min_lengths = {
                'p': 10,     # 20 caractères minimum
                'sec': 20,   # 30 caractères  
                'bdy': 30    # 50 caractères
            }
            
            min_chars = min_lengths.get(tag, 20)
            
            # Condition UNIQUE
            if len(cleaned) < min_chars:
                return False
            
            return True
        
        def process_element(elem, level=0):
            """Traite un élément XML récursivement - CORRIGÉE"""
            tag = self._clean_tag(elem.tag)
            
            # CORRECTION : SEULEMENT les tags cibles (pas 'article' !)
            if tag in target_tags:  # ← Retirer "or tag == 'article'"
                text = extract_meaningful_text(elem)
                
                if is_valid_content(text, tag):
                    xml_path = build_inex_path(elem)
                    
                    # S'assurer que le chemin ne soit PAS juste /article[1]
                    if xml_path != '/article[1]' and xml_path.count('/') <= 8:
                        element_id = f"{self.doc_id}_{hashlib.md5(xml_path.encode()).hexdigest()[:8]}"
                        
                        elements.append({
                            'elem_id': element_id,
                            'doc_id': self.doc_id,
                            'tag': tag,
                            'text': text,
                            'xml_path': xml_path,
                            'source_file': self.xml_path,
                            'depth': level,
                            'priority': self._get_tag_priority(tag),
                            'char_count': len(text)
                        })
            
            # Explorer les enfants avec limite de profondeur
            if level < 12:
                for child in elem:
                    process_element(child, level + 1)
        
        process_element(self.root)
        
        # Filtrer pour enlever tout élément qui serait juste /article[1]
        filtered_elements = []
        for elem in elements:
            if elem['xml_path'] != '/article[1]':
                filtered_elements.append(elem)
        
        return filtered_elements

    def _get_elements_regex(self, target_tags: Set[str]) -> List[Dict]:
        #Version regex (fallback) pour extraction sans lxml
        try:
            with open(self.xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extraire l'ID de l'article
            doc_id = self._extract_doc_id_from_content(content)
            
            if not doc_id:
                doc_id = os.path.basename(self.xml_path).replace('.xml', '')
            
            elements = []
            
            # Fonction pour extraire récursivement les balises
            def extract_tag_content(tag_name, text, parent_path="", level=0):
                #Extrait récursivement le contenu d'une balise
                pattern = fr'<{tag_name}[^>]*>((?:(?!<{"|".join(target_tags)}>).)*?)</{tag_name}>'
                matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
                
                for i, match in enumerate(matches):
                    full_content = match.group(0)
                    inner_content = match.group(1)
                    
                    # Chemin XML approximatif
                    elem_index = i + 1
                    current_path = f"{parent_path}/{tag_name}[{elem_index}]" if parent_path else f"/{tag_name}[{elem_index}]"
                    
                    # Extraire le texte
                    text_content = self._remove_balises(inner_content)
                    text_content = self._clean_html_entities(text_content)
                    
                    text_content = INEXDocument.clean_and_normalize_text(text_content)
                    
                    if text_content and len(text_content) > 10:
                        if current_path and current_path != '/article[1]':
                            # ID unique
                            path_hash = hashlib.md5(current_path.encode()).hexdigest()[:8]
                            element_id = f"{doc_id}_{path_hash}"
                            
                            elements.append({
                                'elem_id': element_id,
                                'doc_id': doc_id,
                                'tag': tag_name,
                                'text': text_content,
                                'xml_path': current_path,
                                'full_path': f"/article[1]{current_path}",
                                'file_path': self.xml_path,
                                'priority': self._get_tag_priority(tag_name),
                                'char_count': len(text_content)
                            })
                        
                    # Explorer récursivement les balises cibles à l'intérieur
                    if level < 3:
                        for target in target_tags:
                            if target != tag_name:
                                extract_tag_content(target, inner_content, current_path, level + 1)
            
            # Rechercher le corps principal
            bdy_match = re.search(r'<bdy[^>]*>(.*?)</bdy>', content, re.DOTALL | re.IGNORECASE)
            if bdy_match:
                bdy_content = bdy_match.group(1)
                for tag in target_tags:
                    extract_tag_content(tag, bdy_content, "/bdy[1]")
            else:
                article_match = re.search(r'<article[^>]*>(.*?)</article>', content, re.DOTALL | re.IGNORECASE)
                if article_match:
                    article_content = article_match.group(1)
                    for tag in target_tags:
                        extract_tag_content(tag, article_content, "/article[1]")
                else:
                    for tag in target_tags:
                        extract_tag_content(tag, content)
            
            # Éliminer les doublons
            unique_elements = []
            seen_texts = set()
            
            for elem in elements:
                text_hash = hashlib.md5(elem['text'].encode()).hexdigest()
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    unique_elements.append(elem)
            
            return unique_elements
            
        except Exception as e:
            print(f"Erreur regex {self.xml_path}: {e}")
            return []

    def _extract_doc_id_from_content(self, content: str) -> Optional[str]:
        """Extrait l'ID depuis le contenu brut"""
        id_match = re.search(r'<title>.*?</title>\s*<id>(\d+)</id>', content)
        return id_match.group(1) if id_match else None
    
    def _remove_balises(self, content: str) -> str:
        """Supprime toutes les balises XML/HTML"""
        removed_balises_without_space = ["link", "/link", "it", "/it", "/weblink"]
        text_content = content
        for balise in removed_balises_without_space:
            text_content = re.sub(rf'<{balise}[^>]*>', '', text_content)
        text = re.sub(r'<[^>]+>', ' ', text_content)
        return unidecode(text)

    @staticmethod
    def _clean_html_entities(text: str) -> str:
        """
        Nettoie TOUTES les entités HTML de manière exhaustive
        """
        if not text:
            return ""

        # Dictionnaire COMPLET des entités
        entity_map = {
            # Entités XML de base
            '&nbsp;': ' ', '&amp;': '&', '&apos;': "'", '&quot;': '"',
            '&lt;': '<', '&gt;': '>',

            # Ponctuation
            '&ndash;': '–', '&mdash;': '—', '&hellip;': '...',
            '&middot;': '·', '&bull;': '•',

            # Guillemets
            '&ldquo;': '"', '&rdquo;': '"',
            '&lsquo;': "'", '&rsquo;': "'",
            '&laquo;': '"', '&raquo;': '"',
            '&lsaquo;': '‹', '&rsaquo;': '›',  # <-- LES PROBLÉMATIQUES

            # Espaces
            '&ensp;': ' ', '&emsp;': '    ', '&thinsp;': ' ',

            # Divers
            '&zwnj;': '', '&zwj;': '', '&lrm;': '', '&rlm;': '',
            '&copy;': '(c)', '&reg;': '(R)', '&trade;': '(TM)',
            '&euro;': '€', '&pound;': '£', '&cent;': '¢',
            '&deg;': '°', '&plusmn;': '±', '&times;': '×', '&divide;': '÷',

            # Lettres accentuées communes
            '&agrave;': 'à', '&aacute;': 'á', '&acirc;': 'â', '&atilde;': 'ã',
            '&egrave;': 'è', '&eacute;': 'é', '&ecirc;': 'ê', '&etilde;': 'ẽ',
            '&igrave;': 'ì', '&iacute;': 'í', '&icirc;': 'î', '&itilde;': 'ĩ',
            '&ograve;': 'ò', '&oacute;': 'ó', '&ocirc;': 'ô', '&otilde;': 'õ',
            '&ugrave;': 'ù', '&uacute;': 'ú', '&ucirc;': 'û', '&utilde;': 'ũ',
            '&yacute;': 'ý', '&yuml;': 'ÿ',

            # Majuscules accentuées
            '&Agrave;': 'À', '&Aacute;': 'Á', '&Acirc;': 'Â', '&Atilde;': 'Ã',
            '&Egrave;': 'È', '&Eacute;': 'É', '&Ecirc;': 'Ê', '&Etilde;': 'Ẽ',
            '&Igrave;': 'Ì', '&Iacute;': 'Í', '&Icirc;': 'Î', '&Itilde;': 'Ĩ',
            '&Ograve;': 'Ò', '&Oacute;': 'Ó', '&Ocirc;': 'Ô', '&Otilde;': 'Õ',
            '&Ugrave;': 'Ù', '&Uacute;': 'Ú', '&Ucirc;': 'Û', '&Utilde;': 'Ũ',
            '&Yacute;': 'Ý',
        }

        # 1. D'abord remplacer les entités connues
        for entity, replacement in entity_map.items():
            text = text.replace(entity, replacement)

        # 2. Ensuite gérer les entités numériques (&#xxx; et &#xhhh;)
        # Entités décimales
        def replace_decimal(match):
            try:
                code = int(match.group(1))
                # Filtrer les caractères de contrôle
                if code < 32 or code == 127:
                    return ' '
                return chr(code)
            except:
                return ' '

        text = re.sub(r'&#(\d+);', replace_decimal, text)

        # Entités hexadécimales
        def replace_hex(match):
            try:
                code = int(match.group(1), 16)
                if code < 32 or code == 127:
                    return ' '
                return chr(code)
            except:
                return ' '

        text = re.sub(r'&#x([0-9a-fA-F]+);', replace_hex, text)

        # 3. Décoder les entités HTML standard
        try:
            text = html.unescape(text)
        except:
            pass

        # 4. Supprimer les caractères de contrôle restants
        control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
        for char in control_chars:
            text = text.replace(char, ' ')

        # 5. Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    
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
    


    def extract_full_article_text(self) -> str:
        """
        Extrait TOUT le texte de l'article avec nettoyage complet
        Similaire à l'ancienne méthode parse_xml_file() mais avec lxml
        """
        if self.root is None:
            return ""
        
        try:
            # Méthode 1: Avec lxml (plus propre)
            if LXML_AVAILABLE:
                # Récupérer tout le texte avec itertext()
                all_text_parts = []
                for t in self.root.itertext():
                    if t and t.strip():
                        cleaned = self._clean_text_segment(t)
                        if cleaned:
                            all_text_parts.append(cleaned)
                
                full_text = ' '.join(all_text_parts)
            
            # Méthode 2: Fallback regex (comme avant)
            else:
                with open(self.xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Supprimer toutes les balises XML
                text_content = self._remove_all_xml_tags(content)
                full_text = self._clean_html_entities(text_content)
            
            # Normalisation finale
            full_text = self._normalize_text(full_text)
            
            return full_text
            
        except Exception as e:
            print(f"Erreur extraction texte article {self.xml_path}: {e}")
            return ""
    
    def _clean_text_segment(self, text: str) -> str:
        """Nettoie un segment de texte"""
        if not text:
            return ""
        
        # Nettoyer les entités HTML
        text = self._clean_html_entities(text)
        
        # Supprimer les caractères de contrôle
        control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
        for char in control_chars:
            text = text.replace(char, ' ')
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _remove_all_xml_tags(self, content: str) -> str:
        """Supprime TOUTES les balises XML (version regex comme avant)"""
        # Garder le contenu des balises de texte importantes
        text_parts = []
        
        # D'abord, supprimer les balises non-textuelles
        non_text_tags = ['link', 'image', 'caption', 'ref', 'figure', 'table']
        for tag in non_text_tags:
            content = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Ensuite, supprimer toutes les balises restantes
        content = re.sub(r'<[^>]+>', ' ', content)
        
        return content
    
    @staticmethod
    def _clean_html_entities(text: str) -> str:
        """
        Nettoie les entités HTML - Version complète comme dans l'ancien code
        """
        if not text:
            return ""
        
        # 1. Décodage standard
        try:
            text = html.unescape(text)
        except:
            pass
        
        # 2. Remplacer les entités courantes manuellement
        entity_map = {
            '&nbsp;': ' ', '&amp;': '&', '&apos;': "'", '&quot;': '"',
            '&lt;': '<', '&gt;': '>', '&ndash;': '–', '&mdash;': '—',
            '&hellip;': '...', '&middot;': '·', '&bull;': '•',
            '&ldquo;': '"', '&rdquo;': '"', '&lsquo;': "'", '&rsquo;': "'",
            '&laquo;': '"', '&raquo;': '"', '&ensp;': ' ', '&emsp;': '    ',
            '&thinsp;': ' ', '&zwnj;': '', '&zwj;': '', '&lrm;': '', '&rlm;': '',
            '&lsaquo;': '‹', '&rsaquo;': '›'
        }
        
        for entity, replacement in entity_map.items():
            text = text.replace(entity, replacement)
        
        # 3. Gérer les entités numériques
        # Entités décimales
        def replace_decimal(match):
            try:
                char_code = int(match.group(1))
                if char_code < 32 or char_code == 127:
                    return ' '
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
        
        # 4. Supprimer les caractères de contrôle restants
        control_chars = ''.join(chr(i) for i in range(32)) + chr(127)
        for char in control_chars:
            text = text.replace(char, ' ')
        
        # 5. Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        #Normalisation finale du texte
        if not text:
            return ""
        
        # Supprimer les caractères non alphabétiques (sauf espaces)
        #text = re.sub(r'[^A-Za-z\s]', ' ', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def clean_and_normalize_text(text: str) -> str:
        """
        Nettoie et normalise n'importe quel texte (réutilisable)
        Pour les champs dans exercices 5-6
        """
        if not text:
            return ""
        
        # 1. Nettoyer les entités HTML
        cleaned = INEXDocument._clean_html_entities(text)
        
        # 2. Normaliser (supprimer non-alphabétique, espaces)
        cleaned = INEXDocument._normalize_text(cleaned)
        
        return cleaned

