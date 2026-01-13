import os
import time
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument

class SimpleFieldWeightedIndex:
    """Version SIMPLE de l'index avec pondération par champs"""
    
    def __init__(self):
        self.index = None
        self.field_tfs = defaultdict(lambda: defaultdict(dict))
        self.field_weights = {}
        self.doc_lengths = {}
        self.doc_ids = []
        self.df = {}
        self.avg_field_lengths = {}
        
    def build_index(self, xml_dir: str, 
                   fields_config: Dict[str, List[str]],
                   field_weights: Dict[str, float],
                   config: Dict) -> int:
        """
        Construit un index SIMPLE
        """
        print(f"Construction de l'index avec {len(fields_config)} champs...")
        
        # Créer l'index principal
        self.index = WeightedInvertedIndex()
        self.index.configure(**config)
        self.field_weights = field_weights
        
        # Lister les fichiers XML
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        print(f"Traitement de {len(xml_files)} fichiers...")
        
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"  {i}/{len(xml_files)}...")
            
            # Parser le document
            doc = INEXDocument(xml_file)
            if not doc.parse():
                continue
            
            doc_id = doc.doc_id
            self.doc_ids.append(doc_id)
            
            # Initialiser
            if doc_id not in self.field_tfs:
                self.field_tfs[doc_id] = defaultdict(dict)
            
            all_terms = []
            doc_length = 0
            
            # Extraire chaque champ
            for field_name, target_tags in fields_config.items():
                # Extraire le texte du champ
                field_text = self._extract_field_text_simple(doc, target_tags)
                
                if field_text:
                    # Tokenization
                    tokens = self.index.apply_tokenization(field_text)
                    terms = self.index.process_tokens(tokens)
                    
                    if terms:
                        # Stocker les fréquences
                        term_counts = Counter(terms)
                        for term, tf in term_counts.items():
                            self.field_tfs[doc_id][field_name][term] = float(tf)
                        
                        all_terms.extend(terms)
                        doc_length += len(terms)
            
            self.doc_lengths[doc_id] = doc_length
            
            # Mettre à jour l'index global
            for term in set(all_terms):
                count = all_terms.count(term)
                if doc_id not in self.index.dictionary[term]:
                    self.index.dictionary[term][doc_id] = 0
                self.index.dictionary[term][doc_id] = count
        
        # Finaliser
        self.index.doc_ids = self.doc_ids
        self.index.doc_lengths = self.doc_lengths
        self.index.doc_count = len(self.doc_ids)
        self.index.total_terms = sum(self.doc_lengths.values())
        
        if self.index.doc_count > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        
        # Calculer DF
        for term, doc_dict in self.index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        # Calculer les longueurs moyennes par champ
        self._compute_avg_field_lengths()
        
        print(f"✅ Index construit: {len(self.doc_ids)} documents")
        return len(self.doc_ids)
    
    def _extract_field_text_simple(self, doc: INEXDocument, target_tags: List[str]) -> str:
        """Extraction SIMPLE du texte d'un champ"""
        all_text = []
        
        for tag in target_tags:
            # Rechercher tous les éléments avec ce tag
            elements = doc.root.findall(f'.//{tag}')
            for elem in elements:
                # Extraire le texte
                if elem.text and elem.text.strip():
                    all_text.append(elem.text.strip())
                # Texte des enfants
                for child in elem:
                    if child.text and child.text.strip():
                        all_text.append(child.text.strip())
                    if child.tail and child.tail.strip():
                        all_text.append(child.tail.strip())
        
        return ' '.join(all_text)
    
    def _compute_avg_field_lengths(self):
        """Calcule la longueur moyenne de chaque champ"""
        for field_name in self.field_weights.keys():
            total_length = 0
            doc_count = 0
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_length = sum(self.field_tfs[doc_id][field_name].values())
                    total_length += field_length
                    doc_count += 1
            
            self.avg_field_lengths[field_name] = total_length / doc_count if doc_count > 0 else 1
    
    def search_bm25fw_simple(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """BM25Fw SIMPLE et EFFICACE"""
        # Tokenization de la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        doc_scores = {}
        
        # Pour chaque document
        for doc_id in self.doc_ids:
            total_score = 0.0
            
            # Pour chaque champ
            for field_name, weight in self.field_weights.items():
                field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                
                if not field_tf_dict:
                    continue
                
                field_length = sum(field_tf_dict.values())
                avg_field_length = self.avg_field_lengths[field_name]
                
                field_score = 0.0
                
                # Calcul BM25 pour ce champ
                for term in query_terms:
                    tf = field_tf_dict.get(term, 0)
                    if tf > 0:
                        # IDF global (similaire à Practice 4)
                        df = self.df.get(term, 0)
                        if df > 0:
                            idf = math.log10((self.index.doc_count - df + 0.5) / (df + 0.5))
                            
                            # Composante TF avec normalisation de champ
                            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (field_length / avg_field_length)))
                            
                            field_score += idf * tf_component
                
                total_score += weight * field_score
            
            #if total_score > 0:
            #    doc_scores[doc_id] = total_score
            doc_scores[doc_id] = total_score
        
        # Trier
        return sorted(doc_scores.items(), key=lambda x: -x[1])[:1500]
    
    def search_bm25fr_simple(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """BM25Fr SIMPLE et EFFICACE"""
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        doc_scores = {}
        
        # Pour chaque document
        for doc_id in self.doc_ids:
            # Combinaison précoce des TF
            combined_tf = defaultdict(float)
            
            for term in query_terms:
                tf_star = 0.0
                for field_name, weight in self.field_weights.items():
                    field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                    tf_in_field = field_tf_dict.get(term, 0)
                    tf_star += weight * tf_in_field
                
                if tf_star > 0:
                    combined_tf[term] = tf_star
            
            # BM25 sur les TF combinés
            doc_score = 0.0
            for term, tf_star in combined_tf.items():
                df = self.df.get(term, 0)
                if df > 0:
                    # IDF (comme dans Practice 4)
                    idf = math.log10((self.index.doc_count - df + 0.5) / (df + 0.5))
                    
                    # Composante TF
                    doc_length = self.doc_lengths[doc_id]
                    tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (doc_length / self.index.avg_doc_length)))
                    
                    doc_score += idf * tf_component
            
            #if doc_score > 0:
            #    doc_scores[doc_id] = doc_score
            doc_scores[doc_id] = doc_score
        
        # Trier
        return sorted(doc_scores.items(), key=lambda x: -x[1])[:1500]

def generate_simple_field_run(run_id: str, run_type: str,
                            xml_dir: str, queries: Dict[int, str],
                            config: Dict,
                            fields_config: Dict = None,
                            field_weights: Dict = None,
                            k1: float = 1.2, b: float = 0.75) -> str:
    """
    Génère un run SIMPLE avec pondération par champs
    """
    print(f"\nGénération {run_type.upper()} - Run {run_id}")
    
    # Configuration par défaut
    if fields_config is None:
        fields_config = {
            'title': ['title'],
            'body': ['bdy'],
            'sections': ['sec'],
            'paragraphs': ['p']
        }
    
    if field_weights is None:
        field_weights = {
            'title': 1.0,  # Commencez avec des poids égaux
            'body': 1.0,
            'sections': 1.0,
            'paragraphs': 1.0
        }
    
    # Construire l'index
    index = SimpleFieldWeightedIndex()
    index.build_index(xml_dir, fields_config, field_weights, config)
    
    # Générer le fichier
    team_name = "AlphaAnaClement"
    filename = f"{team_name}_{run_id}_{run_type}_simple.txt"
    filepath = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"  Query {query_id}...")
            
            # Recherche
            if run_type == "bm25fw":
                results = index.search_bm25fw_simple(query_text, k1, b)
            else:  # bm25fr
                results = index.search_bm25fr_simple(query_text, k1, b)
            
            # Écrire
            for rank, (doc_id, score) in enumerate(results[:1500], 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")
    
    print(f"✅ Fichier généré: {filename}")
    return filepath