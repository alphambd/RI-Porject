import os
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument
from ranked_retrieval import RankedRetrieval

class SimpleFieldWeightedIndex:
    """Version qui utilise RankedRetrieval pour avoir le même comportement"""
    
    def __init__(self):
        self.index = None
        self.ranker = None
        self.field_tfs = defaultdict(lambda: defaultdict(dict))
        self.field_weights = {}
        self.avg_field_lengths = {}
        self.doc_ids = []
        self.doc_lengths = {}
        self.df = {}
        self.field_stats = {}
    
    def build_index(self, xml_dir: str, 
                   fields_config: Dict[str, List[str]],
                   field_weights: Dict[str, float],
                   config: Dict) -> int:
        """
        Construit un index avec pondération par champs
        """
        print(f"Construction de l'index avec {len(fields_config)} champs...")
        
        # Créer l'index principal
        self.index = WeightedInvertedIndex()
        self.index.configure(**config)
        self.field_weights = field_weights
        
        # Lister les fichiers XML
        xml_files = self._get_xml_files(xml_dir)
        
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
        
        # Finaliser l'index
        self.index.doc_ids = self.doc_ids
        self.index.doc_lengths = self.doc_lengths
        self.index.doc_count = len(self.doc_ids)
        self.index.total_terms = sum(self.doc_lengths.values())
        
        if self.index.doc_count > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        
        # Calculer DF
        for term, doc_dict in self.index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        # Calculer les statistiques de champ
        self._compute_field_stats()
        
        # Créer le ranker
        self.ranker = RankedRetrieval(self.index)
        
        print(f"✅ Index construit: {len(self.doc_ids)} documents")
        return len(self.doc_ids)
    
    def _get_xml_files(self, xml_dir: str) -> List[str]:
        """Liste les fichiers XML"""
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        return xml_files
    
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
    
    def _compute_field_stats(self):
        """Calcule les statistiques de chaque champ"""
        for field_name in self.field_weights.keys():
            total_length = 0
            doc_count = 0
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_length = sum(self.field_tfs[doc_id][field_name].values())
                    total_length += field_length
                    doc_count += 1
            
            self.avg_field_lengths[field_name] = total_length / doc_count if doc_count > 0 else 1
            
            # Calculer DF par champ
            field_dfs = defaultdict(int)
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    for term in self.field_tfs[doc_id][field_name].keys():
                        field_dfs[term] += 1
            
            self.field_stats[field_name] = {
                'avg_length': self.avg_field_lengths[field_name],
                'doc_count': doc_count,
                'dfs': dict(field_dfs)
            }
    
    # ==================== MÉTHODES DE RECHERCHE ====================
    
    def search_bm25fw_with_ranker(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        Utilise RankedRetrieval pour avoir EXACTEMENT le même comportement que Practice 4
        - Simple BM25 sans prise en compte des champs
        - Garantit toujours 1500 résultats
        """
        return self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
    
    def search_bm25fw_field_aware(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        Version avec prise en compte des champs MAIS toujours 1500 résultats
        Combine la logique des champs avec la garantie de RankedRetrieval
        """
        # D'abord, obtenir les résultats de base avec RankedRetrieval
        base_results = self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
        
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return base_results
        
        # Créer un dictionnaire pour ajuster les scores avec les champs
        adjusted_scores = {}
        
        # Pour chaque document dans les résultats de base
        for doc_id, base_score in base_results:
            adjusted_score = base_score  # Score de base
            
            # Ajuster avec les champs
            for field_name, weight in self.field_weights.items():
                field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                
                if not field_tf_dict:
                    continue
                
                # Calculer un bonus basé sur la présence dans ce champ
                field_bonus = 0.0
                for term in query_terms:
                    tf = field_tf_dict.get(term, 0)
                    if tf > 0:
                        # Bonus proportionnel au poids du champ et à la fréquence
                        field_bonus += weight * (tf / (1.0 + tf))
                
                # Appliquer le bonus (vous pouvez ajuster ce facteur)
                adjusted_score += field_bonus * 0.1  # Petit bonus pour éviter de tout changer
            
            adjusted_scores[doc_id] = adjusted_score
        
        # Retrier avec les scores ajustés
        sorted_results = sorted(adjusted_scores.items(), key=lambda x: -x[1])
        
        # Garantir 1500 résultats
        return sorted_results[:1500]
    
    def search_bm25fw_optimized(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fw optimisé : Late combination avec garantie de 1500 résultats
        """
        # D'abord, le fallback standard
        base_results = self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
        
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms or len(query_terms) == 0:
            return base_results
        
        doc_scores = {}
        
        # Pour chaque document dans les résultats de base (optimisation)
        for doc_id, _ in base_results:
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
                        # Utiliser DF spécifique au champ si disponible, sinon global
                        if term in self.field_stats[field_name]['dfs']:
                            df_field = self.field_stats[field_name]['dfs'][term]
                        else:
                            df_field = self.df.get(term, 0)
                        
                        if df_field > 0:
                            # IDF spécifique au champ
                            field_doc_count = self.field_stats[field_name]['doc_count']
                            idf = math.log((field_doc_count - df_field + 0.5) / (df_field + 0.5))
                            
                            # Composante TF avec normalisation de champ
                            if avg_field_length > 0:
                                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (field_length / avg_field_length)))
                            else:
                                tf_component = (tf * (k1 + 1)) / (tf + k1)
                            
                            field_score += idf * tf_component
                
                total_score += weight * field_score
            
            # Score minimal pour garantir un score
            doc_scores[doc_id] = max(total_score, 0.00001)
        
        # Si on a des scores, on les utilise
        if any(score > 0.00001 for score in doc_scores.values()):
            sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
            
            # Compléter si nécessaire
            if len(sorted_results) < 1500:
                # Ajouter des documents du fallback
                used_docs = set(doc_id for doc_id, _ in sorted_results)
                for doc_id, base_score in base_results:
                    if len(sorted_results) >= 1500:
                        break
                    if doc_id not in used_docs:
                        sorted_results.append((doc_id, base_score))
            
            return sorted_results[:1500]
        else:
            # Fallback complet
            return base_results
    
    def search_bm25fr_with_ranker(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr utilisant RankedRetrieval comme base
        """
        # Version simple : même que BM25 sans champs
        return self.search_bm25fw_with_ranker(query, k1, b)
    
    def search_bm25fr_field_aware(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr avec prise en compte des champs
        """
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
        
        # Obtenir les documents pertinents
        relevant_docs = set()
        for term in query_terms:
            if term in self.index.dictionary:
                relevant_docs.update(self.index.dictionary[term].keys())
        
        # Si peu de documents, utiliser le fallback
        if len(relevant_docs) < 100:
            return self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
        
        doc_scores = {}
        
        # Pour chaque document pertinent
        for doc_id in relevant_docs:
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
                    # IDF
                    idf = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
                    
                    # Composante TF
                    doc_length = self.doc_lengths[doc_id]
                    if self.index.avg_doc_length > 0:
                        tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (doc_length / self.index.avg_doc_length)))
                    else:
                        tf_component = (tf_star * (k1 + 1)) / (tf_star + k1)
                    
                    doc_score += idf * tf_component
            
            doc_scores[doc_id] = max(doc_score, 0.00001)
        
        # Trier
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        
        # Compléter avec fallback si nécessaire
        if len(sorted_results) < 1500:
            base_results = self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)
            used_docs = set(doc_id for doc_id, _ in sorted_results)
            
            for doc_id, base_score in base_results:
                if len(sorted_results) >= 1500:
                    break
                if doc_id not in used_docs:
                    sorted_results.append((doc_id, base_score))
        
        return sorted_results[:1500]
    
    def search(self, query: str, method: str = "bm25fw", variant: str = "ranker",
              k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        Méthode unifiée de recherche
        """
        if method == "bm25fw":
            if variant == "ranker":
                return self.search_bm25fw_with_ranker(query, k1, b)
            elif variant == "field_aware":
                return self.search_bm25fw_field_aware(query, k1, b)
            elif variant == "optimized":
                return self.search_bm25fw_optimized(query, k1, b)
            else:
                return self.search_bm25fw_with_ranker(query, k1, b)
        
        elif method == "bm25fr":
            if variant == "ranker":
                return self.search_bm25fr_with_ranker(query, k1, b)
            elif variant == "field_aware":
                return self.search_bm25fr_field_aware(query, k1, b)
            else:
                return self.search_bm25fr_with_ranker(query, k1, b)
        
        else:
            # Fallback vers BM25 standard
            return self.ranker.search_query(query, "bm25", top_k=1500, k1=k1, b=b)

# ==================== FONCTIONS UTILITAIRES ====================

def generate_field_run_with_ranker(run_id: str, run_type: str,
                                 xml_dir: str, queries: Dict[int, str],
                                 config: Dict,
                                 fields_config: Dict = None,
                                 field_weights: Dict = None,
                                 variant: str = "ranker",
                                 k1: float = 1.2, b: float = 0.75) -> str:
    """
    Génère un run utilisant SimpleFieldWeightedIndex avec RankedRetrieval
    """
    print(f"\n▶️  Génération {run_type.upper()} - Variant: {variant}")
    print(f"   Run ID: {run_id}")
    
    # Configuration par défaut
    if fields_config is None:
        fields_config = {
            'title': ['title'],
            'body': ['bdy']
        }
    
    if field_weights is None:
        field_weights = {
            'title': 1.0,
            'body': 1.0
        }
    
    # Construire l'index
    index = SimpleFieldWeightedIndex()
    doc_count = index.build_index(xml_dir, fields_config, field_weights, config)
    
    # Générer le fichier
    team_name = "AlphaAnaClement"
    filename = f"{team_name}_{run_id}_{run_type}_{variant}_k{k1:.1f}_b{b:.2f}.txt"
    filepath = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    total_results = 0
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"  Query {query_id}: {query_text[:40]}...")
            
            # Recherche
            results = index.search(query_text, run_type, variant, k1, b)
            
            # Vérification
            if len(results) < 1500:
                print(f"    ⚠️  Attention: {len(results)} résultats seulement")
                print(f"    Utilisation du fallback RankedRetrieval...")
                # Fallback
                results = index.ranker.search_query(query_text, "bm25", top_k=1500, k1=k1, b=b)
            
            # Écrire les résultats
            for rank, (doc_id, score) in enumerate(results[:1500], 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")
                total_results += 1
    
    # Vérification finale
    expected_results = len(queries) * 1500
    if total_results == expected_results:
        print(f"✅ Fichier généré: {filename}")
        print(f"📊 {total_results} résultats (attendu: {expected_results})")
    else:
        print(f"⚠️  ATTENTION: {total_results} résultats au lieu de {expected_results}")
        print(f"📁 Fichier: {filename}")
    
    return filepath