import os
import time
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import pickle

from indexer import WeightedInvertedIndex, INEXDocument
from ranked_retrieval import RankedRetrieval

class FieldWeightedIndex:
    """Pondération par champs conforme à Wilkinson94 et Robertson94"""
    
    def __init__(self):
        self.index = None
        # CORRECTION: Changer defaultdict(float) en defaultdict(dict)
        self.field_tfs = defaultdict(lambda: defaultdict(dict))  # doc_id -> field -> term -> tf
        self.field_weights = {}
        self.doc_lengths = {}
        self.doc_ids = []
        self.field_stats = {}
    
    def configure(self, tokenization="basic", stemmer="nostem", stop_words="nostop"):
        """Configure l'index principal"""
        self.tokenization = tokenization
        self.stemmer = stemmer
        self.stop_words = stop_words
        
        self.index = WeightedInvertedIndex()
        self.index.configure(
            tokenization=tokenization,
            stemmer=stemmer,
            stop_words=stop_words
        )
    
    def build_field_index(self, xml_dir: str, 
                         fields_mapping: Dict[str, List[str]],
                         field_weights: Dict[str, float],
                         max_files: int = None):
        """
        Construit un index avec marquage des champs
        fields_mapping = {
            'title': ['title'],       # CHAMP SIMPLE ET UNIQUE
            'body': ['bdy'],          # CHAMP SIMPLE (pas sec, p!)
        }
        """
        print(f"Construction index avec champs...")
        start_time = time.time()
        
        self.field_weights = field_weights
        
        # Validation: pas de champs répétables
        for field_name, tags in fields_mapping.items():
            if len(tags) > 1:
                print(f"⚠️  Champ '{field_name}' a {len(tags)} tags. Choisis UN SEUL tag principal.")
        
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"  Traitement {i}/{len(xml_files)}...")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
                continue
            
            doc_id = doc.doc_id
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = 0
            
            # Initialiser les structures pour ce document
            # CORRECTION: S'assurer que field_tfs[doc_id] existe et a les bons champs
            if doc_id not in self.field_tfs:
                self.field_tfs[doc_id] = defaultdict(dict)
            
            # Pour chaque champ, extraire et indexer
            doc_field_tfs = defaultdict(Counter)  # field -> Counter(term -> tf)
            all_terms = []
            
            for field_name, target_tags in fields_mapping.items():
                # Prendre seulement le premier tag comme représentatif
                main_tag = target_tags[0]
                field_text = self._extract_field_simple(doc, main_tag)
                
                if field_text:
                    # Tokenization et processing
                    tokens = self.index.apply_tokenization(field_text)
                    terms = self.index.process_tokens(tokens)
                    
                    if terms:
                        # Stocker les TF par champ
                        term_counter = Counter(terms)
                        doc_field_tfs[field_name] = term_counter
                        all_terms.extend(terms)
                        
                        # CORRECTION: Stocker directement dans field_tfs
                        for term, tf in term_counter.items():
                            self.field_tfs[doc_id][field_name][term] = float(tf)
            
            # Calculer la longueur du document (tous champs confondus)
            self.doc_lengths[doc_id] = len(all_terms)
            
            # Indexer globalement (pour statistiques)
            global_counter = Counter(all_terms)
            for term, tf in global_counter.items():
                if doc_id not in self.index.dictionary[term]:
                    self.index.dictionary[term][doc_id] = 0
                self.index.dictionary[term][doc_id] = tf
        
        # Finaliser l'index global pour les statistiques
        self.index.doc_ids = self.doc_ids
        self.index.doc_lengths = self.doc_lengths
        self.index.doc_count = len(self.doc_ids)
        self.index.total_terms = sum(self.doc_lengths.values())
        if self.index.doc_count > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        
        # Calculer df pour chaque terme
        self.df = {}
        for term, doc_dict in self.index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        print(f"Index avec champs construit en {time.time() - start_time:.2f}s")
        print(f"Champs: {list(fields_mapping.keys())}")
        print(f"Documents: {len(self.doc_ids)}")
    
    def _extract_field_simple(self, doc: INEXDocument, target_tag: str) -> str:
        """
        Extrait le texte d'un CHAMP SIMPLE (non répétable)
        Version corrigée pour éviter les warnings
        """
        if doc.root is None:
            return ""
        
        # Recherche simple: trouver la première occurrence
        def find_element(elem, tag):
            elem_tag = doc._clean_tag(elem.tag)
            if elem_tag == tag:
                return elem
            
            for child in elem:
                # CORRECTION: Vérifier si child n'est pas None
                if child is not None:
                    result = find_element(child, tag)
                    if result is not None:
                        return result
            return None
        
        target_elem = find_element(doc.root, target_tag)
        if target_elem is not None:
            # Extraire tout le texte de cet élément
            text_parts = []
            try:
                for t in target_elem.itertext():
                    if t and t.strip():
                        text_parts.append(t.strip())
                
                raw_text = ' '.join(text_parts)
            except AttributeError:
                # Fallback pour ElementTree
                text = (target_elem.text or "").strip()
                for child in target_elem:
                    if child.text:
                        text += " " + child.text.strip()
                    if child.tail:
                        text += " " + child.tail.strip()
                text_parts.append(text)
                raw_text = ' '.join(text_parts)
            
            return INEXDocument.clean_and_normalize_text(raw_text)
        
        return ""
    
    def _get_xml_files(self, xml_dir: str, max_files: int = None) -> List[str]:
        """Liste les fichiers XML"""
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        if max_files:
            xml_files = xml_files[:max_files]
        
        return xml_files
    
    # ==================== BM25Fw ====================
    
    def search_bm25fw(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fw OPTIMISÉ : Late combination (Wilkinson94)
        """
        from collections import defaultdict
        
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # PRÉCALCULER les statistiques par champ (une seule fois)
        field_stats = {}
        for field_name in self.field_weights.keys():
            # Calculer avg_field_length pour ce champ
            total_field_length = 0
            field_doc_count = 0
            field_dfs = defaultdict(int)  # term -> df dans ce champ
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_tf_dict = self.field_tfs[doc_id][field_name]
                    field_length = sum(field_tf_dict.values())
                    total_field_length += field_length
                    field_doc_count += 1
                    
                    # Compter les df
                    for term in field_tf_dict.keys():
                        field_dfs[term] += 1
            
            avg_field_length = total_field_length / field_doc_count if field_doc_count > 0 else 1
            
            field_stats[field_name] = {
                'avg_length': avg_field_length,
                'doc_count': field_doc_count,
                'dfs': field_dfs
            }
        
        # Dictionnaire pour scores combinés
        doc_scores = defaultdict(float)
        
        # OPTIMISATION: Traiter seulement les documents qui ont au moins un terme de la requête
        relevant_docs = set()
        
        # Pour chaque terme de la requête, trouver les documents qui le contiennent
        for term in query_terms:
            for doc_id in self.doc_ids:
                for field_name in self.field_weights.keys():
                    if term in self.field_tfs[doc_id].get(field_name, {}):
                        relevant_docs.add(doc_id)
                        break  # Passer au document suivant
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        # Pour chaque document pertinent, calculer BM25Fw
        for doc_id in relevant_docs:
            total_score = 0.0
            
            # Pour chaque champ
            for field_name, weight in self.field_weights.items():
                field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                
                if not field_tf_dict:
                    continue
                
                field_length = sum(field_tf_dict.values())
                field_stat = field_stats[field_name]
                avg_field_length = field_stat['avg_length']
                field_dfs = field_stat['dfs']
                
                field_score = 0.0
                
                # Pour chaque terme de la requête
                for term in query_terms:
                    tf = field_tf_dict.get(term, 0)
                    
                    if tf > 0:
                        df_field = field_dfs.get(term, 0)
                        
                        if df_field > 0:
                            # BM25 pour ce champ
                            idf = math.log((field_stat['doc_count'] - df_field + 0.5) / (df_field + 0.5))
                            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (field_length / avg_field_length)))
                            field_score += idf * tf_component
                
                total_score += weight * field_score
            
            if total_score > 0:
                doc_scores[doc_id] = total_score
        
        # OPTIMISATION: Si peu de documents, ajouter un score minimal aux autres
        if len(doc_scores) < 1500:
            # Pour les documents non pertinents, donner un petit score
            for doc_id in self.doc_ids:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.00001
        
        # Trier et limiter
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    # ==================== BM25Fr ====================
    
    def search_bm25fr(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr OPTIMISÉ : Early combination (Robertson94)
        """
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # OPTIMISATION: Pré-calculer les documents pertinents
        relevant_docs = set()
        for term in query_terms:
            if term in self.index.dictionary:
                relevant_docs.update(self.index.dictionary[term].keys())
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        doc_scores = defaultdict(float)
        
        # Pour chaque document pertinent seulement
        for doc_id in relevant_docs:
            combined_tf = defaultdict(float)
            
            # Étape 1: Combiner les TF pondérés
            for term in query_terms:
                tf_star = 0.0
                for field_name, weight in self.field_weights.items():
                    field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                    tf_in_field = field_tf_dict.get(term, 0)
                    tf_star += weight * tf_in_field
                
                if tf_star > 0:
                    combined_tf[term] = tf_star
            
            # Étape 2: BM25 sur les TF combinés
            doc_score = 0.0
            for term, tf_star in combined_tf.items():
                # DF global
                df = self.df.get(term, 0)
                
                if df > 0:
                    # BM25 avec tf_star
                    idf = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
                    tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (self.doc_lengths[doc_id] / self.index.avg_doc_length)))
                    doc_score += idf * tf_component
            
            if doc_score > 0:
                doc_scores[doc_id] = doc_score
        
        # OPTIMISATION: Si peu de documents, ajouter un score minimal
        if len(doc_scores) < 1500:
            for doc_id in self.doc_ids:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.00001
        
        # Trier
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    def search_bm25fr_optimized(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr ULTRA OPTIMISÉ : Early combination
        """
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # OPTIMISATION 1: Filtrer les termes qui existent dans la collection
        valid_terms = [term for term in query_terms if term in self.df]
        if not valid_terms:
            return []
        
        # OPTIMISATION 2: Précalculer l'IDF pour chaque terme
        idf_cache = {}
        for term in valid_terms:
            df = self.df[term]
            idf_cache[term] = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
        
        # OPTIMISATION 3: Documents qui contiennent au moins un terme
        relevant_docs = set()
        for term in valid_terms:
            if term in self.index.dictionary:
                relevant_docs.update(self.index.dictionary[term].keys())
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        # OPTIMISATION 4: Batch processing
        batch_size = 500
        doc_scores = {}
        
        doc_list = list(relevant_docs)
        for i in range(0, len(doc_list), batch_size):
            batch = doc_list[i:min(i+batch_size, len(doc_list))]
            
            for doc_id in batch:
                combined_tf = 0.0
                total_tf_star = 0.0
                
                # Pour chaque terme
                for term in valid_terms:
                    tf_star = 0.0
                    # Combiner les TF pondérés
                    for field_name, weight in self.field_weights.items():
                        field_tf_dict = self.field_tfs.get(doc_id, {}).get(field_name, {})
                        tf_in_field = field_tf_dict.get(term, 0)
                        tf_star += weight * tf_in_field
                    
                    if tf_star > 0:
                        combined_tf += tf_star
                        
                        # BM25 partiel
                        idf = idf_cache[term]
                        tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (self.doc_lengths[doc_id] / self.index.avg_doc_length)))
                        total_tf_star += idf * tf_component
                
                if total_tf_star > 0:
                    doc_scores[doc_id] = total_tf_star
        
        # OPTIMISATION 5: Si besoin, ajouter des documents avec score minimal
        if len(doc_scores) < 1500:
            needed = 1500 - len(doc_scores)
            other_docs = [d for d in self.doc_ids if d not in doc_scores]
            
            # Prendre les premiers autres documents
            for doc_id in other_docs[:needed]:
                doc_scores[doc_id] = 0.00001
        
        # Trier
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
def generate_field_weighted_run(generator, run_id: str, run_type: str,
                               xml_dir: str, queries: Dict[int, str],
                               config: Dict, run_params: Dict,
                               fields_config: Dict[str, List[str]] = None,
                               field_weights: Dict[str, float] = None):
    """
    Génère un run avec pondération par champs (CORRIGÉ)
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION RUN {run_id} - {run_type.upper()}")
    print('='*70)
    
    # Configuration par défaut CORRECTE
    if fields_config is None:
        fields_config = {
            'title': ['title'],     # CHAMP UNIQUE
            'body': ['bdy'],        # CHAMP UNIQUE (pas sec!)
        }
    
    if field_weights is None:
        field_weights = {
            'title': 3.0,  # Le titre est très important
            'body': 1.0    # Le corps moins
        }
    
    # Construire l'index avec champs
    field_index = FieldWeightedIndex()  # <-- NOTE: Garder le nom original
    field_index.configure(**config)
    
    start_time = time.time()
    print("Construction de l'index avec champs...")
    field_index.build_field_index(
        xml_dir=xml_dir,
        fields_mapping=fields_config,
        field_weights=field_weights,
        max_files=run_params.get('max_files', None)
    )
    
    # Générer le nom de fichier
    team_name = "AlphaAnaClement"
    fields_str = '-'.join(fields_config.keys())
    filename = f"{team_name}_12_{run_id}_{run_type}_fields-{fields_str}_{config['stemmer']}_{config['stop_words']}_k{run_params.get('k1', 1.2)}_b{run_params.get('b', 0.75)}.txt"
    filename = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    results_count = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"\n[Query {query_id}] {query_text[:50]}...")
            query_start = time.time()
            
            # Sélectionner la méthode
            if run_type == 'bm25fw':
                results = field_index.search_bm25fw(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            else:  # bm25fr
                #results = field_index.search_bm25fr(
                results = field_index.search_bm25fr_optimized(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            
            # Écrire les résultats (articles entiers)
            rank = 1
            for doc_id, score in results[:1500]:
                # FORMAT: articles entiers, pas d'éléments
                f.write(
                    f"{query_id} Q0 {doc_id} {rank} "
                    f"{score:.6f} {team_name} /article[1]\n"
                )
                rank += 1
                results_count += 1
            
            query_time = time.time() - query_start
            print(f"  {len(results)} articles, temps: {query_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"RUN {run_type.upper()} TERMINÉ: {filename}")
    print(f"Total résultats: {results_count}")
    print(f"Temps total: {total_time:.2f}s")
    print('='*70)
    
    return filename

def exercice5_corrected():
    """Exercice 5: BM25Fw - Late combination (Wilkinson94) - CORRIGÉ"""
    print("=" * 70)
    print("EXERCICE 5: BM25Fw - Late combination of fields (CORRECTED)")
    print("=" * 70)
    
    from xml_run_manager import INEXRunGenerator
    
    generator = INEXRunGenerator()
    
    # Requêtes INEX
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    # Configuration
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671',
    }
    
    # ✅ CONFIGURATION CORRECTE: champs uniques et non répétables
    fields_config = {
        'title': ['title'],   # Seulement le title (unique)
        'body': ['bdy'],      # Seulement bdy (unique, contient tout)
    }
    
    field_weights = {
        'title': 2.5,  # Titre très important
        'body': 1.0    # Corps standard
    }
    
    # Paramètres BM25
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None  # Tous les fichiers
    }
    
    # Générer le run
    filename = generate_field_weighted_run(
        generator=generator,
        run_id="test5",
        run_type="bm25fw",
        xml_dir="data/Practice_05_data/XML-Coll-withSem",
        queries=queries,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    # Validation
    generator.validate_run_file(filename)
    
    print(f"\n✅ Exercice 5 terminé")
    print(f"📁 Run généré: {filename}")
    
    return filename


def exercice6_corrected():
    """Exercice 6: BM25Fr - Early combination (Robertson94) - CORRIGÉ"""
    print("\n" + "=" * 70)
    print("EXERCICE 6: BM25Fr - Early combination of fields (CORRECTED)")
    print("=" * 70)
    
    from xml_run_manager import INEXRunGenerator
    
    generator = INEXRunGenerator()
    
    # Requêtes INEX
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    # Configuration différente
    config = {
        'tokenization': 'basic',
        'stemmer': 'snowball',
        'stop_words': 'nostop'  # Sans stopwords pour voir la différence
    }
    
    # ✅ Test avec 3 champs simples
    fields_config = {
        'title': ['title'],
        'abstract': ['bdy'],  # On utilise bdy comme "abstract"
        'body': ['bdy']       # Même source mais poids différent
    }
    
    field_weights = {
        'title': 3.0,
        'abstract': 1.5,
        'body': 1.0
    }
    
    # Paramètres BM25 différents
    run_params = {
        'k1': 1.5,
        'b': 0.8,
        'max_files': None
    }
    
    # Générer le run
    filename = generate_field_weighted_run(
        generator=generator,
        run_id="test6",
        run_type="bm25fr",
        xml_dir="data/Practice_05_data/XML-Coll-withSem",
        queries=queries,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    # Validation
    generator.validate_run_file(filename)
    
    print(f"\n✅ Exercice 6 terminé")
    print(f"📁 Run généré: {filename}")
    
    return filename


def main_exercices_5_6_corrected():
    """Exécute les exercices 5 et 6 corrigés"""
    
    print("=" * 70)
    print("EXERCICES 5 & 6: Field Weighting Methods (CORRECTED)")
    print("=" * 70)
    
    # Exercice 5
    #file5 = exercice5_corrected()
    
    # Exercice 6
    file6 = exercice6_corrected()
    
    print("\n" + "="*70)
    print("RÉSUMÉ EXERCICES 5-6 CORRIGÉS")
    print("="*70)
    #print(f"1. Exercice 5 (BM25Fw): {os.path.basename(file5)}")
    print(f"2. Exercice 6 (BM25Fr): {os.path.basename(file6)}")
    
    #return [file5, file6]


if __name__ == "__main__":
    main_exercices_5_6_corrected()