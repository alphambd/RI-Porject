import os
import time
import hashlib
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import math

from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument

class FieldWeightedIndex:
    """Index avec pondération par champs - VERSION SIMPLIFIÉE"""
    
    def __init__(self, cache_dir="data/cache/field_index"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialiser les structures de données
        self._init_data_structures()
        
    def _init_data_structures(self):
        """Initialise toutes les structures de données"""
        self.index = None
        self.field_tfs = defaultdict(lambda: defaultdict(dict))
        self.field_weights = {}
        self.doc_lengths = {}
        self.doc_ids = []
        self.df = {}
        self.field_stats = {}
        self.field_extraction_cache = {}

        # Stocker la configuration des champs
        self.fields_config = None
        
        # Configuration des tags
        self.unique_tags = {'title', 'bdy'}
        self.repeatable_tags = {'sec', 'p', 'caption', 'link', 'it'}
    
    # ==================== GESTION CACHE ====================
    
    def _get_cache_key(self, xml_dir: str, fields_config: Dict, 
                      field_weights: Dict, config: Dict) -> str:
        """Génère une clé de cache unique"""
        params = {
            'xml_dir': xml_dir,
            'fields_config': fields_config,
            'field_weights': field_weights,
            'tokenization': config.get('tokenization', 'basic'),
            'stemmer': config.get('stemmer', 'nostem'),
            'stop_words': config.get('stop_words', 'nostop'),
        }
        params_str = str(sorted(params.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    def _save_to_cache(self, cache_key: str):
        """Sauvegarde l'index dans le cache"""
        cache_data = {
            'field_tfs': dict(self.field_tfs),
            'field_weights': self.field_weights,
            'doc_lengths': self.doc_lengths,
            'doc_ids': self.doc_ids,
            'df': self.df,
            'index_data': self.index.get_cache_data() if self.index else None,
            'field_stats': self.field_stats
        }
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"💾 Index sauvegardé dans le cache: {cache_file}")
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Charge l'index depuis le cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            print(f"📂 Chargement depuis le cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restaurer les données
            self._init_data_structures()
            self.field_tfs = defaultdict(lambda: defaultdict(dict), cache_data['field_tfs'])
            self.field_weights = cache_data['field_weights']
            self.doc_lengths = cache_data['doc_lengths']
            self.doc_ids = cache_data['doc_ids']
            self.df = cache_data['df']
            self.field_stats = cache_data.get('field_stats', {})
            
            # Restaurer l'index principal
            if cache_data['index_data']:
                self._restore_index(cache_data['index_data'])
            
            print(f"✅ Index chargé: {len(self.doc_ids)} documents")
            print(f"   avg_doc_length = {self.index.avg_doc_length:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement cache: {e}")
            return False
    
    def _restore_index(self, index_data):
        """Restaure l'index principal depuis les données"""
        self.index = WeightedInvertedIndex()
        
        # Restaurer les données de base
        self.index.dictionary = defaultdict(dict, index_data['dictionary'])
        self.index.doc_ids = index_data['doc_ids']
        self.index.doc_lengths = index_data['doc_lengths']
        self.index.doc_count = index_data['doc_count']
        self.index.total_terms = index_data['total_terms']
        
        # Calculer avg_doc_length
        if self.index.doc_count > 0 and self.index.total_terms > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        else:
            total_length = sum(self.index.doc_lengths.values())
            self.index.avg_doc_length = total_length / self.index.doc_count if self.index.doc_count > 0 else 0
        
        # Configuration
        self.index.stop_list_name = index_data['config']['stop_list_name']
        self.index.stemmer_name = index_data['config']['stemmer_name']
        self.index.tokenization_method = index_data['config']['tokenization_method']
    
    # ==================== CONFIGURATION ====================
    
    def configure(self, tokenization="basic", stemmer="nostem", stop_words="nostop"):
        """Configure l'index principal"""
        self.index = WeightedInvertedIndex()
        self.index.configure(
            tokenization=tokenization,
            stemmer=stemmer,
            stop_words=stop_words
        )
    
    # ==================== CONSTRUCTION INDEX ====================
    
    def build_or_load_field_index(self, xml_dir: str, 
                                fields_config: Dict[str, List[str]],
                                field_weights: Dict[str, float],
                                config: Dict,
                                max_files: Optional[int] = None,
                                force_rebuild: bool = False) -> int:
        """
        Construit OU charge depuis cache un index avec pondération par champs.
        """
        # NOUVEAU: Stocker la configuration pour l'extraction du "rest"
        self.fields_config = fields_config
        
        # 1. Générer clé de cache
        cache_key = self._get_cache_key(xml_dir, fields_config, field_weights, config)
        
        # 2. Essayer de charger depuis cache
        if not force_rebuild and self._load_from_cache(cache_key):
            return len(self.doc_ids)
        
        # 3. Construire l'index
        return self._build_field_index(xml_dir, fields_config, field_weights, config, max_files, cache_key)

    def _build_field_index(self, xml_dir, fields_config, field_weights, config, max_files, cache_key):
        """Construit l'index (méthode interne)"""
        print("🔄 Construction de l'index...")
        self.configure(**config)
        self.field_weights = field_weights
        
        start_time = time.time()
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        print(f"📄 Traitement de {len(xml_files)} fichiers XML...")
        
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"  Traitement {i}/{len(xml_files)}...")
            
            self._process_xml_file(xml_file, fields_config)
        
        self._finalize_index()
        
        # Calculer les statistiques
        self._compute_field_stats()
        
        indexing_time = time.time() - start_time
        print(f"✅ Index construit en {indexing_time:.2f}s: {len(self.doc_ids)} documents")
        print(f"   Champs: {list(fields_config.keys())}")
        print(f"   Longueur moyenne: {self.index.avg_doc_length:.1f} termes")
        
        # Sauvegarder dans le cache
        self._save_to_cache(cache_key)
        
        return len(self.doc_ids)
    
    def _process_xml_file(self, xml_file, fields_config):
        """Traite un fichier XML individuel - VERSION AVEC REST"""
        doc = INEXDocument(xml_file)
        if not doc.parse():
            return
        
        doc_id = doc.doc_id
        self.doc_ids.append(doc_id)
        
        # Initialiser les structures pour ce document
        self.doc_lengths[doc_id] = 0
        if doc_id not in self.field_tfs:
            self.field_tfs[doc_id] = defaultdict(dict)
        
        all_terms = []
        
        # TRAITER CHAQUE CHAMP SPÉCIFIQUE
        for field_name, target_tags in fields_config.items():
            # Si c'est le champ "rest", on le traite séparément
            if target_tags == ['__REST__']:
                continue  # On le traitera après
            
            field_text = self._extract_field_text(doc, target_tags)
            
            if field_text:
                # Tokenization et traitement
                tokens = self.index.apply_tokenization(field_text)
                terms = self.index.process_tokens(tokens)
                
                if terms:
                    # Stocker les TF pour ce champ
                    term_counts = Counter(terms)
                    for term, tf in term_counts.items():
                        self.field_tfs[doc_id][field_name][term] = float(tf)
                    
                    all_terms.extend(terms)
        
        # TRAITER LE CHAMP "rest" S'IL EXISTE
        if '__REST__' in [tag for tags in fields_config.values() for tag in tags]:
            rest_text = self._extract_remaining_text(doc)
            if rest_text:
                tokens = self.index.apply_tokenization(rest_text)
                terms = self.index.process_tokens(tokens)
                
                if terms:
                    # Stocker dans le champ "rest"
                    term_counts = Counter(terms)
                    for term, tf in term_counts.items():
                        self.field_tfs[doc_id]['rest'][term] = float(tf)
                    
                    all_terms.extend(terms)
        
        # Mettre à jour les statistiques du document
        self.doc_lengths[doc_id] = len(all_terms)
        
        # Mettre à jour l'index global
        for term in set(all_terms):
            count = all_terms.count(term)
            if doc_id not in self.index.dictionary[term]:
                self.index.dictionary[term][doc_id] = 0
            self.index.dictionary[term][doc_id] += count

    def _extract_field_text(self, doc: INEXDocument, target_tags: List[str]) -> str:
        """Extrait le texte pour un champ donné"""
        # Si c'est le champ "rest", retourner vide (on le traite dans _extract_remaining_text)
        if target_tags == ['__REST__']:
            return ""
        
        field_texts = []
        
        for tag in target_tags:
            if tag in self.unique_tags:
                text = self._extract_single_element(doc, tag)
            else:
                text = self._extract_all_elements(doc, tag)
            
            if text and text.strip():
                field_texts.append(text.strip())
        
        return ' '.join(field_texts) if field_texts else ""

    def _extract_single_element(self, doc: INEXDocument, target_tag: str) -> str:
        """Extrait le premier élément d'un tag unique"""
        if doc.root is None:
            return ""
        
        # Recherche récursive
        def find_element(elem, tag):
            elem_tag = doc._clean_tag(elem.tag)
            if elem_tag == tag:
                return elem
            
            for child in elem:
                if child is not None:
                    result = find_element(child, tag)
                    if result is not None:
                        return result
            return None
        
        target_elem = find_element(doc.root, target_tag)
        if target_elem is None:
            return ""
        
        # Extraire le texte
        text_parts = []
        try:
            # Méthode lxml
            for t in target_elem.itertext():
                if t and t.strip():
                    text_parts.append(t.strip())
        except AttributeError:
            # Fallback ElementTree
            text = (target_elem.text or "").strip()
            for child in target_elem:
                if child.text:
                    text += " " + child.text.strip()
                if child.tail:
                    text += " " + child.tail.strip()
            text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _extract_all_elements(self, doc: INEXDocument, target_tag: str) -> str:
        """Extrait TOUS les éléments d'un tag répétable"""
        if doc.root is None:
            return ""
        
        all_texts = []
        
        def collect_elements(elem, tag):
            elem_tag = doc._clean_tag(elem.tag)
            if elem_tag == tag:
                # Extraire le texte de cet élément
                text_parts = []
                try:
                    for t in elem.itertext():
                        if t and t.strip():
                            text_parts.append(t.strip())
                except AttributeError:
                    text = (elem.text or "").strip()
                    for child in elem:
                        if child.text:
                            text += " " + child.text.strip()
                        if child.tail:
                            text += " " + child.tail.strip()
                    text_parts.append(text)
                
                if text_parts:
                    all_texts.append(' '.join(text_parts))
            
            # Explorer les enfants
            for child in elem:
                if child is not None:
                    collect_elements(child, tag)
        
        collect_elements(doc.root, target_tag)
        return ' '.join(all_texts)
    
    def _extract_remaining_text(self, doc: INEXDocument) -> str:
        """
        Extrait tout le texte qui n'est pas déjà capturé par les champs spécifiés.
        Utilise self.fields_config pour savoir quels tags sont déjà utilisés.
        """
        if doc.root is None or not self.fields_config:
            return ""
        
        # Récupérer tous les tags utilisés par les champs (sauf "__REST__")
        used_tags = set()
        for field_tags in self.fields_config.values():
            for tag in field_tags:
                if tag != '__REST__':
                    used_tags.add(tag)
        
        all_texts = []
        
        def collect_remaining(elem, in_used_tag=False):
            """
            Parcourt récursivement l'arbre XML.
            in_used_tag: True si on est déjà dans un tag utilisé par un champ spécifique
            """
            current_tag = doc._clean_tag(elem.tag)
            
            # Si ce tag est utilisé par un champ spécifique, on marque
            if current_tag in used_tags:
                # On ne prend pas le texte de ce tag (il sera pris par le champ spécifique)
                # Mais on explore ses enfants (certains enfants pourraient être du "rest")
                in_current_tag = True
            else:
                # Ce tag n'est pas spécifié -> on peut prendre son texte
                in_current_tag = False
                
                # Prendre le texte de l'élément lui-même
                if elem.text and elem.text.strip():
                    # Nettoyer le texte
                    text = elem.text.strip()
                    if len(text) > 2:  # Ignorer les textes trop courts
                        all_texts.append(text)
            
            # Explorer les enfants
            for child in elem:
                if child is not None:
                    collect_remaining(child, in_used_tag or in_current_tag)
            
            # Texte "tail" (après la balise fermante)
            if not in_used_tag and elem.tail and elem.tail.strip():
                tail_text = elem.tail.strip()
                if len(tail_text) > 2:
                    all_texts.append(tail_text)
        
        # Démarrer la collecte
        collect_remaining(doc.root, False)
        
        return ' '.join(all_texts)

    def _finalize_index(self):
        """Finalise les statistiques de l'index"""
        self.index.doc_ids = self.doc_ids
        self.index.doc_lengths = self.doc_lengths
        self.index.doc_count = len(self.doc_ids)
        self.index.total_terms = sum(self.doc_lengths.values())
        
        if self.index.doc_count > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        
        # Calculer les DF
        for term, doc_dict in self.index.dictionary.items():
            self.df[term] = len(doc_dict)
    
    def _compute_field_stats(self):
        """Calcule les statistiques par champ"""
        print("📊 Calcul des statistiques par champ...")
        
        for field_name in self.field_weights.keys():
            total_field_length = 0
            field_doc_count = 0
            field_dfs = defaultdict(int)
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_tf_dict = self.field_tfs[doc_id][field_name]
                    field_length = sum(field_tf_dict.values())
                    total_field_length += field_length
                    field_doc_count += 1
                    
                    # Compter les df spécifiques au champ
                    for term in field_tf_dict.keys():
                        field_dfs[term] += 1
            
            avg_field_length = total_field_length / field_doc_count if field_doc_count > 0 else 1
            
            self.field_stats[field_name] = {
                'avg_length': avg_field_length,
                'doc_count': field_doc_count,
                'dfs': dict(field_dfs)
            }
            
            print(f"  {field_name}: {field_doc_count} docs, avg_len={avg_field_length:.1f}")
    
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
    
    # ==================== MÉTHODES DE RECHERCHE ====================
    
    def search_bm25fw(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fw : Late combination (Wilkinson94)
        """
        # Traiter la requête
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
                
                # Statistiques du champ
                field_length = sum(field_tf_dict.values())
                field_stat = self.field_stats[field_name]
                avg_field_length = field_stat['avg_length']
                field_dfs = field_stat['dfs']
                
                # Score pour ce champ
                field_score = 0.0
                
                for term in query_terms:
                    tf = field_tf_dict.get(term, 0)
                    if tf > 0:
                        df_field = field_dfs.get(term, 0)
                        if df_field > 0:
                            # BM25 pour ce champ
                            idf = math.log10((field_stat['doc_count'] - df_field + 0.5) / (df_field + 0.5))
                            tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (field_length / avg_field_length)))
                            field_score += idf * tf_comp
                
                total_score += weight * field_score
            
            if total_score > 0:
                doc_scores[doc_id] = total_score
            #doc_scores[doc_id] = total_score

        # Trier et limiter
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    def search_bm25fr(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr : Early combination (Robertson94)
        """
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        doc_scores = {}
        
        # Pour chaque document
        for doc_id in self.doc_ids:
            # Combiner les TF pondérés (combinaison précoce)
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
                    # Protection contre les valeurs nulles
                    if self.index.avg_doc_length <= 0:
                        self.index.avg_doc_length = 1.0
                    
                    doc_length = self.doc_lengths.get(doc_id, 1)
                    length_ratio = doc_length / self.index.avg_doc_length
                    
                    # Calcul BM25
                    idf = math.log10((self.index.doc_count - df + 0.5) / (df + 0.5))
                    denominator = tf_star + k1 * (1 - b + b * length_ratio)
                    
                    if denominator > 0:
                        tf_comp = (tf_star * (k1 + 1)) / denominator
                        doc_score += idf * tf_comp
            
            #if doc_score > 0:
            #    doc_scores[doc_id] = doc_score
            doc_scores[doc_id] = doc_score
        
        # Trier et limiter
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    # ==================== CACHE DES RECHERCHES ====================
    
    def _get_search_cache_key(self, query: str, method: str, 
                            k1: float, b: float) -> str:
        """Clé de cache pour une recherche"""
        params = {
            'query': query,
            'method': method,
            'k1': k1,
            'b': b,
            'field_weights_hash': hash(tuple(sorted(self.field_weights.items())))
        }
        params_str = str(sorted(params.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def search_with_cache(self, query: str, method: str = "bm25fw", 
                         k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """Recherche avec cache des résultats"""
        cache_key = self._get_search_cache_key(query, method, k1, b)
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.pkl")
        
        # Essayer le cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"⚡ Résultats depuis cache: {query[:30]}...")
                return results
            except:
                pass
        
        # Calculer
        print(f"🔍 Calcul {method} pour: {query[:30]}...")
        
        if method == "bm25fw":
            results = self.search_bm25fw(query, k1, b)
        else:  # bm25fr
            results = self.search_bm25fr(query, k1, b)
        
        # Sauvegarder dans le cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde cache: {e}")
        
        return results

# ==================== FONCTION DE GÉNÉRATION ====================

#def generate_field_weighted_run(generator, run_id: str, run_type: str,
def generate_field_weighted_run(run_id: str, run_type: str,
                              xml_dir: str, queries: Dict[int, str],
                              config: Dict, run_params: Dict,
                              fields_config: Dict = None,
                              field_weights: Dict = None) -> str:
    """
    Génère un run avec pondération par champs (version simplifiée)
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION RUN {run_id} - {run_type.upper()}")
    print('='*70)
    
    # Configuration par défaut
    if fields_config is None:
        fields_config = {
            'title': ['title'],
            'bdy': ['bdy'],
            'sec': ['sec'],
            'p': ['p']
        }
    
    if field_weights is None:
        field_weights = {
            'title': 3.0,
            'bdy': 2.0,
            'sec': 1.5,
            'p': 1.0
        }
    
    # Créer l'index
    field_index = FieldWeightedIndex(cache_dir="data/cache/field_weighted")
    
    start_time = time.time()
    print("📚 Chargement/construction de l'index...")
    
    # Construire ou charger l'index
    doc_count = field_index.build_or_load_field_index(
        xml_dir=xml_dir,
        fields_config=fields_config,
        field_weights=field_weights,
        config=config,
        max_files=run_params.get('max_files', None),
        force_rebuild=False
    )
    
    # Générer le fichier
    team_name = "AlphaAnaClement"
    fields_str = '-'.join(fields_config.keys())
    
    filename = (
        f"{team_name}_{run_id}_{run_type}_"
        f"fields-{fields_str}_{config['stop_words']}_"
        f"{config['stemmer']}_k_{run_params.get('k1', 1.2):.1f}_"
        f"b_{run_params.get('b', 0.75):.2f}.txt"
    )
    
    filename = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    results_count = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"\n[Query {query_id}] {query_text[:50]}...")
            query_start = time.time()
            
            # Recherche avec cache
            results = field_index.search_with_cache(
                query_text,
                method=run_type,
                k1=run_params.get('k1', 1.2),
                b=run_params.get('b', 0.75)
            )
            
            # Écrire les résultats
            rank = 1
            for doc_id, score in results[:1500]:
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")
                rank += 1
                results_count += 1
            
            query_time = time.time() - query_start
            print(f"  {len(results)} articles, temps: {query_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ RUN {run_type.upper()} TERMINÉ")
    print(f"📁 Fichier: {filename}")
    print(f"📊 Documents indexés: {doc_count}")
    print(f"⏱️  Temps total: {total_time:.2f}s")
    print(f"📈 Résultats: {results_count} lignes")
    print('='*70)
    
    return filename

def generate_field_weighted_run_simple(run_id: str, run_type: str,
                                     xml_dir: str, queries: Dict[int, str],
                                     config: Dict, run_params: Dict,
                                     fields_config: Dict = None,
                                     field_weights: Dict = None) -> str:
    """
    Version SIMPLE pour générer un run avec pondération par champs.
    """
    # Import ici pour éviter les dépendances circulaires
    from field_weighted_index import FieldWeightedIndex
    
    print(f"\n  Début de {run_type.upper()} - Run {run_id}")
    
    # Configuration par défaut
    if fields_config is None:
        fields_config = {
            'title': ['title'],
            'body': ['bdy']
        }
    
    if field_weights is None:
        field_weights = {
            'title': 2.0,
            'body': 1.0
        }
    
    # 1. Créer l'index
    print("  1. Construction de l'index...")
    field_index = FieldWeightedIndex()
    
    doc_count = field_index.build_or_load_field_index(
        xml_dir=xml_dir,
        fields_config=fields_config,
        field_weights=field_weights,
        config=config,
        max_files=run_params.get('max_files', None)
    )
    
    print(f"     ✅ {doc_count} documents indexés")
    
    # 2. Préparer le fichier
    team_name = "AlphaAnaClement"
    os.makedirs("data/runs", exist_ok=True)
    
    filename = f"{team_name}_{run_id}_{run_type}_simple.txt"
    filepath = os.path.join("data/runs", filename)
    
    # 3. Traiter chaque requête
    print(f"  2. Traitement des {len(queries)} requêtes...")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"    Query {query_id}: {query_text[:40]}...")
            
            # Recherche selon la méthode
            if run_type == "bm25fw":
                results = field_index.search_bm25fw(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            else:  # bm25fr
                results = field_index.search_bm25fr(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            
            # Écrire les résultats
            rank = 1
            for doc_id, score in results[:1500]:  # Limite INEX
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")
                rank += 1
    
    print(f"\n✅ Run généré: {filename}")
    print(f"   📊 Documents: {doc_count}")
    print(f"   📁 Emplacement: {filepath}")
    
    return filepath

def generate_field_weighted_run_with_rest(run_id: str, run_type: str,
                                        xml_dir: str, queries: Dict[int, str],
                                        config: Dict, run_params: Dict,
                                        fields_config: Dict = None,
                                        field_weights: Dict = None,
                                        include_rest: bool = False,
                                        rest_weight: float = 1.0) -> str:
    """
    Génère un run avec pondération par champs.
    
    Args:
        include_rest: Si True, ajoute un champ "rest" avec tout le texte non capturé
        rest_weight: Poids à donner au champ "rest" (défaut: 1.0)
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION RUN {run_id} - {run_type.upper()}")
    if include_rest:
        print(f"AVEC CHAMP 'rest' (poids={rest_weight})")
    print('='*70)
    
    # Configuration par défaut
    if fields_config is None:
        fields_config = {
            'title': ['title'],
            'bdy': ['bdy'],
            'sec': ['sec'],
            'p': ['p']
        }
    
    # AJOUTER LE CHAMP "rest" SI DEMANDÉ
    if include_rest:
        fields_config['rest'] = ['__REST__']  # Tag spécial pour identifier
    
    if field_weights is None:
        field_weights = {
            'title': 1.0,
            'bdy': 1.0,
            'sec': 1.0,
            'p': 1.0
        }
        # AJOUTER LE POIDS POUR "rest" SI INCLU
        if include_rest:
            field_weights['rest'] = rest_weight
    
    # Créer l'index
    field_index = FieldWeightedIndex(cache_dir="data/cache/field_weighted")
    
    start_time = time.time()
    print("📚 Chargement/construction de l'index...")
    
    # Construire ou charger l'index
    # Forcer le recalcul si on inclut "rest" (car différent de la configuration sans "rest")
    force_rebuild = include_rest
    doc_count = field_index.build_or_load_field_index(
        xml_dir=xml_dir,
        fields_config=fields_config,
        field_weights=field_weights,
        config=config,
        max_files=run_params.get('max_files', None),
        force_rebuild=force_rebuild
    )
    
    # Générer le fichier
    team_name = "AlphaAnaClement"
    
    # Créer un nom de fichier descriptif
    fields_list = []
    for field_name, tags in fields_config.items():
        if tags == ['__REST__']:
            fields_list.append(f'rest{rest_weight}')
        else:
            fields_list.append(field_name)
    fields_str = '-'.join(fields_list)
    
    filename = (
        f"{team_name}_{run_id}_{run_type}_"
        f"fields-{fields_str}_{config['stop_words']}_"
        f"{config['stemmer']}_k_{run_params.get('k1', 1.2):.1f}_"
        f"b_{run_params.get('b', 0.75):.2f}.txt"
    )
    
    filename = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    results_count = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"\n[Query {query_id}] {query_text[:50]}...")
            query_start = time.time()
            
            # Recherche
            if run_type == "bm25fw":
                results = field_index.search_bm25fw(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            else:  # bm25fr
                results = field_index.search_bm25fr(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            
            # Écrire les résultats
            rank = 1
            for doc_id, score in results[:1500]:
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")
                rank += 1
                results_count += 1
            
            query_time = time.time() - query_start
            print(f"  {len(results)} articles, temps: {query_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Afficher les statistiques
    print(f"\n{'='*70}")
    print(f"✅ RUN {run_type.upper()} TERMINÉ")
    print(f"📁 Fichier: {os.path.basename(filename)}")
    print(f"📊 Documents indexés: {doc_count}")
    print(f"⏱️  Temps total: {total_time:.2f}s")
    print(f"📈 Résultats: {results_count} lignes")
    
    if include_rest:
        # Afficher quelques statistiques sur le champ "rest"
        rest_doc_count = 0
        rest_total_terms = 0
        
        for doc_id in field_index.doc_ids:
            if 'rest' in field_index.field_tfs[doc_id]:
                rest_doc_count += 1
                rest_total_terms += sum(field_index.field_tfs[doc_id]['rest'].values())
        
        if rest_doc_count > 0:
            avg_rest_terms = rest_total_terms / rest_doc_count
            print(f"\n📊 Statistiques du champ 'rest':")
            print(f"   Documents avec 'rest': {rest_doc_count}/{doc_count} ({rest_doc_count/doc_count*100:.1f}%)")
            print(f"   Termes moyens par document dans 'rest': {avg_rest_terms:.1f}")
    
    print('='*70)
    
    return filename
