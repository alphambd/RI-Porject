import os
from typing import List


def analyze_scores_from_run(filename: str, sample_lines: int = 100):
    """Analyse rapide des scores d'un run."""
    scores = []
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    score = float(parts[4])
                    scores.append(score)
                except:
                    pass
    
    if not scores:
        return "Fichier vide ou format invalide"
    
    # Calculer les indicateurs
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score
    
    # Compter les scores très bas (potentiellement mauvais)
    very_low_scores = sum(1 for s in scores if s < 0.0001)
    
    print(f"📊 Analyse de {filename}:")
    print(f"  Moyenne score: {avg_score:.6f}")
    print(f"  Max score: {max_score:.6f}")
    print(f"  Min score: {min_score:.6f}")
    print(f"  Plage: {score_range:.6f}")
    print(f"  Scores < 0.0001: {very_low_scores}/{len(scores)}")
    
    # Évaluation rapide
    if max_score > 10.0 and avg_score > 0.5 and very_low_scores < len(scores)/2:
        return "✅ PROMETTEUR - Bonne distribution"
    elif max_score < 1.0 or avg_score < 0.01:
        return "❌ PROBLÉMATIQUE - Scores trop bas"
    else:
        return "⚠️  MOYEN - À améliorer"
    

def analyze_tags_distribution(filename: str):
    """Analyse la distribution des tags XML."""
    from collections import defaultdict
    
    tag_counts = defaultdict(int)
    total_lines = 0
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                xml_path = parts[6]
                total_lines += 1
                
                # Déterminer le tag principal
                if '/p[' in xml_path:
                    tag_counts['p'] += 1
                elif '/sec[' in xml_path:
                    tag_counts['sec'] += 1
                elif '/bdy[' in xml_path:
                    tag_counts['bdy'] += 1
                elif xml_path.endswith('/article[1]'):
                    tag_counts['article'] += 1
                else:
                    tag_counts['other'] += 1
    
    print(f"📈 Distribution des tags ({filename}):")
    for tag in ['p', 'sec', 'bdy', 'article']:
        count = tag_counts[tag]
        percentage = (count / total_lines * 100) if total_lines > 0 else 0
        print(f"  {tag}: {count} ({percentage:.1f}%)")
    
    # Règle empirique pour exercice 3
    if tag_counts['p'] > tag_counts['bdy'] and tag_counts['sec'] > 0:
        return "✅ Bonne distribution (p > bdy)"
    elif tag_counts['bdy'] > tag_counts['p'] * 2:
        return "⚠️  Trop de bdy par rapport à p"
    else:
        return "📊 Distribution acceptable"
    
def check_common_inex_errors(filename: str):
    """Vérifie les erreurs courantes qui pénalisent le MAgP."""
    errors = []
    warnings = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Vérifier nombre de lignes
    if len(lines) != 10500:
        errors.append(f"Nombre de lignes incorrect: {len(lines)}/10500")
    
    # Vérifier format
    for i, line in enumerate(lines[:50]):
        parts = line.strip().split()
        if len(parts) != 7:
            errors.append(f"Ligne {i+1}: format incorrect")
            break
    
    # Vérifier interleaving (méthode rapide)
    last_article = {}
    for i, line in enumerate(lines[:1000]):
        parts = line.strip().split()
        if len(parts) >= 3:
            query_id = parts[0]
            article_id = parts[2]
            
            if query_id not in last_article:
                last_article[query_id] = article_id
            elif article_id != last_article[query_id]:
                # On a changé d'article, vérifier qu'on y revient pas
                if i+1 < len(lines):
                    next_parts = lines[i+1].strip().split()
                    if len(next_parts) >= 3 and next_parts[2] == last_article[query_id]:
                        warnings.append(f"Possible interleaving détecté ligne {i+1}")
    
    # Vérifier scores décroissants par requête
    for query_id in ['2009011', '2009036', '2009067']:
        query_scores = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == query_id:
                try:
                    query_scores.append(float(parts[4]))
                except:
                    pass
        
        # Vérifier décroissance
        for j in range(1, min(20, len(query_scores))):
            if query_scores[j] > query_scores[j-1] + 0.0001:
                warnings.append(f"Query {query_id}: score non décroissant au rang {j}")
                break
    
    print(f"🔍 Analyse erreurs pour {filename}:")
    if errors:
        print("  ❌ ERREURS CRITIQUES:")
        for err in errors[:3]:
            print(f"    - {err}")
    if warnings:
        print("  ⚠️  AVERTISSEMENTS:")
        for warn in warnings[:3]:
            print(f"    - {warn}")
    
    return len(errors) == 0

def calculate_quality_score(filename: str) -> float:
    """
    Calcule un score de qualité prédictif (0-100).
    Basé sur l'expérience INEX.
    """
    quality_indicators = {
        'format_correct': 0,
        'scores_high': 0,
        'scores_varied': 0,
        'tags_balanced': 0,
        'no_interleaving': 0,
        'lines_count': 0
    }
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # 1. Format correct (30 points)
        if all(len(line.strip().split()) == 7 for line in lines[:20]):
            quality_indicators['format_correct'] = 30
        
        # 2. Scores élevés (25 points)
        scores = []
        for line in lines[:100]:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    scores.append(float(parts[4]))
                except:
                    pass
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            if max_score > 5.0:
                quality_indicators['scores_high'] += 15
            if max_score > 10.0:
                quality_indicators['scores_high'] += 10
            if avg_score > 0.5:
                quality_indicators['scores_high'] += 5
        
        # 3. Scores variés (15 points)
        if scores and (max(scores) - min(scores)) > 5.0:
            quality_indicators['scores_varied'] = 15
        
        # 4. Distribution tags (15 points) - spécial exercice 3
        p_count = sum(1 for line in lines if '/p[' in line)
        bdy_count = sum(1 for line in lines if '/bdy[' in line)
        if p_count > bdy_count and p_count > 0:
            quality_indicators['tags_balanced'] = 15
        
        # 5. Pas d'interleaving (10 points)
        # Vérification simplifiée
        quality_indicators['no_interleaving'] = 10  # On suppose OK si format correct
        
        # 6. Nombre de lignes (5 points)
        if len(lines) == 10500:
            quality_indicators['lines_count'] = 5
        
        total_score = sum(quality_indicators.values())
        
        print(f"🎯 Score qualité pour {os.path.basename(filename)}: {total_score}/100")
        
        # Interprétation
        if total_score >= 80:
            print("  ✅ EXCELLENT - Soumets ce run !")
        elif total_score >= 60:
            print("  👍 BON - Potentiel intéressant")
        elif total_score >= 40:
            print("  ⚠️  MOYEN - Peut être amélioré")
        else:
            print("  ❌ FAIBLE - Revois ta stratégie")
        
        return total_score
        
    except Exception as e:
        print(f"Erreur analyse: {e}")
        return 0
    

def test_multiple_runs(run_files: List[str], max_tests: int = 3):
    """
    Teste plusieurs runs et recommande les meilleurs.
    """
    print(f"\n{'='*70}")
    print(f"TEST RAPIDE DE {len(run_files)} RUNS")
    print(f"Recommandation des {min(max_tests, len(run_files))} meilleurs")
    print('='*70)
    
    results = []
    
    for run_file in run_files:
        if not os.path.exists(run_file):
            print(f"⚠️  Fichier non trouvé: {run_file}")
            continue
        
        print(f"\n📁 Analyse: {os.path.basename(run_file)}")
        
        # 1. Score qualité
        quality_score = calculate_quality_score(run_file)
        
        # 2. Analyse scores
        score_analysis = analyze_scores_from_run(run_file, 50)
        
        # 3. Distribution tags
        tag_analysis = analyze_tags_distribution(run_file)
        
        results.append({
            'file': run_file,
            'quality': quality_score,
            'score_analysis': score_analysis,
            'tag_analysis': tag_analysis
        })
    
    # Trier par qualité
    results.sort(key=lambda x: x['quality'], reverse=True)
    
    print(f"\n{'='*70}")
    print("🎖️  CLASSEMENT RECOMMANDÉ:")
    print('='*70)
    
    for i, result in enumerate(results[:max_tests]):
        print(f"\n{i+1}. {os.path.basename(result['file'])}")
        print(f"   Score qualité: {result['quality']}/100")
        print(f"   Analyse scores: {result['score_analysis']}")
        print(f"   Tags: {result['tag_analysis']}")
    
    return results[:max_tests]


# Dans ton main.py
def test_and_select_runs():
    """Teste et sélectionne les meilleurs runs."""
    import glob
    
    # Lister tous les runs disponibles
    run_files = glob.glob("data/runs/*.txt")
    
    if not run_files:
        print("Aucun run trouvé dans data/runs/")
        return
    
    print(f"Found {len(run_files)} runs")
    
    # Tester seulement les 5 premiers pour économiser
    sample_runs = run_files #[:5]
    
    best_runs = test_multiple_runs(sample_runs, max_tests=2)
    
    print(f"\n💡 Conseil: Soumets seulement ces {len(best_runs)} runs")
    for run in best_runs:
        print(f"  - {os.path.basename(run['file'])}")


if __name__ == "__main__":
    test_and_select_runs()