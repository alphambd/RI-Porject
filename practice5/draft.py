
def find_all_entities_in_file(xml_file_path):
    """Trouve TOUTES les entités dans un fichier pour débogage"""
    with open(xml_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Trouver toutes les entités &xxx; ou &#xxx; ou &#xXXX;
    entity_pattern = r'&(?:[a-zA-Z]+|\#\d+|\#x[0-9a-fA-F]+);'
    all_entities = re.findall(entity_pattern, content)
    
    # Compter les occurrences
    entity_counts = {}
    for entity in all_entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    # Afficher les plus fréquentes
    print(f"\nEntités dans {os.path.basename(xml_file_path)}:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {entity}: {count} fois")
    
    return entity_counts

def find_all_entities_in_collection(xml_dir, sample_size=None):
    """
    Trouve TOUTES les entités dans toute la collection XML
    et identifie les entités non traitées
    
    Args:
        xml_dir: Répertoire contenant les fichiers XML
        sample_size: Nombre de fichiers à analyser (None pour tous)
    """
    import glob
    import os
    import re
    from collections import defaultdict
    
    # Lister tous les fichiers XML
    xml_files = glob.glob(os.path.join(xml_dir, "**/*.xml"), recursive=True)
    
    if sample_size:
        xml_files = xml_files[:sample_size]
    
    print(f"Analyse des entités dans {len(xml_files)} fichiers XML...")
    
    # Dictionnaire pour compter toutes les entités
    global_entity_counts = defaultdict(int)
    
    # Dictionnaire pour les fichiers contenant chaque entité
    entity_files = defaultdict(set)
    
    # Ensemble de toutes les entités uniques
    all_unique_entities = set()
    
    # Ensemble des entités déjà traitées dans decode_all_entities_exhaustive
    # (À mettre à jour avec votre liste actuelle)
    processed_entities = {
        '&nbsp;', '&#32;', '&amp;', '&mdash;', '&ndash;',
        '&apos;', '&quot;', '&lt;', '&gt;', '&ensp;', '&emsp;', 
        '&thinsp;', '&hellip;', '&ldquo;', '&rdquo;', '&lsquo;',
        '&rsquo;', '&laquo;', '&raquo;', '&middot;', '&bull;',
        '&star;', '&phone;', '&copy;', '&reg;', '&trade;',
        '&plusmn;', '&times;', '&divide;', '&ne;', '&le;',
        '&ge;', '&asymp;', '&infin;', '&ang;', '&perp;',
        '&parallel;', '&sum;', '&prod;', '&int;', '&radic;',
        '&part;', '&nabla;', '&isin;', '&notin;', '&cap;',
        '&cup;', '&sub;', '&sup;', '&sube;', '&supe;',
        '&oplus;', '&otimes;', '&and;', '&or;', '&there4;',
        '&because;', '&forall;', '&exist;', '&empty;', '&nabla;',
        '&ni;', '&minus;', '&lowast;', '&prop;', '&ang;',
        '&sim;', '&cong;', '&equiv;', '&nsub;', '&sdot;',
    }
    
    # Ajouter les entités numériques (&#xxx; et &#xXXX;)
    numeric_pattern = re.compile(r'&#(\d+);|&#x([0-9a-fA-F]+);')
    
    file_count = 0
    for i, xml_file in enumerate(xml_files):
        try:
            with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Trouver toutes les entités
            entity_pattern = r'&(?:[a-zA-Z]+|\#\d+|\#x[0-9a-fA-F]+);'
            all_entities = re.findall(entity_pattern, content)
            
            # Mettre à jour les compteurs
            for entity in all_entities:
                global_entity_counts[entity] += 1
                entity_files[entity].add(os.path.basename(xml_file))
                all_unique_entities.add(entity)
            
            file_count += 1
            
            if file_count % 100 == 0:
                print(f"  Analysé {file_count} fichiers...")
                
        except Exception as e:
            print(f"  Erreur lecture {xml_file}: {e}")
    
    print(f"\n{'='*80}")
    print("ANALYSE COMPLÈTE DES ENTITÉS DANS LA COLLECTION")
    print(f"{'='*80}")
    
    # Afficher toutes les entités triées par fréquence
    print(f"\nToutes les entités trouvées ({len(all_unique_entities)} uniques, {sum(global_entity_counts.values())} total):")
    print("-" * 80)
    
    for entity, count in sorted(global_entity_counts.items(), key=lambda x: x[1], reverse=True):
        # Identifier si l'entité est déjà traitée
        is_processed = entity in processed_entities or numeric_pattern.match(entity)
        status = "[TRAITÉ]" if is_processed else "[NON TRAITÉ]"
        
        # Afficher quelques fichiers contenant cette entité
        sample_files = list(entity_files[entity])[:3]
        files_info = f" (fichiers: {', '.join(sample_files)}" + ("...)" if len(entity_files[entity]) > 3 else ")")
        
        print(f"  {entity:15} {count:8d} fois {status:12} {files_info}")
    
    # Identifier les entités NON TRAITÉES
    untreated_entities = []
    for entity in all_unique_entities:
        if entity not in processed_entities and not numeric_pattern.match(entity):
            untreated_entities.append(entity)
    
    if untreated_entities:
        print(f"\n{'!'*80}")
        print(f"ATTENTION: {len(untreated_entities)} ENTITÉS NON TRAITÉES DÉTECTÉES")
        print(f"{'!'*80}")
        
        print("\nEntités non traitées (à ajouter à votre dictionnaire):")
        print("-" * 80)
        
        for entity in sorted(untreated_entities):
            count = global_entity_counts[entity]
            sample_files = list(entity_files[entity])[:2]
            files_info = f" (ex: {', '.join(sample_files)})"
            print(f"  '{entity}': {count} occurrences{files_info}")
        
        # Générer le code Python pour ajouter ces entités
        print(f"\n{'='*80}")
        print("CODE PYTHON POUR AJOUTER LES ENTITÉS MANQUANTES:")
        print(f"{'='*80}")
        print("\nAjoutez ceci à votre dictionnaire 'common_missing':")
        print("{")
        for entity in sorted(untreated_entities):
            # Essayer de deviner la valeur de remplacement
            # Pour les entités courantes
            if entity.startswith('&#') and ';' in entity:
                # Entité numérique - html.unescape devrait la gérer
                print(f"    '{entity}': '?',  # Entité numérique")
            else:
                # Pour les entités nommées, on ne connaît pas la valeur
                print(f"    '{entity}': '?',  # À déterminer")
        print("}")
    
    # Statistiques résumées
    print(f"\n{'='*80}")
    print("STATISTIQUES RÉSUMÉES:")
    print(f"{'='*80}")
    
    total_entities = sum(global_entity_counts.values())
    processed_count = sum(count for entity, count in global_entity_counts.items() 
                         if entity in processed_entities or numeric_pattern.match(entity))
    
    print(f"Fichiers analysés: {file_count}")
    print(f"Entités totales: {total_entities}")
    print(f"Entités uniques: {len(all_unique_entities)}")
    print(f"Entités traitées: {processed_count} ({processed_count/total_entities*100:.1f}%)")
    print(f"Entités non traitées: {total_entities - processed_count} ({(total_entities - processed_count)/total_entities*100:.1f}%)")
    
    # Top 20 des entités les plus fréquentes
    print(f"\nTop 20 des entités les plus fréquentes:")
    print("-" * 80)
    top_20 = sorted(global_entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (entity, count) in enumerate(top_20, 1):
        percentage = count / total_entities * 100
        is_processed = entity in processed_entities or numeric_pattern.match(entity)
        status = "✓" if is_processed else "✗"
        print(f"  {i:2d}. {entity:15} {count:8d} ({percentage:5.1f}%) {status}")
    
    return {
        'entity_counts': dict(global_entity_counts),
        'entity_files': {k: list(v) for k, v in entity_files.items()},
        'untreated_entities': untreated_entities,
        'all_unique_entities': list(all_unique_entities)
    }
