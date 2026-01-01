import os
import re

def analyze_xml_namespaces(xml_file_path):
    """Analyse les namespaces utilisés dans un fichier XML"""
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        content = f.read(5000)  # Lire les premiers 5000 caractères
    
    # Chercher les déclarations xmlns
    namespace_pattern = r'xmlns(?::(\w+))?=\s*["\']([^"\']+)["\']'
    matches = re.findall(namespace_pattern, content)
    
    namespaces = {}
    for prefix, uri in matches:
        if prefix:
            namespaces[prefix] = uri
        else:
            namespaces['default'] = uri  # Namespace par défaut
    
    # Chercher les balises avec préfixes
    tag_pattern = r'<(\w+):\w+'
    prefixed_tags = re.findall(tag_pattern, content)
    
    return {
        'namespaces': namespaces,
        'prefixed_tags': list(set(prefixed_tags)),
        'has_namespaces': len(namespaces) > 0
    }

# Test sur vos fichiers
sample_files = ["data/file1.xml", "data/file2.xml"]
for file in sample_files:
    if os.path.exists(file):
        result = analyze_xml_namespaces(file)
        print(f"{file}: {result['namespaces']}")

def main():
    x = 2

if __name__=="__main__":

    data_file_path = "data/Practice_05_data/XML-Coll-withSem"
    xml_files = []
    
    i = 0
    for root_dir, dirs, files in os.walk(data_file_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root_dir, file))

                results = analyze_xml_namespaces(xml_files[i])
                print("namespaces : ", results["namespaces"],
                    "\n\nprefixed_tags : ", results['prefixed_tags'], 
                    "\n\nhas_namespaces : ", results['has_namespaces'])
                
                i += 1

    

    