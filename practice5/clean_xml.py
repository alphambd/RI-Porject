import os

XML_DIR = "data/Practice_05_data/XML-Coll-withSem"

replacements = {
    "&nbsp;": " ",
    "&mdash;": "—",
    "&ndash;": "–"
}

for filename in os.listdir(XML_DIR):
    if filename.endswith(".xml"):
        path = os.path.join(XML_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for old, new in replacements.items():
            text = text.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

print("Tous les fichiers XML ont été nettoyés !")
