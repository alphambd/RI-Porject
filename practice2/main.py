import gzip
import os

from matplotlib import pyplot as plt

from indexer import InvertedIndex
import time

#from practice2.portestemmer import PorterStemmer
from portestemmer import PorterStemmer

def main():
    time_exec = [[],[],[]]
    counts_terms = [[],[],[]]
    counts_words = [[],[],[]]
    counts_chars = [[],[],[]]
    docs_ids = [[],[],[]]
    #  Créer et construire l'index
    index = InvertedIndex()

    print("\n\t*************************************************")
    print("\t/             INVERTED INDEX                    /")
    print("\t*************************************************")

    data_path = "practice2_data"
    #for file in os.listdir("data"):
    for file in os.listdir(data_path):
        print(file)
        start = time.time()
        #index.build_from_file("data/"+file)
        time_exec[0].append(time.time() - start)
        index.build_from_file(os.path.join(data_path, file))
        #(count_terms, count_words,count_chars, doc_ids)  = index.get_data("data/" + file)
        (count_terms, count_words,count_chars, doc_ids)  = index.get_data(os.path.join(data_path, file))
        counts_terms[0].append(count_terms)
        counts_words[0].append(count_words)
        counts_chars[0].append(count_chars)
        docs_ids[0].append(doc_ids)

    print("\n\t*************************************************")
    print("\t/        INVERTED INDEX WITH STOP WORDS         /")
    print("\t*************************************************")
    
    index.stop_word_active = True
    for file in os.listdir("data"):
        print(file)
        start = time.time()
        index.build_from_file(os.path.join(data_path, file))
        time_exec[1].append(time.time() - start)
        (count_terms, count_words,count_chars, doc_ids)  = index.get_data(os.path.join(data_path, file))
        counts_terms[1].append(count_terms)
        counts_words[1].append(count_words)
        counts_chars[1].append(count_chars)
        docs_ids[1].append(doc_ids)

        
    print("\n\t*************************************************")
    print("\t/  INVERTED INDEX WITH STOP WORDS AND STEMMER   /")
    print("\t*************************************************")

    index.stemmer_active = True
    for file in os.listdir("data"):
        print(file)
        start = time.time()
        index.build_from_file(os.path.join(data_path, file))
        time_exec[2].append(time.time() - start)
        (count_terms, count_words,count_chars, doc_ids)  = index.get_data(os.path.join(data_path, file))
        counts_terms[2].append(count_terms)
        counts_words[2].append(count_words)
        counts_chars[2].append(count_chars)
        docs_ids[2].append(doc_ids)
    

    print("\n\t***************************************************")
    print("\n\tdata")
    print(counts_terms)
    print(counts_words)
    print(counts_chars)
    #print(counts_chars)
    print(docs_ids)
    print(time_exec)
    plt.plot(counts_words[0],time_exec[0], '-bo')
    plt.plot(counts_words[0],time_exec[1], '-ro')
    plt.plot(counts_words[0],time_exec[2], '-go')
    plt.title("Indexing time vs size of the coll")
    plt.xlabel("#mots")
    plt.ylabel("sec")
    plt.show()
    
    avg_count_therm = [[],[],[]]
    for i in range(len(counts_terms)):
        for i2 in range(len(counts_terms[0])):
            avg_count_therm[i].append(counts_terms[i][i2]/len(docs_ids[i][i2]))

    plt.plot(counts_words[0],avg_count_therm[0], '-bo')
    plt.plot(counts_words[0],avg_count_therm[1], '-ro')
    plt.plot(counts_words[0],avg_count_therm[2], '-go')
    plt.title("avg doc length vs size of the coll")
    plt.xlabel("#mots")
    plt.ylabel("#terms")
    plt.legend(['Base', 'Stopwords', 'Stemmer'])
    plt.show()

    avg_count_char = [[],[],[]]
    for i in range(len(counts_terms)):
        for i2 in range(len(counts_terms[0])):
            avg_count_char[i].append(counts_chars[i][i2]/counts_terms[i][i2])

    plt.plot(counts_words[0],avg_count_char[0], '-bo')
    plt.plot(counts_words[0],avg_count_char[1], '-ro')
    plt.plot(counts_words[0],avg_count_char[2], '-go')
    plt.title("avg terms length vs size of the coll")
    plt.xlabel("#mots")
    plt.ylabel("#chars")
    plt.legend(['Base', 'Stopwords', 'Stemmer'])
    plt.show()

    plt.plot(counts_words[0],counts_terms[0], '-bo')
    plt.plot(counts_words[0],counts_terms[1], '-ro')
    plt.plot(counts_words[0],counts_terms[2], '-go')
    plt.title("vocabulary size vs size of the coll")
    plt.xlabel("#mots")
    plt.ylabel("#terms")
    plt.legend(['Base', 'Stopwords', 'Stemmer'])
    plt.show()
    
if __name__ == "__main__":
    main()
