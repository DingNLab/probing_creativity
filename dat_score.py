"""Compute score for Divergent Association Task,
a quick and simple measure of creativity"""

import io
import re
import itertools
import numpy
import scipy.spatial.distance
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText

class Model:
    """Create model to compute DAT"""

    def __init__(self, model="glove.840B.300d.txt", dictionary="words.txt", model_type="glove", pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary"""

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))
        
        # Join words with model
        vectors = {}
        if model_type=='glove':
            with open(model, "r", encoding="utf8") as f:
                for line in f:
                    tokens = line.split(" ")
                    word = tokens[0]
                    if word in words:
                        vector = numpy.asarray(tokens[1:], "float32")
                        vectors[word] = vector
        elif model_type=='word2vec':
            model = KeyedVectors.load_word2vec_format(model, binary=True)
            for word in words:
                try: 
                    vectors[word] = model[word]
                except:
                    None
        elif model_type=='fasttext':
            fin = io.open(model, 'r', encoding='utf-8', newline='\n', errors='ignore')
            for line in fin:
                tokens = line.rstrip().split(' ')
                if tokens[0] in words:
                    vectors[tokens[0]] = numpy.array([float(i) for i in tokens[1:]])
        self.vectors = vectors


    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None # Word too short

        # Generate candidates for possible compound words
        # "valid" -> ["valid"]
        # "cul de sac" -> ["cul-de-sac", "culdesac"]
        # "top-hat" -> ["top-hat", "tophat"]
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        for cand in candidates:
            if cand in self.vectors:
                return cand # Return first word that is in model
        return None # Could not find valid word


    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        return scipy.spatial.distance.cosine(self.vectors[word1], self.vectors[word2])
        return scipy.spatial.distance.cosine(self.vectors.get(word1), self.vectors.get(word2))


    def dat(self, words, minimum=7):
        """Compute DAT score"""
        # Keep only valid unique words
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return None # Not enough valid words

        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = self.distance(word1, word2)
            distances.append(dist)

        # Compute the DAT score (average semantic distance multiplied by 100)
        return (sum(distances) / len(distances)) * 100



class Model_old:
    """Create model to compute DAT"""

    def __init__(self, model="glove.840B.300d.txt", dictionary="words.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary"""

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        # Join words with model
        vectors = {}
        with open(model, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                word = tokens[0]
                if word in words:
                    vector = numpy.asarray(tokens[1:], "float32")
                    vectors[word] = vector
        self.vectors = vectors


    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None # Word too short

        # Generate candidates for possible compound words
        # "valid" -> ["valid"]
        # "cul de sac" -> ["cul-de-sac", "culdesac"]
        # "top-hat" -> ["top-hat", "tophat"]
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        for cand in candidates:
            if cand in self.vectors:
                return cand # Return first word that is in model
        return None # Could not find valid word


    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""

        return scipy.spatial.distance.cosine(self.vectors.get(word1), self.vectors.get(word2))


    def dat(self, words, minimum=7):
        """Compute DAT score"""
        # Keep only valid unique words
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return None # Not enough valid words

        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = self.distance(word1, word2)
            distances.append(dist)

        # Compute the DAT score (average semantic distance multiplied by 100)
        return (sum(distances) / len(distances)) * 100
