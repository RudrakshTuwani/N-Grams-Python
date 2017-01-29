#!/home/rudraksh/anaconda2/bin/python
# -*- coding: utf-8 -*-

import re
import codecs
import pickle
from math import log, exp

class Vocab():
    """ A general vocabulary class which can be used for purposes
        much more diverse than simply serving unigrams. """
    
    def __init__(self):
        self._vocab = set()
        self._unigrams = dict()
        self._word_to_id = dict()
        self._id_to_word = dict()
        self._n = len(self._word_to_id)
        
    def read_file(self, file_path, encoding = 'utf-8'):
        """
            This function can be called again and again to read from multiple files,
            into the same dictionary.
        """
        
        # Read from disk, one line at a time.
        with codecs.open(file_path,'r', encoding) as f:
            for line in f:
                for word in line.split():                        
                    if word not in self._vocab:
                        self._vocab.add(word)
                        self._unigrams[word] = 1
                    else:
                        self._unigrams[word] += 1
    
    
    def update_vocab(self,word):
        """ Update vocabulary with a single word. """
        assert word not in self._vocab, "Word already exists in vocabulary"
        self._vocab.add(word)
        self._unigrams[word] = 1

    def _mappings(self):
        """ Creates word to id and id to word mappings. """
        
        # Check if word_to_id has been already created
        if self._n == 0:
            for i, word in enumerate(self._vocab):
                self._word_to_id[word] = i
                self._id_to_word[i] = word
                self._n = len(self._word_to_id)
                
        else:
            if self._n != len(self._vocab):
                n = len(self._word_to_id)
                
                existing = self._word_to_id.keys()
                for i, word in enumerate([word for word in self._vocab if word not in existing]):
                    self._word_to_id[word] = n+i
                    self._id_to_word[n+i] = word
                    
                self._n +=  i+1
                
    @property
    def word_to_id(self):
        
        self._mappings()
        return self._word_to_id
    
    @property
    def id_to_word(self):
        self._mappings()
        return self._id_to_word
        
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def unigrams(self):
        return self._unigrams
    
    @property
    def total_num_words(self):
        return len(self._vocab)
                            
def find_ngrams(input_list, n):
	""" Function for making N-grams """
    return zip(*[input_list[i:] for i in range(n)])


class Bigrams():
    """ Class for dealing with bigrams."""
    def __init__(self, vocab):
        self.vocab = vocab
        self._bigrams = {}
        self._num_bigrams = 0
        self._unique_bigrams = 0
        self._bigram_vocab = set()
        
    def read_file(self, file_path, encoding = 'utf-8'):
        """
            Refer to the PreProcessing function to see what kind of preprocessing 
            is performed. Usually, in this case, you would like to preprocess the file 
            somewhere else, splitting paragraphs into individual lines and then saving it 
            on disk. The bigrams can then be made by going through the file line by line.
        """
        
        # Read from disk, one line at a time.
        with codecs.open(file_path,'r', encoding) as f:
            for line in f:
                words = line.split()
                
                if len(words) > 1:
                    bigrams = find_ngrams(words,2)
                    
                    for bigram in bigrams:
                        if bigram[0] in self._bigram_vocab:
                            
                            try:
                                self._bigrams[bigram[0]][bigram[1]] += 1
                                self._num_bigrams += 1
                            except KeyError:
                                self._bigrams[bigram[0]][bigram[1]] = 1
                                self._num_bigrams += 1
                                self._unique_bigrams += 1
                                
                                
                        else:
                            self._bigrams[bigram[0]] = {bigram[1]:1}
                            self._bigram_vocab.add(bigram[0])
                            self._num_bigrams += 1
                            self._unique_bigrams += 1
                            
                            
    def update_bigrams(self,tuple_bigram):
        """ Adds new bigrams to the model. """
        if tuple_bigram[0] in self._bigram_vocab:
            try:
                self._bigrams[tuple_bigram[0]][tuple_bigram[1]] += 1
                self._num_bigrams += 1
            except KeyError:
                self._bigrams[tuple_bigram[0]][tuple_bigram[1]] = 1
                self._num_bigrams += 1
                self._unique_bigrams += 1
        else:
            self._bigrams[tuple_bigram[0]] = {tuple_bigram[1] : 1}
            self._bigram_vocab.add(tuple_bigram[0])
            self._num_bigrams += 1
            self._unique_bigrams += 1
            
    def new_bigrams_bigrams(self, _vocab = None, method='simple', min_word_count = 0.0):
        """ Function to smoothen bigrams.
            Args:
        
            method -> Three methods are supported at present.
            - 'simple' : Normalizes the bigrams by dividing by unigram count of word[0]
            - 'add1' : Simply adds 1 to the count of all possible bigrams and normalizes
                       similar to method 'simple'
            - 'unk' : Replaces words whose count falls below the threshold with <unk> tag
                      and recomputes bigrams and unigrams.
        
            min_word_count -> Minimum count for method 'unk'
            
            _vocab: Specify some other vocabulary for smoothing the bigrams.
            
            """
        if _vocab is None:
            _vocab = self.vocab

        # Simple Normalizing criteria
        if method == 'simple':
            unigrams = _vocab.unigrams
            
            new_bigrams = {}
            for word, bigram in self._bigrams.items():
                normalizing_factor = unigrams[word]

                new_bigrams[word] = {}

                for wrd, count in bigram.items():
                    new_bigrams[word][wrd] = count / float(normalizing_factor)
            
            
            n = float(_vocab.total_num_words)
            new_unigrams = {word:count/n for word,count in unigrams.items()}
            return new_bigrams, new_unigrams
        
        
        # Add one smoothing criteria (VERY MEMORY INTENSIVE [V * V * 1])
        if method == 'add1':
            unigrams = _vocab.unigrams
            vocabulary = _vocab.vocab
            
            new_bigrams = {}
            for word in self.bigrams.keys():
                normalizing_factor = unigrams[word] + _vocab.total_num_words
                
                new_bigrams = self.bigrams[word]
                for wrd in vocabulary:
                    try:
                        new_bigrams[wrd] += 1
                        new_bigrams[wrd] /= float(normalizing_factor)
                    except:
                        new_bigrams[wrd] = 1
                        new_bigrams[wrd] /= float(normalizing_factor)
                        
                new_bigrams[word] = new_bigrams
            
            n = float(_vocab.total_num_words)
            new_unigrams = {word:(count)/n for word,count in unigrams.items()}
            return new_bigrams, new_unigrams
                            
        # Replacing words falling below a threshold with <unk> tag and recomputing
        # bigrams.
        if method == 'unk':
            unigrams = _vocab.unigrams
            new_unigrams = {}
            unk = set()
            
            # Replace words which occur less with unknown words.
            for word,count in unigrams.items():
                if count <= min_word_count:
                    unk.add(word)
                else:
                    new_unigrams[word] = count
                    
            # Make corrresponding change in bigrams
            new_bigrams = {}
            key_b = {}
            
            #1 Merge bigrams of less frequent words together.
            for word in unk:
                try: #Since the list was created from unigrams, some bigrams may be absent
                    old = self.bigrams[word]
                except:
                    continue
                
                for word,count in old.items():
                    try:
                        key_b[word] += count
                    except KeyError:
                        key_b[word] = count
            
            new_bigrams['<unk>'] = key_b
            
            #2 Iterate over all the bigrams, replacing words with <unk> wherever necessary
            for word,word_b in self.bigrams.items():
                if word in unk:
                    pass
                else:
                    for key, count in word_b.items():
                        new = {}
                        
                        if key in unk:
                            
                            try:
                                new['<unk>'] += count
                            except KeyError:
                                new['<unk>'] = count
                        
                        else:
                            new[key] = count
                    new_bigrams[word] = new
                
            #3 Do the same for <unk>.
            new = {}
            for word, count in new_bigrams['<unk>'].items():
                if word in unk:
                    try:
                        new['<unk>'] += count
                    except KeyError:
                        new['<unk>'] = count
                        
                else:
                    new[key] = count
            new_bigrams['<unk>'] = new
            return new_bigrams, new_unigrams
            
    def num_bigrams(self,return_ = False):
        print('The number of unique bigrams is {}.'.format(self._unique_bigrams))
        print('The total number of bigrams are {}.'.format(self._num_bigrams))
        
        if return_:
            return self._unique_bigrams, self._num_bigrams
        
    @property
    def bigrams(self):
        return self._bigrams

class Model():
    """ Contains various measures for both evaluation and prediction using existing n-grams."""
    
    def __init__(self, unigrams, bigrams):
        """ Enter the unigrams and bigrams whose counts have been smoothed to probabilities. """
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.calc = False
        
    def score_sentence(self, sentence, preprocess=False):
        """  Returns the probability of observing the given sentence. """
        
        if preprocess:
            pass  # Add your custom functon here!
   
        words = sentence.split()
        pairs = zip(*[words[i:] for i in range(2)]) #skips the first word.
        
        # For the first word.
        prob = log(self.unigrams[words[0]])
            
        # Second word onwards.
        for pair in pairs:
            prob += log(self.bigrams[pair[0]][pair[1]])
            
        return exp(prob)
    
    def predict(self, sentence, n, preprocess=False):
        """ Predicts the next word in the sequence. """
        if not self.calc:
            self.top()
        
        if preprocess:
            pass  # Add your custom functon here!
        
        words = sentence.split()
        
        if len(words) > 1:
            # Get last two words of the sequence.
            l2 = words[-2:]
            return self.topn[l2[1]][:n]
        else:
            return self.topn[words[0]][:n]                
    def top(self):
        """ Sorts the bigram for each word by probability"""
        self.topn = {}       
        for key,words in self.bigrams.items():
            sorted_words = sorted(words.items(), key=operator.itemgetter(1),
                                  reverse=True)
            
            self.topn[key] = sorted_words
        self.calc = True
