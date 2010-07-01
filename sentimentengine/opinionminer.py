#! /usr/bin/env python
#
# Copyright (c) 2010 Okoye Chuka D.<okoye9@gmail.com>
#                    All rights reserved.
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
 
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

from nltk.corpus import movie_reviews
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.classify import decisiontree
from nltk.classify.util import accuracy
from cPickle import dump, load
from PyML import VectorDataSet, SparseDataSet
#from PyML import SVM
#from PyML.classifiers import loadSVM
from sys import exc_info
import speechtagger
import logger
import os

class OpinionMiner:

   def __init__(self, train_data=None):
      '''initializes classifier with important features required for
         classification. Accepts a list of tuples each of the form
         ('label',  'sentence belonging to this class')'''

      self.classifier = None
      c_dist = ConditionalFreqDist()
      f_dist = FreqDist()
      self._setDefaultInformativeFeatures()
      
      #Generate conditional freq and freq dist for each word
      if train_data:
         try:
            for (tag, sentence) in train_data:
               for word in word_tokenize(sentence.lower()):
                  c_dist[tag].inc(word)
                  f_dist.inc(word)
         except:
            logger.crawl_logs(["ERROR: ",str(exc_info()[0])])

      self._computeInstanceInformativeWords(c_dist, f_dist)


   @classmethod
   def _setDefaultInformativeFeatures(self):
      self._setSelectedPOSTags()
      self._setDefaultPositiveNegativeWords()

   @classmethod
   def _setSelectedPOSTags(self):

      buff = self._loadData('selective_pos.bin')

      if buff:
         self.selective_pos = buff
         return

      #First get all (word, tag) in corpuses
      sentences = brown.tagged_sents(simplify_tags=True)
      self.selected_tags = ["ADJ","ADV", "CNJ"]
      self.selective_pos = ConditionalFreqDist()
      temp_dist = ConditionalFreqDist()
      for sentence in sentences:
         for (word, tag) in sentence:
            if tag in self.selected_tags:
               temp_dist[tag].inc(str(word).lower())

      #Now, get the words with frequency > 10
      for category in temp_dist.conditions():
         fredist = temp_dist[category]
         for key in fredist.keys():
            if fredist[key] > 4:
               self.selective_pos[category].inc(key)

      self._saveData('selective_pos.bin',self.selective_pos)


   @classmethod
   def _setDefaultPositiveNegativeWords(self):

      buff1 = self._loadData('positive_adjectives.bin')
      buff2 = self._loadData('positive_adverbs.bin')
      buff3 = self._loadData('negative_adverbs.bin')
      buff4 = self._loadData('negative_adjectives.bin')

      if buff1 and buff2 and buff3 and buff4:
         self.positive_adjectives = buff1
         self.positive_adverbs = buff2
         self.negative_adverbs = buff3
         self.negative_adjectives = buff4
         return

      #First compile list of positive adjectives & adverbs
      #by initially tagging all positive sentences with POS tagger
      tagger = speechtagger.SpeechTagger()
      processed_sents = []
      self.positive_adjectives = set()
      self.positive_adverbs = set()
      self.negative_adverbs = set()
      self.negative_adjectives = set()

      train_bound_pos = int(len(movie_reviews.sents(categories="pos"))*0.8)
      train_bound_neg = int(len(movie_reviews.sents(categories="neg"))*0.8)

      #***************positive******************#
      for sentence in movie_reviews.sents(categories="pos")[:train_bound_pos]:
         concat_sent = (" ".join(sentence)).lower()
         processed_sents.append(concat_sent)

      tagged_sents = tagger.tag(processed_sents) #TODO: Save to file

      for sentence in tagged_sents:
         for (word, tag) in sentence:
            if tag is 'ADJ' or word in self.selective_pos['ADJ']:
               self.positive_adjectives.add(word)
            elif tag is 'ADV' or word in self.selective_pos['ADV']:
               self.positive_adverbs.add(word)
         
      #**************negative*****************#
      processed_sents = []
      for sentence in movie_reviews.sents(categories="neg")[:train_bound_neg]:
         concat_sent = (" ".join(sentence)).lower()
         processed_sents.append(concat_sent)

      tagged_sents = tagger.tag(processed_sents) #TODO: Save to file

      for sentence in tagged_sents:
         for (word, tag) in sentence:
            if tag is 'ADJ' or word in self.selective_pos['ADJ']:
               self.negative_adjectives.add(word)
            elif tag is 'ADV' or word in self.selective_pos['ADV']:
               self.negative_adverbs.add(word)

      self._saveData('positive_adjectives.bin',self.positive_adjectives)
      self._saveData('positive_adverbs.bin', self.positive_adverbs)
      self._saveData('negative_adjectives.bin', self.negative_adjectives)
      self._saveData('negative_adverbs.bin', self.negative_adverbs)


   @classmethod
   def _computeInstanceInformativeWords(self, cf_dist=None, f_dist=None):
      '''using chi_square distribution, computes and returns the words
         that contribute the most significant info. That is words that
         are mostly unique to each set(positive, negative)'''

      buff = self._loadData('informative_words.bin')
      if buff:
         self.informative_words = buff
         return
      elif cf_dist == None or f_dist == None:
         self.informative_words = dict()
         return

      total_num_words = f_dist.N()
      total_positive_words = cf_dist["positive"].N()
      total_negative_words = cf_dist["negative"].N()
      words_score = dict()
        
      for word in f_dist.keys():
         pos_score = BigramAssocMeasures.chi_sq(cf_dist["positive"][word],
                                    (f_dist[word], total_positive_words),
                                    total_num_words)
         neg_score = BigramAssocMeasures.chi_sq(cf_dist["negative"][word],
                                    (f_dist[word], total_negative_words),
                                    total_num_words)


         words_score[word] = pos_score + neg_score

      #Return 0.001% most useful words 
      self.informative_words = dict(sorted(words_score.iteritems(),
                                 key=lambda (word, score): score,
                                 reverse=True)[:int(0.0001*len(words_score))])

      self._saveData('informative_words.bin',self.informative_words)

   def trainClassifier(self, train_data):
      '''trains a decision tree, svm and naive bayes classifier'''

      feature_set = []
      labels = []

      self.classifier = self._loadData('opinion_classifier.bin')
      if self.classifier:
         print 'Loaded saved classifier'
         return
      
      print 'Extracting features from training data'
      for sentence, label in train_data:
         feat = self.getFeatures(sentence)
         feature_set.append((feat,label))
         
      #SVM Implementation
      #vector_data = VectorDataSet(feature_set, L=labels) #Linear Discriminant
      #svm = SVM() 
      #svm.train(vector_data, saveSpace=False)
      #svm.save('opinion-classifier')

      #TODO: Remove this implementation for hello chair demo.
      print feature_set[:10]
      self.classifier = naivebayes.DecisionTreeClassifier.train(feature_set)

      self._saveData('opinion_classifier.bin', self.classifier)

   def classify(self, sentence):
      
      return self.classifier.classify(self.getFeatures(sentence))

   def getFeatures(self, sentence):

      conjunc = self.getConjunctionFeats(sentence)
      norm_score = self.computeNormalizedScores(sentence)
      trans_feat = self.getTransitiveFeatures(sentence)
      inst_feat = self.getSpecificInstanceFeatures(sentence)
            
      temp = dict()

      #getConjunctionFeats Result
      #for value in conjunc.values():
      #   temp.append(value)

      #computeNormalizedScores Result
      for key in norm_score:
         temp[key] = norm_score[key]

      #getTransitiveFeatures Result
      for key in trans_feat:
         temp[key] = trans_feat[key]

      #getSpecificInstanceFeat Result
      for key in inst_feat:
         temp[key] = inst_feat[key]

      return temp

   @classmethod
   def _saveData(self, filename, data):
      absolute_path = os.path.join(os.path.dirname(__file__),filename)
      FILE = open(absolute_path, "wb")
      dump(data, FILE, 1)
      FILE.close()

   @classmethod
   def _loadData(self, filename):
      absolute_path = os.path.join(os.path.dirname(__file__),filename)
      try:
         FILE = open(absolute_path, "rb")
         buff = load(FILE)
         FILE.close()
         return buff
      except:
         logger.crawl_logs("A problem occured when reading  "+filename)
         return None

   #################Features######################
   def getConjunctionFeats(self, sentence):
      '''computes the presence or absence of certain conjunctions
         given a sentence in string format'''

      tokenized_sent = word_tokenize(sentence.lower())
      word_dict = dict()
      sentence_features = dict()

      for word in tokenized_sent: #speeds up search
         word_dict[word] = True

      for conjunction in self.selective_pos['CNJ'].keys():
         if conjunction in word_dict:
            sentence_features[conjunction] = 1
         else:
            sentence_features[conjunction] = 0

      return sentence_features

   def computeNormalizedScores(self, sentence):
      '''computes normalized pos & neg scoresfor adjectives and
         adverbs and returns a tuple (pos_adj_score, pos_adv_score...)'''

      #We have two options:
      #1. Do POS Tagging then compute score on relevat pos tags or,
      #2. Convert sentence to dict the check for presence.

      #Currently, this module uses method 2.
      total_adjectives_pos = len(self.positive_adjectives)
      total_adverbs_pos = len(self.positive_adverbs)
      total_adjectives_neg = len(self.negative_adjectives)
      total_adverbs_neg = len(self.negative_adverbs)
      pos_adj_score = 0
      pos_adv_score = 0
      neg_adj_score = 0
      neg_adv_score = 0

      word_dict = dict() #TODO Refactor this pre-processing aspect
      tokenized_sent = word_tokenize(sentence.lower())

      for word in tokenized_sent:
         word_dict[word] = True

      #Now compute scores for positive values
      for key in word_dict:
         if key in self.positive_adjectives:
            pos_adj_score += 1
         elif key in self.positive_adverbs: #mutally exclusive
            pos_adv_score += 1

      pos_adj_score = float(pos_adj_score)/float(total_adjectives_pos)
      pos_adv_score = float(pos_adv_score)/float(total_adverbs_pos)

      #Compute scores for negative values
      for key in word_dict:
         if key in self.negative_adjectives:
            neg_adj_score += 1
         elif key in self.negative_adverbs:
            neg_adv_score += 1
      
      neg_adj_score = float(neg_adj_score)/float(total_adjectives_neg)
      neg_adv_score = float(neg_adv_score)/float(total_adverbs_neg)

      normalized_score = dict()
      normalized_score = {
                           "pos_adj":pos_adv_score,
                           "pos_adv":pos_adv_score,
                           "neg_adj":neg_adj_score,
                           "neg_adv":neg_adv_score
                         }

      return normalized_score
      
   def getTransitiveFeatures(self, sentence):
      wordlist = set(['however','but','nevertheless','still',
                     'withal','yet','all','same', 'even' 'so',
                     'nonetheless', 'not','standing', 'notwithstanding'
                     'evenso', 'none','less'])

      tokenized_sent = word_tokenize(sentence.lower())
      features = dict()
      word_dict = dict()

      for word in tokenized_sent:
         word_dict[word] = True

      for word in wordlist:
         if word in word_dict:
            features[word] = 1
         else:
            features[word] = 0

      return features

   def getSpecificInstanceFeatures(self, sentence):
      word_dict = dict()
      tokenized_sent = word_tokenize(sentence.lower())
      features = dict()

      for word in tokenized_sent:
         word_dict[word] = True

      for word in self.informative_words:
         if word in word_dict:
            features[word] = 1
         else:
            features[word] = 0

      return features


   ##############End Features#################
