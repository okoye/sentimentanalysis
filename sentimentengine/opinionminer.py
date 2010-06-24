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
import speechtagger

class OpinionMiner:

   def __init__(self, train_data=None):
      '''initializes classifier with important features required for
         classification. Accepts a list of tuples each of the form
         ('label',  'sentence belonging to this class')'''

      self.classifier = None
      self._setDefaultInformativeFeatures()
      
      #Generate conditional frequency data for each word
      if train_data:
         cond_dist = ConditionalFreqDist()
         freq_dist = FreqDist()
         try:
            for (tag, sentence) in train_data:
               for word in word_tokenize(sentence.lower()):
                  cond_dist[tag].inc(word)





   @classmethod
   def _setDefaultInformativeFeatures(self):
      self._setSelectedPOSTags()
      self._setDefaultPositiveNegativeWords()
      #self._computeSpecificInstanceInformativeWords()

   @classmethod
   def _setSelectedPOSTags(self):

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


   @classmethod
   def _setDefaultPositiveNegativeWords(self):

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

      tagged_sents = tagger.tag(processed_sents)
      print tagged_sents

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

      tagged_sents = tagger.tag(processed_sents)

      for sentence in tagged_sents:
         for (word, tag) in sentence:
            if tag is 'ADJ' or word in self.selective_pos['ADJ']:
               self.negative_adjectives.add(word)
            elif tag is 'ADV' or word in self.selective_pos['ADV']:
               self.negative_adverbs.add(word)
      print 'Some positive adjectives:',self.positive_adjectives[:100]
      print 'Some negative adverbs:',self.negative_adverbs
   
   @classmethod
   def _computeSpecificInstanceInformativeWords(self, cf_dist, f_dist):
      '''using chi_square distribution, computes and returns the words
         that contribute the most significant info. That is words that
         are mostly unique to each set(positive, negative)'''
         
      total_num_words = f_dist.N()
      total_positive_words = cf_dist["positive"].N()
      total_negative_words = cf_dist["negative"].N()
      words_score = dict()
         
      for word in f_dist.iteritems():
         pos_score = BigramAssocMeasures.chi_sq(cf_dist["positive"][word],
                                    (f_dist[word], total_positive_words),
                                    total_num_words)

         neg_score = BigramAssocMeasure.chi_sq(cf_dist["negative"][word],
                                    (f_dist[word], total_negative_words),
                                    total_num_words)

         words_score[word] = pos_score + neg_score

      #Return 10% most useful words 
      self.informative_words = sorted(words_score.iteritems(),
                                 key=lambda (word, score): score,
                                 reverse=True)[:0.1*int(len(words_score))]


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
            sentence_features[conjunction] = True
         else:
            sentence_features[conjunction] = False

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

      pos_adj_score = pos_adj_score/total_adjectives_pos
      pos_adv_score = pos_adv_score/total_adverbs_pos

      #Compute scores for negative values
      for key in word_dict:
         if key in self.negative_adjectives:
            neg_adj_score += 1
         elif key in self.negative_adverbs:
            neg_adv_score += 1

      neg_adj_score = neg_adj_score/total_adjectives_neg
      neg_adv_score = neg_adv_score/total_adverbs_neg

      return (pos_adj_score, pos_adv_score, neg_adj_score, neg_adv_score)
      
   def train(self):
         pass
