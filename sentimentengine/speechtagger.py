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

from nltk.tag import *
from nltk.classify import naivebayes, maxent, accuracy
from nltk import TaggerI
from nltk.corpus import brown, conll2000, treebank
from cPickle import dump, load
from os import path
from logger import crawl_logs
import random

class SpeechTagger(TaggerI):

   def __init__(self):
      '''Initialize variables containing classifier training data'''
      #tagged_sents = brown.tagged_sents(categories='news') 
      #tagged_sents = conll2000.tagged_sents()
      tagged_sents = treebank.tagged_sents()
      self.train = tagged_sents[: int(0.8 * len(tagged_sents))]
      self.test = tagged_sents[int(0.8 * len(tagged_sents)): ]

   def tag(self, untagged_sentence):
      '''use pickled classifier or retrain, store then use classifier'''

      if path.isfile(path.join(path.dirname('__file__'), 'pos.pk')):
         input = open('pos.pk', 'rb')
         self.classifier = load(input)
         input.close()
         crawl_logs(["loaded classifier"])
      
      else:
         train_set = []
         for sentence in self.train:
            history = []
            for i, (word, tag) in enumerate(sentence):
               train_set.append((self._pos_features(untag(sentence),
                                              i,
                                              history),tag))
               history.append(tag)
         crawl_logs(["re-training classifier"])
         self.bayes = naivebayes.NaiveBayesClassifier.train(train_set)
         self.classifier = self.bayes

         crawl_logs(['classifier accuracy:',accuracy(self.classifier,
                                                      self.test)])
         crawl_logs('pickling classifier')
         
         output = open('pos.pk', 'wb')
         dump(self.bigram, output, -1)
         output.close()

      #TODO: Split sentence into word tokens
      history = []
      for i, word in enumerate(untagged_sentence):
         tag = self.classifier.classify(_pos_features(sentence, i, history))
         history.append(tag)
      
      return zip(sentence, history)

   @classmethod
   def _pos_features(self, sentence, i, history):
      features = {}
      if i == 0:
         features["p_tag"] = "<start>"
         features["p_suffix"] = "<start>"
      else:
         features["p_tag"] = history[i-1]
         features["p_suffix"] = history[i-1][-2:]

      if i == len(sentence) - 1:
         features["n_suffix"] = "<end>"
      else:
         features["n_suffix"] = sentence[i+1][-2:]

      features["c_suffix1"] = sentence[i][-1:]
      features["c_suffix2"] = sentence[i][-2:]
      features["c_suffix3"] = sentence[i][-3:]
      features["current_length"] = len(sentence[i])

      return features



if __name__ == "__main__":
   x = SpeechTagger()
   x.tag(['chuka', 'is', 'not', 'effective', 'but'])
