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

from nltk.corpus import brown, conll2000, treebank
from nltk.classify import naivebayes
from nltk.tag import untag
from nltk.tag.brill import *
from nltk.tokenize import word_tokenize
from cPickle import dump, load
from os.path import exists

templates = [
               SymmetricProximateTokensTemplate(ProximateTagsRule, (1,1)),
               SymmetricProximateTokensTemplate(ProximateTagsRule, (2,2)),
               SymmetricProximateTokensTemplate(ProximateTagsRule, (1,2)),
               SymmetricProximateTokensTemplate(ProximateTagsRule, (1,3)),
               SymmetricProximateTokensTemplate(ProximateWordsRule, (1,1)),
               SymmetricProximateTokensTemplate(ProximateWordsRule, (2,2)),
               SymmetricProximateTokensTemplate(ProximateWordsRule, (1,2)),
               SymmetricProximateTokensTemplate(ProximateWordsRule, (1,3)),
               ProximateTokensTemplate(ProximateTagsRule, (-1,-1), (1,1)),
               ProximateTokensTemplate(ProximateWordsRule, (-1,-1), (1,1)),
            ]
file = "tagger.pickle"
class SpeechTagger:
   
   def __init__(self):
      '''initialize and train brill and naive bayes classifiers'''
     
      #TODO: Fix bug where it loads tagger from calling module dir
      if exists(file):
         input = open(file, 'rb')
         self.classifier = load(input)
         input.close()
         print 'Successfully loaded saved classifier'
         return

      self.bayes = NaiveBayesTagger()
      boundary = int(len(brown.tagged_sents())*0.8)
      train = brown.tagged_sents(simplify_tags=True)[:boundary]

      brill_trainer = FastBrillTaggerTrainer(initial_tagger = self.bayes,
                                             templates = templates,
                                             trace = 3,
                                             deterministic = True)
         
      self.classifier = brill_trainer.train(train, max_rules=10)
         
      print 'Saving Taggers to file: "pos_tagger.pickle"'
      output = open(file, 'wb')
      dump(self.classifier, output, 1)
      output.close()

   def evaluate(self):
      '''run tests on conll2000 and treebank data'''

      test = treebank.tagged_sents()[:100]
      treebank_result = (100*self.classifier.evaluate(test))

      test = conll2000.tagged_sents()[:100]
      conll2000_result = (100*self.classifier.evaluate(test))

      test = brown.tagged_sents()[int(len(brown.tagged_sents())*0.8):]
      brown_result = (100*self.classifier.evaluate(test))

      return (treebank_result, conll2000_result, brown_result)

   def retrain(self, train_data):
      '''Attempts to retrain the brill tagger using the specified data''' 
      #TODO: Retrain naive bayes classifier also
      brill_trainer = FastBrillTaggerTrainer(initial_tagger = self.bayes,
                                             templates = templates,
                                             trace = 3,
                                             deterministic = True)

      try:
         self.classifier = brill_trainer.train(train_data, max_rules=10)
         output = open(file, 'wb')
         dump(self.classifier, output, 1)
         output.close()
      except:
         print "Failed to retrain brill tagger"
         return -1

      return 0

   def tag(self, untagged_sentences):
      '''Tags all sentences provided in the dict untagged_sentences'''
      #first tokenize all the words in each sentence provided
      tokenized_sents = []
      for sentence in untagged_sentences:
         tokenized_sents.append(word_tokenize(sentence))
      return self.classifier.batch_tag(tokenized_sents)
      

class NaiveBayesTagger(TaggerI):

   def __init__(self):

      boundary = int(len(brown.tagged_sents())*0.8)
      train_naive = brown.tagged_sents(simplify_tags=True)[:boundary] 
      temp_train_data = []
      for sentence in train_naive:
         untagged_sent = untag(sentence)
         history = []
         for i, (word, tag) in enumerate(sentence):
            temp_train_data.append((self.featextract(untagged_sent,
                                                      i,
                                                      history),
                                                      tag))
            history.append(tag)
      self.bayes=naivebayes.NaiveBayesClassifier.train(temp_train_data)

   def tag(self, sentence):
      tagged_sentence = []
      history = []
      for i, word in enumerate(sentence):
         tag=self.bayes.classify(self.featextract(sentence,
                                                   i,
                                                   history))
         tagged_sentence.append((word,tag))
         history.append(tag)
      return tagged_sentence


   @classmethod
   def featextract(self, sentence, i, history, mode="bayes"):
      '''extract features from data. relevant modes includes bayes'''

      if mode == "bayes":
         features = {}
         if i == 0:
            features['p_tag'] = "<start>"
            features['p_suffix'] = "<start>"

         else:
            features['p_tag'] = history[i-1]
            features['p_suffix'] = sentence[i-1][-2:]

         if i == (len(sentence) - 1):
            features['n_suffix'] = "<end>"
         
         else:
            features['n_suffix'] = sentence[i+1][-2:]

         features['suffix1'] = sentence[i][-1:]
         features['suffix2'] = sentence[i][-2:]
         features['suffix3'] = sentence[i][-3:]
         features['length'] = len(sentence[i])

         return features
      
      elif mode == "brill":
         pass

