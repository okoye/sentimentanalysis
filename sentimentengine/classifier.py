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

#   Refer to NLTK Manual (Chapter 06 & 07)

from nltk.tag import *
from nltk.classify import naivebayes, maxent, decisiontree
from nltk import TaggerI
from nltk.corpus import stopwords

class SpeechTagger(TaggerI):
   
   def __init__(self, training_sents):
      train_set = []
      for sentence in training_sents:
         untagged_sent = untag(sentence)
         history = []
         for indices, (word, tag) in enumerate(sentence):
               train_set.append((feature_extract(untagged_sent,indices,
                               history),tag))
               history.append(tag)

      self.bayes_classifier = naivebayes.NaiveBayesClassifier.train(train_set)  
      #self.decisiont_classifier = decisiontree.DecisionTreeClassifier.train(
                                    #train_set, depth_cutoff=400)
   
   def tag(self, sentence):
      history = []
      for i, word in enumerate(sentence):
         tag1=self.bayes_classifier.classify(feature_extract(sentence,i,history))
       #  tag2=self.decisiont_classifier.classify(feature_extract(sentence,i,
                                             #history))
         history.append(tag1)
      return zip(sentence, history)


def feature_extract(sentence,word_index, history):
   features = {}
   if word_index == 0:
      features["previous_tag"] = "<start>"
      features["previous_suffix2"] = "<start>"
   else:
      features["previous_tag"] = history[word_index-1]
      features["previous_suffix2"] = sentence[word_index-1][-2:]

   if word_index == len(sentence)-1:
      features["next_suffix2"] = "<end>"
   else:
      features["next_suffix2"] = sentence[word_index+1][-2:]

   features["current_suffix2"] = sentence[word_index][-2:]
   features["current_suffix1"] = sentence[word_index][-1:]
   features["current_suffix3"] = sentence[word_index][-3:]
   features["current_length"] = len(sentence[word_index])

   return features


def testClassifier():
   from nltk.corpus import brown

   tagged_sents = brown.tagged_sents(categories='news')
   train = tagged_sents[:int(0.8*len(tagged_sents))]
   test = tagged_sents[int(0.8*len(tagged_sents)):]

   posTagger = SpeechTagger(train)
   print "Classifier training completed"
   print posTagger.evaluate(test) 


if __name__ == "__main__":
   testClassifier()
