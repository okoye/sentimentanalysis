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
from nltk.corpus import brown, conll2000, treebank
from nltk.probability import ConditionalFreqDist

selective_pos = ConditionalFreqDist()
selected_tags = {'ADJ':True, 'ADV': True, 'CNJ':True}

class OpinionMiner:

   def __init__(self):
      self.classifier = None
      self._setInformativeFeatures()

   @classmethod
   def _setInformativeFeatures(self):
      _setSelectedPOSTags()
      _setCommonWords()

   def _setSelectedPOSTags(self):
      #first get all (word, tag) in corpuses
      sentences = brown.tagged_sents(simplify_tags=True)
                  + conll2000.tagged_sents(simplify_tags=True)
                  + treebank.tagged_sents(simplify_tags=True)

      for sentence in sentences:
         condFreq = ConditionalFreqDist((tag, word)
                     for (word, tag) in enumerate(sentence)
                     if (selected_tags.get(tag) != None)
      
      selective_pos = condFreq
      print 'No of sample outcomes ', selective_pos.N()
      print 'Conditions: ', selective_pos.conditions()
      print selective_pos[1]
         

   def train(self):
      adjectives = _computeFreqAdjectives()

   def _computeFreqAdjectives(self):
      '''returns the most frequent adjectives used from the corpus'''
      for sentence in movie_reviews.sents():
         
