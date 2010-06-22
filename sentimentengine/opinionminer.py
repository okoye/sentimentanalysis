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
from nltk.probability import ConditionalFreqDist
from nltk.probability import FreqDist

class OpinionMiner:

   def __init__(self):
      self.classifier = None
      self._setInformativeFeatures()

   @classmethod
   def _setInformativeFeatures(self):
      self._setSelectedPOSTags()
      self._setPositiveWords()

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
            if fredist[key] > 10:
               self.selective_pos[category].inc(key)

      for category in self.selective_pos.conditions():
         dist = self.selective_pos[category]
         print category, dist.keys()

   @classmethod
   def _setPositiveWords():
      sentences = movie_reviews.sents()
      

   def train(self):
         pass
