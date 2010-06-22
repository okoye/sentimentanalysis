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

class OpinionMiner:

   def __init__(self):
      self.classifier = None


      self._setInformativeFeatures()

   @classmethod
   def _setInformativeFeatures(self):
      self._setSelectedPOSTags()
      #_setCommonWords()

   @classmethod
   def _setSelectedPOSTags(self):
      #first get all (word, tag) in corpuses
      sentences = brown.tagged_sents(simplify_tags=True)
      self.selected_tags = ["ADJ","ADV", "CNJ"]
      self.selective_pos = ConditionalFreqDist()

      for sentence in sentences:
         for (word, tag) in sentence:
            if tag in self.selected_tags:
               self.selective_pos[tag].inc(str(word).lower())

         

   def train(self):
         pass
