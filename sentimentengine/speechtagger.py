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

from nltk.tag import TaggerI, hmm, untag
from nltk.classify import naivebayes, maxent
from nltk import TaggerI
from nltk.corpus import brown, conll2000, treebank

class SpeechTagger(TaggerI):

   def __init__(self):
      '''train brill classifier'''
      #tagged_sents = brown.tagged_sents(categories='news') 
      #tagged_sents = conll2000.tagged_sents()
           


def _pos_features(sentence, i, history):
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

