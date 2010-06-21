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

import sys
sys.path.append("../sentimentengine")
import unittest
import speechtagger
from nltk.corpus import brown

class TestSpeechTagger(unittest.TestCase):

   def setUp(self):
      '''performs initial training on taggers and classifiers'''
      self.atagger = speechtagger.SpeechTagger()
      self.assert_(self.atagger != None)

   def test_evaluate(self):
      '''ensures taggers pass at least 80% accuracy for the 3 corpuses'''
      (treebank, conll, brown) = self.atagger.evaluate()
      self.assert_(treebank >= 80)
      self.assert_(conll2000 >= 80)
      self.assert_(brown >= 80)

   def test_retrain(self):
      '''retrains only tagger with specified training data'''
      train = brown.tagged_sents(categories="news")
      self.assert_(self.atagger.retrain(train) != -1)


if __name__ == '__main__':
   unittest.main()



