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
import opinionminer
import random
from nltk.corpus import movie_reviews

class TestOpinionMiner():
   
      def test_calibration(self):
         '''tests whether all features like default adjectives 
            are generated when class is instantiated'''

         train = []
         test = []
 
         bound = int(len(movie_reviews.sents(categories="pos"))*0.8)
         for i,sent in enumerate(movie_reviews.sents(categories="pos")[:bound]):
            train.append(("positive"," ".join(sent)))

         for sent in movie_reviews.sents(categories="pos")[bound:]:
            test.append(("positive"," ".join(sent)))

         bound = int(len(movie_reviews.sents(categories="neg"))*0.8)
         for sent in movie_reviews.sents(categories="neg")[:bound]:
            train.append(("negative", " ".join(sent)))

         for sent in movie_reviews.sents(categories="neg")[bound:]:
            test.append(("negative"," ".join(sent)))
            

         random.shuffle(train)
         random.shuffle(test)

         x = opinionminer.OpinionMiner()

         x.trainClassifier(train)
         print x.classify("chuka is a good boy") 


         print 'Tags are generated: ',len(x.selective_pos.conditions())>0
         print 'Positive Adverbs exist:', (len(x.positive_adverbs) > 2)
         print 'Positive Adjectives exist:', (len(x.positive_adjectives)>2)
         print 'Number of Negative adj', len(x.negative_adjectives)
         print 'Number of Positive adj', len(x.positive_adjectives)
         print 'Common sets of adj', len(x.positive_adjectives - 
                                          x.negative_adjectives)


if __name__ == '__main__':
   tester = TestOpinionMiner()
   tester.test_calibration()


