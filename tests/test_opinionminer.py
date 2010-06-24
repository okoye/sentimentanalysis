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

class TestOpinionMiner():
   
      def test_tag_generation(self):
         '''tests whether tags are generated'''
         x = opinionminer.OpinionMiner()

         print 'Tags are generated: ',len(x.selective_pos.conditions())>0
         print 'Positive Adverbs exist:', (len(x.positive_adverbs) > 2)
         print 'Positive Adjectives exist:', (len(x.positive_adjectives)>2)


if __name__ == '__main__':
   tester = TestOpinionMiner()
   tester.test_tag_generation()


