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

from google.appengine.ext import db
from google.appengine.tools import bulkloader
import models
import datetime

class TwitterMessageLoader(bulkloader.Loader):
   def __init__(self):
      bulkloader.Loader.__init__(self,
                                 'TwitterMessage',
                                 [('date_time_generated',
                                 lambda x: datetime.datetime.strptime(x,
                                 '%a %b %d %H:%M:%S')),
                                 ('text_data',str ),
                                 ('id', str),
                                 ('user_name', str)
                                 ])

loader = [TwitterMessageLoader]
