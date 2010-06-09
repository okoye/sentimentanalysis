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
from google.appengine.ext.db import polymodel

class TwitterMessage(db.Model):
   date_time_generated = db.DateTimeProperty()
   identified_entities = db.StringListProperty()
   entity_extract_flag = db.BooleanProperty()
   sentiment_scores = db.BlobProperty()
   sentiment_score_flag = db.BooleanProperty()
   text_data = db.StringProperty()
   id = db.StringProperty()
   user_name = db.StringProperty()
