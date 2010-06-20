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

import tweepy
from logger import crawl_logs
from models import TwitterMessage
from couchdbkit import Server

#Instantiate elements server and database
server = Server()
db = server.get_or_create_db('crawls')

#Associate all models with Couchdb 
TwitterMessage.set_db(db)

class CrawlingService:
   def crawl(self):
      raise NotImplementedError("crawl method must be implemented in subclass")


class GoogleNewsCrawlingService(CrawlingService):
   pass


class BingNewsCrawlingService(CrawlingService):
   pass


class BloggerCrawlingService(CrawlingService):
   pass

class TwitterCrawlingService(CrawlingService):

   class StreamingLib(tweepy.StreamListener):
      def __init__(self):
         tweepy.StreamListener.__init__(self)
         crawl_logs(['instantiated new Stream Listener'])

      def on_status(self, status):
         message = TwitterMessage(date_time_generated=status.created_at,
                                    text_message = status.text,
                                    message_id = str(status.id),
                                    user_name = status.author.screen_name)
         message.save()

      def on_error(self, status_code):
         crawl_logs(['an error with status code %s occured' % (status_code)])
         return True

      def on_timeout(self):
         crawl_logs(['connection timed out.'])
   
   def crawl(self):
      stream = tweepy.Stream('cscyberspace1','zxcrty09()', self.StreamingLib(),
                  timeout=60.0)
      track_list = ['aapl', 'apple', 'goog', 'google', 'msft', 'microsoft',
                     'obama', 'beiber', 'justin', 'soccer', 'south africa']
      
      try:
         stream.filter(None, track=track_list)
      except:
         crawl_logs(['a fatal exception occured', 'ending stream...'])


class BackTypeCrawlingService(CrawlingService):
   pass

################# Main Instance of Class #####################
#Should be multi threaded for each crawling service offered.

if __name__ == '__main__':
   #Invoke a thread for each crawling service
   twitter = TwitterCrawlingService()
   twitter.crawl()
