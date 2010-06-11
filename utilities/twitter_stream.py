#!/usr/bin/python 

import csv
import tweepy
import os

class StreamingLogger(tweepy.StreamListener):
   def __init__(self):
      tweepy.StreamListener.__init__(self)
      self.counter = 0
      self.file_path = os.path.join(os.path.dirname(__file__), 'dumps.csv')
      self.csv_writer = csv.writer(open(self.file_path,'w'), delimiter=',')
                        
   def on_status(self, status):
      try:
         self.csv_writer.writerow([status.created_at, status.text, status.id,
                                    status.author.screen_name])
         self.counter += 1
      except:
         pass

   def on_error(self, status_code):
      print 'An error with status code %s occured' % status_code
      return True

   def on_timeout(self):
      print "Connection timed out!"

def main():
   stream = tweepy.Stream('cscyberspace1','zxcrty09()', StreamingLogger(),
               timeout=6.0)

   track_list = ['Apple', 'Google', 'Microsoft', 'Obama', 'Beiber', 'Justin']
   print "starting stream..."
   stream.sample()#filter(None,track=track_list)
   print "ending stream..."

if __name__ == '__main__':
   try:
      main()
   except:
      print "Closing program..."
