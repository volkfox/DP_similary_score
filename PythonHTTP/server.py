#!/usr/bin/env python3
"""
Simple HTTP server in python for Google Sentence Similarity requests
Usage::
    python ./server.py [<port>]
Testing::
   curl -H "Content-Type: application/json" -X POST -d '{"sentences":["The quick brown fox", "Fox is quick"]}' http://127.0.0.1:8000/api
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import dumps
import logging
import json

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class S(BaseHTTPRequestHandler):

    def _send_cors_headers(self):
      """ Sets headers required for CORS """
      self.send_header("Access-Control-Allow-Origin", "*")
      self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
      self.send_header("Access-Control-Allow-Headers", "x-api-key,Content-Type")

    def do_OPTIONS(self):
       self.send_response(200)
       self._send_cors_headers()
       self.end_headers()

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

        help = "do POST request to route /api with JSON object {'sentences':[]} holding an array of texts"
        self._set_response()
        self.wfile.write(help.encode('utf-8'))


    def do_POST(self):

        error = 'JSON object misses "sentences" key array'

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        objectDict = json.loads(post_data.decode('utf-8'))
       
        message = {}
        print(f"received {objectDict}") 
        if ("sentences" in objectDict) and isinstance(objectDict["sentences"], list):
           embeddings = embed(objectDict["sentences"])
           corr = np.inner(embeddings, embeddings)
           similarity = corr[0,1:]
           #print(f"Similarity score: {similarity}")
           message =  json.dumps(similarity.tolist())
        else: 
           message = error        
        
        self.wfile.write(message.encode('utf-8'))

        #logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n", str(self.path), str(self.headers), post_data.decode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8000):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
