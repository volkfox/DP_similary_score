# similary score webservice wrapper
Similarity score services based on Google Universal Sentence Encoder 

*Files*:  

**Stanford_DD.ipynb**  -  sandbox for playing with scores in Collab. Click on the file, then on the image ![Tux, the Linux mascot](https://colab.research.google.com/assets/colab-badge.svg)  

*Folders*:  

**NodeJS** - Javascript HTTP server responding with similarity matrix based on the POST request with data (array of sentences or paragraphs)  

**PythonHTTP** - Python HTTP server responding with similarity matrix based on the POST request with data (array of sentences or paragraphs)

Testing servers with POST request:
$ curl -H "Content-Type: application/json" -X POST -d '{"sentences":["The quick brown fox", "Fox is quick"]}' http://127.0.0.1:8000/api
