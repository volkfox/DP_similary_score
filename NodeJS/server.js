/*
* node.js server for calculating the Google Sentence Encoder Similarity Score matrix for array of sentences
* 
* Run: 
* npm install; node server.js
*
* Test on UNIX/Macos:
* curl -H "Content-Type: application/json" -X POST -d '{"sentences": ["Quick brown fox jumps over the lazy dog","Fox is a quick animal"]}' http://127.0.0.1:3000/api
*/

require('@tensorflow/tfjs-node');
const use = require('@tensorflow-models/universal-sentence-encoder');

const bodyParser = require('body-parser');
const express = require('express');
const fetch = require('node-fetch');

const app = express();
const jsonParser = bodyParser.json();

app.use(express.static('public'));

async function main() {
    const port = process.env.PORT || 3000;
    await app.listen(port);
    console.log(`Server listening on port ${port}!`);
}

main();

/* returns all existing queries to the client via GET /api  */
async function onPost(req, res) {

    const messageBody = req.body;
    //const messageKeys = Object.keys(messageBody);
    //console.log("POST Keys: " + messageKeys);

    const sentences = messageBody["sentences"]
    const embeddings = await embed(sentences);
    const array = await embeddings.arraySync();
    const cosine_score = cosine_similarity_matrix(array);
    res.json(cosine_score);
}

app.post('/api', jsonParser, onPost);


/* returns test message to the client via GET /api  */
async function onGet(req, res) {

    const help = "do POST request to route /api with JSON object {'sentences':[]} holding an array of texts";
    res.json({
       help 
    });

}

app.get('/api', onGet);

function dot(a, b){
  var hasOwnProperty = Object.prototype.hasOwnProperty;
  var sum = 0;
  for (var key in a) {
    if (hasOwnProperty.call(a, key) && hasOwnProperty.call(b, key)) {
      sum += a[key] * b[key]
    }
  }
  return sum
}

function similarity(a, b) {
  var magnitudeA = Math.sqrt(dot(a, a));
  var magnitudeB = Math.sqrt(dot(b, b));
  if (magnitudeA && magnitudeB)
    return dot(a, b) / (magnitudeA * magnitudeB);
  else return false
}

function cosine_similarity_matrix(matrix){

  let cosine_similarity_matrix = [];
  for(let i=0;i<matrix.length;i++){
    let row = [];
    for(let j=0;j<i;j++){
      row.push(cosine_similarity_matrix[j][i]);
    }
    row.push(1);
    for(let j=(i+1);j<matrix.length;j++){
      row.push(similarity(matrix[i],matrix[j]));
    }
    cosine_similarity_matrix.push(row);
  }
  return cosine_similarity_matrix;
}

async function embed(sentences) {

 const model = await use.load();
 const embeddings = await model.embed(sentences); 
 return(embeddings);
}


