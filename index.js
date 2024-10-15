import { Ollama } from "@langchain/community/llms/ollama";
import { ElasticVectorSearch } from "@langchain/community/vectorstores/elasticsearch";
import { Client } from "@elastic/elasticsearch";
import "dotenv/config";
import { OllamaEmbeddings } from "@langchain/ollama";
import {
  CharacterTextSplitter,
  RecursiveCharacterTextSplitter,
} from "@langchain/textsplitters";
import { StringOutputParser } from "@langchain/core/output_parsers";

import * as fs from "node:fs";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434", // Default value
  model: "llama3.1:latest", // Default value
});

const config = {
  node: process.env.ELASTIC_URL ?? "https://127.0.0.1:9200",
  tls: {
    rejectUnauthorized: false,
  },
};

if (process.env.ELASTIC_API_KEY) {
  config.auth = {
    apiKey: process.env.ELASTIC_API_KEY,
  };
} else if (process.env.ELASTIC_USERNAME && process.env.ELASTIC_PASSWORD) {
  config.auth = {
    username: process.env.ELASTIC_USERNAME,
    password: process.env.ELASTIC_PASSWORD,
  };
}

const formatDocumentsAsString = (documents) => {
  return documents.map((document) => document.pageContent).join("\n\n");
};

const ollama = new Ollama({
  baseUrl: "http://127.0.0.1:11434",
  model: "llama3.1:latest",
});
const clientArgs = {
  client: new Client(config),
  indexName: process.env.ELASTIC_INDEX ?? "test_vectorstore",
};

const vectorStore = new ElasticVectorSearch(embeddings, clientArgs);

// const text = fs.readFileSync("file2.txt", "utf8");
// const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
// const docs = await textSplitter.createDocuments([text]);

// const documents = docs.map((d, index) => ({ ...d, id: index + 1 }));
// console.log(documents.map((i, index) => (index + 1).toString()));

// await vectorStore
//   .addDocuments(documents, {
//     ids: documents.map((i, index) => (index + 1).toString()),
//   })
//   .then((data) => {
//     console.log("embedings done ", data);
//   });

const template = `
you are an assistant for question-answering tasks.
use the following pieces of retrieved context to answer the question.
if you don't know the answer, just say toz fik a khaali ,don't try to make up an answer.
use two sentences minimum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
`;

const vectorStoreRetriever = vectorStore.asRetriever();

const prompt = ChatPromptTemplate.fromTemplate(template);

const chain = RunnableSequence.from([
  {
    context: vectorStoreRetriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  ollama,
  new StringOutputParser(),
]);

const answer = await chain.invoke("what is for pizza today");

console.log({ answer });
