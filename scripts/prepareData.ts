import fs from "fs";
import csv from "csv-parser";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import dotenv from "dotenv";

// dotenv.config();
dotenv.config({ path: ".env.local" });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
  baseURL: process.env.OPENAI_BASE_URL!,   // IMPORTANT for llmod.ai
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!, process.env.PINECONE_HOST!);

const CHUNK_SIZE = 1024;
const OVERLAP_RATIO = 0.2;

function chunkText(text: string): string[] {
  const words = text.split(/\s+/);
  const overlap = Math.floor(CHUNK_SIZE * OVERLAP_RATIO);
  const step = CHUNK_SIZE - overlap;

  const chunks: string[] = [];
  for (let start = 0; start < words.length; start += step) {
    chunks.push(words.slice(start, start + CHUNK_SIZE).join(" "));
  }
  return chunks;
}

async function processCSV() {
  const results: any[] = [];
  let totalTalks = 0;

  const csvPath = process.env.TED_CSV_PATH!;
  if (!fs.existsSync(csvPath)) {
    console.error("ERROR: TED_CSV_PATH does not exist:", csvPath);
    process.exit(1);
  }

  console.log("Reading CSV from:", csvPath);

  return new Promise<void>((resolve) => {
    fs.createReadStream(csvPath)
      .pipe(csv())
      .on("data", (data) => results.push(data))
      .on("end", async () => {
        console.log("Loaded", results.length, "rows.");

        totalTalks = results.length;

        // Precompute total chunks across all talks
        let globalChunkCount = 0;
        for (const r of results) {
          const contentTmp = `
Title: ${r.title}
Speaker: ${r.speaker_1}
Description: ${r.description}
Transcript: ${r.transcript}
  `;
          globalChunkCount += chunkText(contentTmp).length;
        }
        console.log("Total chunks across all talks:", globalChunkCount);

        let processedGlobalChunks = 0;

        for (let i = 0; i < results.length; i++) {
        //for (let i = 0; i < Math.min(40, results.length); i++) {
          const row = results[i];

          const content = `
Title: ${row.title}
Speaker: ${row.speaker_1}
Description: ${row.description}
Transcript: ${row.transcript}
          `;

          const chunks = chunkText(content);

          for (let j = 0; j < chunks.length; j++) {
            const chunkText = chunks[j];

            // === Embedding ===
            const embeddingRes = await openai.embeddings.create({
              model: "RPRTHPB-text-embedding-3-small",
              input: chunkText,
            });

            const embedding = embeddingRes.data[0].embedding;

            // === Pinecone upsert ===
            await index.upsert([
              {
                id: `${row.talk_id}-${j}`,
                values: embedding,
                metadata: {
                  talk_id: row.talk_id,
                  title: row.title,
                  speaker: row.speaker_1,
                  chunk_text: chunkText,
                  chunk_index: j,
                },
              },
            ]);

            processedGlobalChunks++;
          }
          if (i % 20 === 0) {
            console.log(
              `Inserted up to global chunk ${processedGlobalChunks}/${globalChunkCount} — finished talk ${i + 1}/${totalTalks}`
            );
          }
        }

        console.log("DONE embedding and uploading to Pinecone!");
        resolve();
      });
  });
}

processCSV().catch((err) => {
  console.error("ERROR during processing:", err);
});