console.log("/api/prompt route LOADED by Next.js");
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

type MatchWithMetadata = {
  id: string;
  score: number;
  metadata: {
    talk_id: string;
    title: string;
    chunk_text: string;
  };
};

function hasMetadata(m: any): m is MatchWithMetadata {
  return (
    !!m.metadata &&
    typeof m.metadata.talk_id === "string" &&
    typeof m.metadata.title === "string" &&
    typeof m.metadata.chunk_text === "string"
  );
}

// ------------------------------
//  Init OpenAI + Pinecone
// ------------------------------
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
  baseURL: process.env.OPENAI_BASE_URL!,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const index = pinecone.Index(
  process.env.PINECONE_INDEX_NAME!,
  process.env.PINECONE_HOST!
);

const TOP_K = 5; // get more chunks for classification

// ------------------------------
//  System Prompt
// ------------------------------
const SYSTEM_PROMPT = `
You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. 
If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided TED data.”
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.
`;

// ------------------------------
//  Detect Query Type
// ------------------------------
function detectQueryType(question: string): "fact" | "list" | "summary" | "recommend" {
  const q = question.toLowerCase();

  if (q.includes("recommend") || q.includes("suggest"))
    return "recommend";

  if (q.includes("summary") || q.includes("summarize") || q.includes("key idea") || q.includes("main idea"))
    return "summary";

  if (q.includes("3 ") || q.includes("three") || q.includes("list") || q.includes("multiple"))
    return "list";

  return "fact";
}

// ------------------------------
//  Extract unique talks
// ------------------------------
function uniqueTalks(matches: any[]) {
  const map = new Map();
  for (const m of matches) {
    if (!m.metadata) continue;
    const md = m.metadata;
    if (!map.has(md.talk_id)) {
      map.set(md.talk_id, {
        talk_id: md.talk_id,
        title: md.title,
        chunk: md.chunk_text,
        score: m.score ?? 0
      });
    }
  }
  return Array.from(map.values());
}

// ------------------------------
//  Build Context Text
// ------------------------------
function buildContextText(matches: any[]) {
  let txt = "";
  for (const m of matches) {
    if (!m.metadata) continue;
    const md = m.metadata;
    txt += `[Talk ${md.talk_id}] "${md.title}"\n${md.chunk_text}\n---\n`;
  }
  return txt;
}

// ------------------------------
//  POST handler
// ------------------------------
export async function POST(req: NextRequest) {
  console.log("ENV DEBUG:", {
    OPENAI: process.env.OPENAI_API_KEY,
    BASE: process.env.OPENAI_BASE_URL,
    PINE: process.env.PINECONE_API_KEY,
    HOST: process.env.PINECONE_HOST,
    INDEX: process.env.PINECONE_INDEX_NAME
    });

  try {
    const { question } = await req.json();

    if (!question)
      return NextResponse.json({ error: 'Missing "question"' }, { status: 400 });

    // Detect query type
    const mode = detectQueryType(question);

    // 1. Embed
    const emb = await openai.embeddings.create({
      model: "RPRTHPB-text-embedding-3-small",
      input: question,
    });
    const vector = emb.data[0].embedding;

    // 2. Pinecone Query
    const search = await index.query({
      vector,
      topK: TOP_K,
      includeMetadata: true,
    });

    if (!search.matches.length) {
      return NextResponse.json({
        response: "I don’t know based on the provided TED data.",
        context: [],
      });
    }

    const contextText = buildContextText(search.matches);
    const talks = uniqueTalks(search.matches);

    let userPrompt = "";
    let assistantBehavior = "";

    // ------------------------------
    //  Mode: FACT
    // ------------------------------
    if (mode === "fact") {
      assistantBehavior = `
Answer the factual question STRICTLY from the retrieved chunks.
If uncertain, say "I don't know based on the provided TED data."
Use quotes/evidence from context.`;
    }

    // ------------------------------
    //  Mode: LIST (3 TALKS)
    // ------------------------------
    if (mode === "list") {
      assistantBehavior = `
Return EXACTLY 3 TED talk titles that match the topic.
They MUST be 3 different talks.
If fewer than 3 talks appear in context, return only those available.
Do not invent talks. Use only retrieved context.`;
    }

    // ------------------------------
    //  Mode: SUMMARY
    // ------------------------------
    if (mode === "summary") {
      assistantBehavior = `
Identify the SINGLE most relevant talk.
Produce a short key‑idea summary using only the retrieved chunks.
Do not use outside knowledge.`;
    }

    // ------------------------------
    //  Mode: RECOMMENDATION
    // ------------------------------
    if (mode === "recommend") {
      assistantBehavior = `
Choose the SINGLE best TED talk relevant to the question.
Provide:
1. Title
2. Why it matches — using evidence from the context only.
No external knowledge allowed.`;
    }

    userPrompt = `
User question:
${question}

Task mode: ${mode}

Follow these instructions:
${assistantBehavior}

Use ONLY the following context:
${contextText}
`;

    // 3. Completion
    const completion = await openai.chat.completions.create({
      model: "RPRTHPB-gpt-5-mini",
      temperature: 1,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
    });

    const answer = completion.choices[0].message?.content?.trim() ?? "";

    return NextResponse.json({
        response: answer,
        context: search.matches
        .filter(hasMetadata)
        .map(m => ({
            talk_id: m.metadata.talk_id,
            title: m.metadata.title,
            chunk: m.metadata.chunk_text,
            score: m.score
        })),
        Augmented_prompt: {
            System: SYSTEM_PROMPT,
            User: userPrompt,
        },
        debug: {
            mode,
            top_k_used: TOP_K,
            raw_matches: search.matches.length
        }
    });

  } catch (err: any) {
    console.error("ERROR /api/prompt:", err);
    return NextResponse.json(
      { error: "Internal server error", details: err.message },
      { status: 500 }
    );
  }
}