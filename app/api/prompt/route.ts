import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
  baseURL: process.env.OPENAI_BASE_URL!,   // Required for llmod.ai
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const index = pinecone.Index(
  process.env.PINECONE_INDEX_NAME!,
  process.env.PINECONE_HOST!
);

const TOP_K = 5;

const SYSTEM_PROMPT = `
You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use external knowledge, the Internet, or prior learned assumptions.
If the answer cannot be determined from the provided chunks, reply exactly:
"I don’t know based on the provided TED data."

Always explain your reasoning using ONLY the provided chunks.
`;

export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();
    if (!question) {
      return NextResponse.json(
        { error: 'Missing "question" field.' },
        { status: 400 }
      );
    }

    // 1. Embed question
    const emb = await openai.embeddings.create({
      model: "RPRTHPB-text-embedding-3-small",
      input: question,
    });
    const vector = emb.data[0].embedding;

    // 2. Query Pinecone
    const search = await index.query({
      vector,
      topK: TOP_K,
      includeMetadata: true,
    });

    // 3. Build context response + text for prompt
    let contextText = "";
    const contextList =
      search.matches?.map((match) => {
        const md: any = match.metadata;

        contextText += `
[Talk ${md.talk_id}] Title: ${md.title}
${md.chunk_text}
---
`;

        return {
          talk_id: md.talk_id,
          title: md.title,
          chunk: md.chunk_text,
          score: match.score ?? 0,
        };
      }) ?? [];

    // If no context found, follow assignment requirements
    if (contextList.length === 0) {
      return NextResponse.json({
        response: "I don’t know based on the provided TED data.",
        context: [],
        Augmented_prompt: {
          System: SYSTEM_PROMPT,
          User: `User question: ${question}\n\nNo context retrieved.`,
        },
      });
    }

    // 4. Build user prompt
    const userPrompt = `
User question:
${question}

Use ONLY the following TED context chunks:
${contextText}
`;

    // 5. Chat completion
    const completion = await openai.chat.completions.create({
      model: "RPRTHPB-gpt-5-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      temperature: 1,
    });

    const answer = completion.choices[0]?.message?.content?.trim() ?? "";

    return NextResponse.json({
      response: answer,
      context: contextList,
      Augmented_prompt: {
        System: SYSTEM_PROMPT,
        User: userPrompt,
      },
    });
  } catch (err: any) {
    console.error("ERROR /api/prompt:", err);
    return NextResponse.json(
      { error: "Internal server error", details: err.message },
      { status: 500 }
    );
  }
}