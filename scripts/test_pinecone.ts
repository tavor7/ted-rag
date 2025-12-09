import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

async function test() {
  console.log("🔗 Connecting to Pinecone...");

  try {
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pc.Index(
      process.env.PINECONE_INDEX_NAME!,
      process.env.PINECONE_HOST!
    );

    console.log("📡 Fetching index statistics...");
    const stats = await index.describeIndexStats();

    console.log("\n✅ Pinecone connection successful!");
    console.log(JSON.stringify(stats, null, 2));
  } catch (err: any) {
    console.error("\n❌ Pinecone test failed!");
    console.error(err.message || err);
    process.exit(1);
  }
}

test();