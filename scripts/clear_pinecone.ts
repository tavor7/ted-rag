import readline from "readline";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

async function confirm(question: string): Promise<boolean> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise(resolve =>
    rl.question(question, answer => {
      rl.close();
      resolve(answer.toLowerCase() === "yes");
    })
  );
}

async function clearIndex() {
  console.log("⚠️  WARNING: This will delete ALL vectors in the index.");
  console.log(`Index: ${process.env.PINECONE_INDEX_NAME}`);
  console.log("");

  const ok = await confirm("Type 'yes' to confirm deletion: ");
  if (!ok) {
    console.log("❌ Aborted.");
    process.exit(0);
  }

  try {
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pc.Index(
      process.env.PINECONE_INDEX_NAME!,
      process.env.PINECONE_HOST!
    );

    console.log("🧹 Clearing index...");
    await index.deleteAll();

    console.log("✅ Index cleared successfully!");
  } catch (err: any) {
    console.error("\n❌ Clearing failed!");
    console.error(err.message || err);
  }
}

clearIndex();