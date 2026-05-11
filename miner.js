require("dotenv").config();

const { ethers } = require("ethers");
const { spawn }  = require("child_process");
const readline   = require("readline");
const path       = require("path");
const fs         = require("fs");

const RPC_URL         = process.env.RPC_URL;
const PRIVATE_KEY     = process.env.PRIVATE_KEY;
const CONTRACT_ADDRESS = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";
const GPU_BIN         = path.join(__dirname, "miner_gpu");

const ABI = [
  "function getChallenge(address miner) view returns (bytes32)",
  "function miningState() view returns (uint256 era,uint256 reward,uint256 difficulty,uint256 minted,uint256 remaining,uint256 epoch,uint256 epochBlocksLeft_)",
  "function mine(uint256 nonce)"
];

function requireEnv() {
  if (!RPC_URL || !PRIVATE_KEY) {
    console.error("Isi RPC_URL dan PRIVATE_KEY di file .env dulu.");
    console.error("Contoh: cp .env.example .env lalu edit file tersebut.");
    process.exit(1);
  }
  if (!PRIVATE_KEY.startsWith("0x")) {
    console.error("PRIVATE_KEY harus diawali 0x.");
    process.exit(1);
  }
}

function toHex32(n) {
  return "0x" + n.toString(16).padStart(64, "0");
}

/* Spawn GPU miner and resolve with the winning nonce (BigInt) */
function mineOnGPU(challenge, difficulty) {
  const diffHex = toHex32(BigInt(difficulty.toString()));

  return new Promise((resolve, reject) => {
    const proc = spawn(GPU_BIN, [challenge, diffHex], { stdio: ["ignore", "pipe", "pipe"] });

    proc.stderr.on("data", d => process.stderr.write("[GPU] " + d));

    const rl = readline.createInterface({ input: proc.stdout, crlfDelay: Infinity });
    rl.on("line", line => {
      if (line.startsWith("FOUND:")) {
        const nonce = BigInt(line.slice(6).trim());
        proc.kill("SIGTERM");
        resolve(nonce);
      }
    });

    proc.on("error", err => reject(new Error("GPU binary error: " + err.message)));
    proc.on("close", code => {
      if (code !== 0 && code !== null)
        reject(new Error(`GPU miner exited with code ${code}`));
    });
  });
}

async function main() {
  requireEnv();

  if (!fs.existsSync(GPU_BIN)) {
    console.error(`\nGPU binary tidak ditemukan: ${GPU_BIN}`);
    console.error("Build dulu dengan: bash build.sh\n");
    process.exit(1);
  }

  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const wallet   = new ethers.Wallet(PRIVATE_KEY, provider);
  const contract = new ethers.Contract(CONTRACT_ADDRESS, ABI, wallet);

  console.log("Wallet  :", wallet.address);
  console.log("Contract:", CONTRACT_ADDRESS);

  while (true) {
    const [state, challenge] = await Promise.all([
      contract.miningState(),
      contract.getChallenge(wallet.address),
    ]);

    const difficulty = BigInt(state.difficulty.toString());

    console.log("\n─── Mining Round ───────────────────────────────");
    console.log("Era       :", state.era.toString());
    console.log("Reward    :", ethers.formatUnits(state.reward, 18), "HASH");
    console.log("Difficulty:", difficulty.toString());
    console.log("Epoch     :", state.epoch.toString());
    console.log("Challenge :", challenge);

    try {
      const nonce = await mineOnGPU(challenge, difficulty);

      console.log("\nFOUND nonce:", nonce.toString());

      /* Verify challenge hasn't changed while GPU was mining */
      const currentChallenge = await contract.getChallenge(wallet.address);
      if (currentChallenge !== challenge) {
        console.log("Challenge changed during mining – restarting...");
        continue;
      }

      const tx = await contract.mine(nonce);
      console.log("TX sent  :", tx.hash);
      const receipt = await tx.wait();
      console.log("Success  : block", receipt.blockNumber);
    } catch (err) {
      console.error("Error:", err.shortMessage || err.message);
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}

main().catch(err => {
  console.error(err.shortMessage || err.message || err);
  process.exit(1);
});
