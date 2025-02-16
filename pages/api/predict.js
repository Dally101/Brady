// pages/api/predict.js

import fs from 'fs';
import { IncomingForm } from 'formidable';
import sharp from 'sharp';

const modelPath = "C:/Users/Abhiroop/runs/classify/train#2/weights/best.onnx";

let session; // Cached session

// Dynamically load onnxruntime-node
async function loadORT() {
  return await import("onnxruntime-node");
}

async function loadModel() {
  if (!session) {
    const ort = await loadORT();
    session = await ort.InferenceSession.create(modelPath);
  }
  return session;
}

// Preprocess image: resize to 640x640, remove alpha channel, and normalize.
async function preprocessImage(imageBuffer) {
  const width = 640;
  const height = 640;
  const { data } = await sharp(imageBuffer)
    .resize(width, height)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  
  const numElements = data.length; // 640 * 640 * 3
  const nhwcTensor = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    nhwcTensor[i] = data[i] / 255.0;
  }
  
  // Convert NHWC to NCHW
  const nchwTensor = new Float32Array(640 * 640 * 3);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const nhwcIndex = i * width * 3 + j * 3 + c;
        const nchwIndex = c * height * width + i * width + j;
        nchwTensor[nchwIndex] = nhwcTensor[nhwcIndex];
      }
    }
  }
  return nchwTensor;
}

// Run inference using the ONNX model.
async function runInference(imageBuffer) {
  const model = await loadModel();
  const inputTensor = await preprocessImage(imageBuffer);
  // Adjust the input key to match your model's requirement.
  const feeds = { images: new (await loadORT()).Tensor("float32", inputTensor, [1, 3, 640, 640]) };
  const results = await model.run(feeds);
  // Use the first available key if "output" isn't defined.
  const outputKey = results.output ? "output" : Object.keys(results)[0];
  const outputTensor = results[outputKey];
  if (!outputTensor || !outputTensor.data) {
    throw new Error(`Output tensor not found for key ${outputKey}`);
  }
  return Array.from(outputTensor.data);
}

const baseCodeDescriptions = {
  "OSHA 1910.37(a)(3)": "Employers must provide machine guarding for fixed machinery to protect workers from moving parts.",
  "OSHA 1910.303(e)(1)": "Electrical equipment must be marked with the manufacturer's identification and rating information.",
  "OSHA 1910.303(g)(1)": "Adequate working space must be maintained around electrical equipment for safe operation and maintenance.",
  "OSHA 1910.157(c)(1)": "Portable fire extinguishers must be provided, properly mounted, and clearly identified for quick access.",
  "ANSI A13.1 (Pipe Marking)": "This standard defines requirements for marking and identifying piping systems using color codes and labels.",
  "ANSI Z358.1-2014 (Emergency Equipment)": "This standard specifies requirements for emergency eyewash and shower equipment to ensure rapid decontamination."
};

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const form = new IncomingForm();
  form.parse(req, async (err, fields, files) => {
    if (err) {
      console.error("Error parsing form:", err);
      return res.status(500).json({ error: 'Error parsing the file' });
    }

    let file = files.image;
    if (Array.isArray(file)) file = file[0];
    if (!file) {
      return res.status(400).json({ error: 'No image provided' });
    }
    const filePath = file.filepath || file.path;
    if (!filePath) {
      console.error("Invalid file path:", file);
      return res.status(400).json({ error: 'No valid file path provided.' });
    }

    const imageBuffer = fs.readFileSync(filePath);
    const predictions = await runInference(imageBuffer);
    console.log("Raw predictions:", predictions);

    const classNames = [
      "OSHA 1910.37(a)(3)_Before",
      "OSHA 1910.37(a)(3)_After",
      "OSHA 1910.303(e)(1)_Before",
      "OSHA 1910.303(e)(1)_After",
      "OSHA 1910.303(g)(1)_Before",
      "OSHA 1910.303(g)(1)_After",
      "OSHA 1910.157(c)(1)_Before",
      "OSHA 1910.157(c)(1)_After",
      "ANSI A13.1 (Pipe Marking)_Before",
      "ANSI A13.1 (Pipe Marking)_After",
      "ANSI Z358.1-2014 (Emergency Equipment)_Before",
      "ANSI Z358.1-2014 (Emergency Equipment)_After"
    ];

    const sortedIndices = predictions
      .map((prob, idx) => ({ index: idx, prob }))
      .sort((a, b) => b.prob - a.prob);
    console.log("Sorted indices:", sortedIndices);

    // Accumulate predictions until at least 99% of the probability mass is reached.
    let cumulativeProb = 0;
    let result = [];
    for (let i = 0; i < sortedIndices.length; i++) {
      cumulativeProb += sortedIndices[i].prob;
      const idx = sortedIndices[i].index;
      const baseCode = idx % 2 === 0
        ? classNames[idx].replace('_Before', '')
        : classNames[idx].replace('_After', '');
      result.push({
        prediction: idx % 2 === 0
          ? `Violation - ${baseCode}`
          : `No Violation - ${baseCode}`,
        probability: sortedIndices[i].prob,
        caption: idx % 2 === 0
          ? baseCodeDescriptions[baseCode] || "No description available."
          : "No violation detected.",
        code: baseCode
      });
      if (cumulativeProb >= 0.99) break;
    }

    return res.status(200).json({ predictions: result });
  });
}
