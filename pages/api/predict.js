// pages/api/predict.js

import fs from 'fs';
import { IncomingForm } from 'formidable';
import sharp from 'sharp';
import * as ort from 'onnxruntime-node';

// Use a relative path from the project root.
const modelPath = "public/model/best.onnx";

let session;
async function loadModel() {
  if (!session) {
    session = await ort.InferenceSession.create(modelPath);
  }
  return session;
}

// Preprocess the image: resize to 640x640, remove alpha channel, and normalize pixel values.
async function preprocessImage(imageBuffer) {
  const width = 640;
  const height = 640;
  const { data } = await sharp(imageBuffer)
    .resize(width, height)
    .removeAlpha() // ensures image has exactly 3 channels (RGB)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const numElements = data.length; // Expected: 640 * 640 * 3 = 1,228,800
  const nhwcTensor = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    nhwcTensor[i] = data[i] / 255.0;
  }

  // Convert NHWC to NCHW, as many models expect NCHW input
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

// Run inference using onnxruntime-node.
async function runInference(imageBuffer) {
  const model = await loadModel();
  const inputTensor = await preprocessImage(imageBuffer);
  // The model expects an input with key "images" and shape [1, 3, 640, 640]
  const feeds = { images: new ort.Tensor("float32", inputTensor, [1, 3, 640, 640]) };
  const results = await model.run(feeds);
  // If the output is not under "output", take the first key.
  const outputKey = results.output ? "output" : Object.keys(results)[0];
  const outputTensor = results[outputKey];
  if (!outputTensor || !outputTensor.data) {
    throw new Error(`Output tensor not found for key ${outputKey}`);
  }
  return Array.from(outputTensor.data);
}

// Base code descriptions for each regulatory standard.
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
    bodyParser: false, // Disable Next.js's default body parser so that formidable can handle file uploads.
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

    // Handle the "image" field; if multiple, take the first.
    let file = files.image;
    if (Array.isArray(file)) {
      file = file[0];
    }
    if (!file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const filePath = file.filepath || file.path;
    if (!filePath) {
      console.error("File object does not have a valid path:", file);
      return res.status(400).json({ error: 'No valid file path provided.' });
    }

    // Read the image file into a buffer.
    const imageBuffer = fs.readFileSync(filePath);

    // Run inference on the image buffer.
    const predictions = await runInference(imageBuffer);
    console.log("Raw predictions:", predictions);

    // Define the 12 class names in order.
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

    // Sort predictions by probability in descending order.
    const sortedIndices = predictions
      .map((prob, idx) => ({ index: idx, prob }))
      .sort((a, b) => b.prob - a.prob);
    console.log("Sorted indices:", sortedIndices);

    // Instead of limiting to a fixed number, accumulate predictions until
    // the cumulative probability is at least 0.99.
    let cumulativeProb = 0;
    let result = [];
    for (let i = 0; i < sortedIndices.length; i++) {
      cumulativeProb += sortedIndices[i].prob;
      const idx = sortedIndices[i].index;
      const baseCode = idx % 2 === 0
        ? classNames[idx].replace('_Before', '')
        : classNames[idx].replace('_After', '');
      const predObj = {
        prediction: idx % 2 === 0
          ? `Violation - ${baseCode}`
          : `No Violation - ${baseCode}`,
        probability: sortedIndices[i].prob,
        caption: idx % 2 === 0
          ? baseCodeDescriptions[baseCode] || "No description available."
          : "No violation detected.",
        code: baseCode
      };
      result.push(predObj);
      if (cumulativeProb >= 0.99) break;
    }

    return res.status(200).json({ predictions: result });
  });
}
