import Replicate from 'replicate';
import formidable from 'formidable';
import fs from 'fs';
import fetch from 'node-fetch';

export const config = { api: { bodyParser: false } };

export default async function handler(req, res) {
  // Allow CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    // Parse the uploaded image
    const form = formidable({ maxFileSize: 10 * 1024 * 1024 });
    const [, files] = await form.parse(req);
    const imageFile = files.image?.[0];
    if (!imageFile) return res.status(400).json({ error: 'No image provided' });

    // Read file and convert to base64 data URL
    const imageBuffer = fs.readFileSync(imageFile.filepath);
    const base64Image = `data:${imageFile.mimetype};base64,${imageBuffer.toString('base64')}`;

    // Call Replicate — using grounded SAM to detect "floor" specifically
    const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

    const output = await replicate.run(
      'schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c',
      {
        input: {
          image: base64Image,
          prompt: 'floor, flooring, ground, tile floor, wood floor, concrete floor, carpet',
          box_threshold: 0.3,
          text_threshold: 0.25,
        }
      }
    );

    // output contains segmented image — parse the mask
    // Grounded SAM returns a combined image; we use the mask data
    if (!output || !output.mask) {
      // Fallback: use the bottom 55% of image as floor estimate
      return res.status(200).json({ mask: null, fallback: true });
    }

    // Fetch the mask image from Replicate CDN
    const maskResponse = await fetch(output.mask);
    const maskBuffer = await maskResponse.buffer();

    // Return mask as base64 for the client to process
    return res.status(200).json({
      mask: maskBuffer.toString('base64'),
      maskType: 'image/png',
      success: true
    });

  } catch (error) {
    console.error('Detect floor error:', error);
    return res.status(500).json({ 
      error: 'Floor detection failed', 
      message: error.message,
      fallback: true 
    });
  }
}
