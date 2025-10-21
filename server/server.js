import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';
import WebSocket, { WebSocketServer } from 'ws';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_REALTIME_MODEL = process.env.OPENAI_REALTIME_MODEL ?? 'gpt-4o-realtime-preview-2024-12-17';
const OPENAI_EMOTION_MODEL = process.env.OPENAI_EMOTION_MODEL ?? 'gpt-4o-mini';
const OPENAI_REALTIME_ENDPOINT = process.env.OPENAI_REALTIME_ENDPOINT ?? 'wss://api.openai.com/v1/realtime';
const PORT = Number.parseInt(process.env.PORT ?? '3000', 10);
const HOST = process.env.HOST ?? '0.0.0.0';

if (!OPENAI_API_KEY) {
  throw new Error('OPENAI_API_KEY is required to start the realtime server');
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const publicDir = resolve(__dirname, '../public');

const defaultPersona = `You are an adaptive AI guide named Aurora. You are transparent about being an AI companion and help users navigate immersive experiences. You speak concisely, narrate what you understand, and offer proactive suggestions when appropriate.`;

const headers = {
  Authorization: `Bearer ${OPENAI_API_KEY}`,
  'OpenAI-Beta': 'realtime=v1',
};

function resample24kTo48k(int16Array) {
  const output = new Int16Array(int16Array.length * 2);
  for (const [index, sample] of int16Array.entries()) {
    const targetIndex = index * 2;
    output[targetIndex] = sample;
    output[targetIndex + 1] = sample;
  }
  return output;
}

function bufferFromBase64(base64) {
  return Buffer.from(base64, 'base64');
}

function base64FromInt16(int16Array) {
  return Buffer.from(int16Array.buffer, int16Array.byteOffset, int16Array.byteLength).toString('base64');
}

async function detectEmotion({ text, context }) {
  if (!text?.trim()) {
    return { emotion: 'neutral', intensity: 1 };
  }

  const prompt = `You are an emotion classifier. Given an assistant reply, pick the primary emotion and its intensity from 1 (calm) to 5 (very strong). Respond with strict JSON: {"emotion":"<emotion>","intensity":<1-5>}. Reply in lowercase emotion labels. Consider the conversation context if provided.

Context: ${context ?? 'n/a'}
Assistant reply: ${text}`;

  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: OPENAI_EMOTION_MODEL,
      input: prompt,
      temperature: 0,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Emotion analysis failed', errorText);
    return { emotion: 'neutral', intensity: 1 };
  }

  const payload = await response.json();
  let raw = payload?.output?.[0]?.content?.[0]?.text;
  if (!raw) {
    if (Array.isArray(payload?.output_text)) {
      raw = payload.output_text.join(' ');
    } else if (typeof payload?.output_text === 'string') {
      raw = payload.output_text;
    }
  }
  raw = raw ?? '';
  try {
    const parsed = JSON.parse(raw);
    const normalizedIntensity = Number.parseInt(parsed.intensity ?? 1, 10) || 1;
    return {
      emotion: String(parsed.emotion ?? 'neutral'),
      intensity: Math.min(5, Math.max(1, normalizedIntensity)),
    };
  } catch (error) {
    console.error('Emotion analysis parsing error', error, raw);
    return { emotion: 'neutral', intensity: 1 };
  }
}

class ClientSession {
  constructor(socket) {
    this.client = socket;
    this.persona = defaultPersona;
    this.intro = 'Hello! I am Aurora, your AI guide. How can I help you today?';
    this.voice = process.env.OPENAI_VOICE ?? 'verse';
    this.micEnabled = true;
    this.streaming = false;
    this.pendingText = '';
    this.openAIConnected = false;
    this.contextHistory = [];

    this.setupClientListeners();
    this.safeSend({ type: 'mic', enabled: this.micEnabled });
    this.connectToOpenAI().catch((error) => {
      console.error('Failed to connect to OpenAI realtime', error);
      this.safeSend({
        type: 'error',
        message: 'Failed to connect to OpenAI realtime service',
      });
    });
  }

  safeSend(message) {
    if (this.client.readyState === WebSocket.OPEN) {
      this.client.send(JSON.stringify(message));
    }
  }

  async connectToOpenAI() {
    const url = `${OPENAI_REALTIME_ENDPOINT}?model=${encodeURIComponent(OPENAI_REALTIME_MODEL)}`;
    this.openAI = new WebSocket(url, { headers });

    this.openAI.on('open', () => {
      this.openAIConnected = true;
      this.updateSessionInstructions();
      this.safeSend({ type: 'status', status: 'connected' });
    });

    this.openAI.on('message', (data) => {
      const payload = data.toString();
      if (!payload.trim()) {
        return;
      }
      try {
        const event = JSON.parse(payload);
        this.handleOpenAIEvent(event);
      } catch (error) {
        console.error('Failed to parse realtime event', error, payload);
      }
    });

    this.openAI.on('close', () => {
      this.openAIConnected = false;
      this.safeSend({ type: 'status', status: 'disconnected' });
    });

    this.openAI.on('error', (error) => {
      console.error('Realtime socket error', error);
      this.safeSend({ type: 'error', message: 'Realtime socket error' });
    });
  }

  setupClientListeners() {
    this.client.on('message', (message) => {
      let data;
      try {
        data = JSON.parse(message.toString());
      } catch (error) {
        console.error('Invalid client message', error);
        return;
      }

      switch (data.type) {
        case 'text': {
          this.handleTextInput(data.text ?? '');
          break;
        }
        case 'audio-chunk': {
          this.handleAudioChunk(data);
          break;
        }
        case 'audio-stop': {
          this.commitAudio();
          break;
        }
        case 'toggle-mic': {
          this.toggleMic(Boolean(data.enabled));
          break;
        }
        case 'set-persona': {
          this.setPersona(data.persona, data.voice);
          break;
        }
        case 'set-intro': {
          this.setIntro(data.intro);
          break;
        }
        case 'flush': {
          this.flushOutputs('client-request');
          break;
        }
        default: {
          console.warn('Unhandled client event', data);
          break;
        }
      }
    });

    this.client.on('close', () => {
      this.teardown();
    });
  }

  updateSessionInstructions() {
    if (!this.openAIConnected) {
      return;
    }

    const introClause = this.intro
      ? `Your first reply in any new conversation must start with exactly: "${this.intro}" before continuing naturally.`
      : '';

    this.openAI.send(JSON.stringify({
      type: 'session.update',
      session: {
        instructions: `${this.persona}\n\n${introClause}`.trim(),
        modalities: ['text', 'audio'],
        voice: this.voice,
      },
    }));

    this.safeSend({
      type: 'session',
      persona: this.persona,
      intro: this.intro,
      voice: this.voice,
    });
  }

  setPersona(persona, voice) {
    const sanitizedPersona = persona?.trim();
    this.persona = sanitizedPersona || defaultPersona;

    if (voice?.trim()) {
      this.voice = voice.trim();
    }

    this.updateSessionInstructions();
  }

  setIntro(intro) {
    if (intro?.trim()) {
      this.intro = intro.trim();
    }
    this.updateSessionInstructions();
  }

  handleTextInput(text) {
    const payload = text?.trim();
    if (!payload) {
      return;
    }

    this.bargeIn();
    this.contextHistory.push({ role: 'user', text: payload });
    if (this.contextHistory.length > 20) {
      this.contextHistory = this.contextHistory.slice(-20);
    }

    if (!this.openAIConnected) {
      this.safeSend({ type: 'error', message: 'Realtime service unavailable' });
      return;
    }

    this.openAI.send(JSON.stringify({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [
          {
            type: 'input_text',
            text: payload,
          },
        ],
      },
    }));

    this.openAI.send(JSON.stringify({ type: 'response.create' }));
  }

  handleAudioChunk({ audio }) {
    if (!this.micEnabled || !audio) {
      return;
    }

    this.bargeIn();

    if (!this.openAIConnected) {
      return;
    }

    this.openAI.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio,
    }));
  }

  commitAudio() {
    if (!this.openAIConnected) {
      return;
    }

    this.openAI.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    this.openAI.send(JSON.stringify({ type: 'response.create' }));
  }

  toggleMic(enabled) {
    this.micEnabled = enabled;
    this.safeSend({ type: 'mic', enabled: this.micEnabled });
  }

  flushOutputs(reason = 'barge-in') {
    if (this.streaming && this.openAIConnected) {
      this.openAI.send(JSON.stringify({ type: 'response.cancel' }));
    }

    this.streaming = false;
    this.pendingText = '';
    this.safeSend({ type: 'flush', reason });
  }

  bargeIn() {
    if (this.streaming) {
      this.flushOutputs('barge-in');
    }
  }

  async handleOpenAIEvent(event) {
    switch (event.type) {
      case 'response.output_text.delta': {
        this.streaming = true;
        this.pendingText += event.delta ?? '';
        this.safeSend({
          type: 'text-delta',
          delta: event.delta ?? '',
          full: this.pendingText,
        });
        break;
      }
      case 'response.output_audio.delta': {
        this.streaming = true;
        if (event.delta) {
          const buffer = bufferFromBase64(event.delta);
          const int16 = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
          const resampled = resample24kTo48k(int16);
          const base64 = base64FromInt16(resampled);
          this.safeSend({ type: 'audio-delta', audio: base64 });
        }
        break;
      }
      case 'response.completed': {
        await this.handleResponseCompleted(event);
        break;
      }
      case 'response.error': {
        console.error('Realtime response error', event);
        this.safeSend({ type: 'error', message: 'Realtime response error' });
        this.streaming = false;
        this.pendingText = '';
        break;
      }
      case 'session.updated': {
        this.safeSend({ type: 'status', status: 'ready' });
        break;
      }
      default: {
        break;
      }
    }
  }

  async handleResponseCompleted(event) {
    this.streaming = false;
    const finalText = this.pendingText.trim();
    this.pendingText = '';

    if (!finalText) {
      this.safeSend({ type: 'response-complete', text: '' });
      return;
    }

    this.contextHistory.push({ role: 'assistant', text: finalText });
    if (this.contextHistory.length > 20) {
      this.contextHistory = this.contextHistory.slice(-20);
    }

    const recentContext = this.contextHistory.slice(-6)
      .map((entry) => `${entry.role.toUpperCase()}: ${entry.text}`)
      .join('\n');

    const emotion = await detectEmotion({ text: finalText, context: recentContext });

    this.safeSend({
      type: 'response-complete',
      text: finalText,
      emotion: emotion.emotion,
      intensity: emotion.intensity,
    });
  }

  teardown() {
    if (this.openAI && this.openAI.readyState === WebSocket.OPEN) {
      this.openAI.close();
    }
  }
}

async function serveFile(path, res) {
  try {
    const stream = createReadStream(path);
    stream.on('error', (error) => {
      console.error('Static file error', error);
      res.statusCode = 500;
      res.end('Internal Server Error');
    });
    stream.pipe(res);
  } catch (error) {
    console.error('Static file streaming error', error);
    res.statusCode = 404;
    res.end('Not Found');
  }
}

const server = createServer(async (req, res) => {
  if (!req.url) {
    res.statusCode = 400;
    res.end('Bad Request');
    return;
  }

  const url = new URL(req.url, 'http://localhost');
  const requestPath = url.pathname === '/' ? '/tester.html' : url.pathname;
  const filePath = resolve(publicDir, `.${requestPath}`);

  if (!filePath.startsWith(publicDir)) {
    res.statusCode = 403;
    res.end('Forbidden');
    return;
  }

  try {
    await readFile(filePath);
    const ext = extname(filePath);
    switch (ext) {
      case '.html': {
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        break;
      }
      case '.js': {
        res.setHeader('Content-Type', 'application/javascript');
        break;
      }
      case '.css': {
        res.setHeader('Content-Type', 'text/css');
        break;
      }
      default: {
        break;
      }
    }
    await serveFile(filePath, res);
  } catch (error) {
    console.error('Static request error', error);
    res.statusCode = 404;
    res.end('Not Found');
  }
});

const wss = new WebSocketServer({ server });
wss.on('connection', (socket) => {
  new ClientSession(socket);
});

server.listen(PORT, HOST, () => {
  console.log(`Realtime server listening on http://${HOST}:${PORT}`);
  console.log('Serving tester client at /tester.html');
});
