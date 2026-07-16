// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'mistralai/mistral-medium-3.5-128b',  // leve e rápido, bom para respostas ágeis
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',        // mantido, forte em coding/agentic
  'gpt-4-turbo': 'zai-org/GLM-5.2',                      // multilíngue, top-tier, substitui Kimi K2 (descontinuado)
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash',             // mantido, conforme solicitado
  'claude-3-opus': 'deepseek-ai/deepseek-v4-pro',        // #1 no ranking atual, ótimo para roleplay/chat pesado
  'claude-3-sonnet': 'openai/gpt-oss-20b',               // mantido, sólido custo-benefício
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'       // mantido, bom para raciocínio passo a passo
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Fallback models to try, in order, if the primary model is degraded/unavailable
const FALLBACK_MODELS = [
  'deepseek-ai/deepseek-v4-pro',
  'zai-org/GLM-5.2',
  'openai/gpt-oss-20b'
];

// Helper: sleep for a given number of ms
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Helper: does this error mean the model itself is broken (not just rate-limited)?
function isModelUnavailableError(error) {
  const status = error.response?.status;
  const detail = error.response?.data?.detail || error.response?.data?.error?.message || '';
  return status === 400 && /degraded|cannot be invoked/i.test(detail);
}

// Helper: call NIM API with retry (429/503) + automatic model fallback (degraded model)
async function callNimWithRetry(nimRequest, axiosConfig, maxRetries = 1) {
  const configWithTimeout = { ...axiosConfig, timeout: 25000 };
  const modelsToTry = [nimRequest.model, ...FALLBACK_MODELS.filter(m => m !== nimRequest.model)];

  for (let modelIndex = 0; modelIndex < modelsToTry.length; modelIndex++) {
    const currentModel = modelsToTry[modelIndex];
    const requestForThisModel = { ...nimRequest, model: currentModel };
    let attempt = 0;

    while (attempt <= maxRetries) {
      try {
        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, requestForThisModel, configWithTimeout);
        if (currentModel !== nimRequest.model) {
          console.warn(`Fallback succeeded: ${nimRequest.model} -> ${currentModel}`);
        }
        return response;
      } catch (error) {
        const status = error.response?.status;

        // Model itself is broken -> stop retrying this model, move to next fallback model
        if (isModelUnavailableError(error)) {
          console.warn(`Model ${currentModel} unavailable (degraded). Trying next fallback...`);
          break;
        }

        // Rate limited / temporarily overloaded -> retry same model with backoff
        const isRetryable = status === 429 || status === 503;
        if (!isRetryable || attempt === maxRetries) {
          // Not retryable, or out of retries for this model -> try next fallback model
          if (modelIndex === modelsToTry.length - 1) {
            throw error; // no more fallbacks left
          }
          console.warn(`Model ${currentModel} failed (status ${status}). Trying next fallback...`);
          break;
        }

        const retryAfterHeader = error.response?.headers?.['retry-after'];
        let delayMs;
        if (retryAfterHeader) {
          delayMs = parseInt(retryAfterHeader, 10) * 1000;
        } else {
          const baseDelay = 1000 * Math.pow(2, attempt);
          const jitter = Math.random() * 500;
          delayMs = baseDelay + jitter;
        }

        console.warn(`NIM API returned ${status} for ${currentModel}. Retry ${attempt + 1}/${maxRetries} after ${Math.round(delayMs)}ms`);
        await sleep(delayMs);
        attempt++;
      }
    }
  }

  throw new Error('All models (primary + fallbacks) failed');
}

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API (with automatic retry on 429/503)
    const response = await callNimWithRetry(nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500,
        nim_details: error.response?.data || null
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  });
}

module.exports = app;