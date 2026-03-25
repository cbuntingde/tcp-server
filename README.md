We'll create a **custom agent** that uses declarative tool definitions from `tsp.json` files, along with documentation you could use as official reference. This approach gives you the flexibility of MCP-like tool definitions without the overhead of separate servers.

---

## 1. What is tsp.json?

`tsp.json` (Tool Specification) is a configuration file that defines a tool's metadata, parameters, and the location of its TypeScript implementation. Instead of running a separate MCP server, the agent loads these definitions directly and calls the functions in‑process.

**Benefits:**
- ✅ Lower latency and token usage
- ✅ Easier security (API keys stay in your code/env)
- ✅ Full TypeScript type safety
- ✅ No process management
- ✅ Declarative configuration for reusability

---

## 2. Architecture Overview

```
project/
├── tools/
│   ├── brave-search.tsp.json
│   ├── brave-search.ts          # implementation
│   └── weather.tsp.json
├── agent/
│   ├── agent.ts                 # custom agent logic
│   └── toolLoader.ts            # loads tools from tsp.json
└── .env                         # API keys
```

**Flow:**
1. Agent starts, loads all `*.tsp.json` files from `tools/` directory.
2. For each tool, imports the specified TypeScript module and function.
3. Converts JSON Schema parameters to Zod (or keeps as JSON Schema) for validation.
4. Registers the tool with a name and executor.
5. When the LLM decides to call a tool, the agent invokes the executor with validated arguments.

---

## 3. Implementation Details

### 3.1 Tool Specification (`brave-search.tsp.json`)

```json
{
  "name": "brave_search",
  "description": "Search the web using Brave Search. Returns titles, URLs, and snippets.",
  "module": "./brave-search.ts",
  "function": "braveSearch",
  "parameters": {
    "type": "object",
    "properties": {
      "q": {
        "type": "string",
        "description": "The search query"
      },
      "count": {
        "type": "number",
        "description": "Number of results (max 20)",
        "minimum": 1,
        "maximum": 20,
        "default": 10
      },
      "safesearch": {
        "type": "string",
        "enum": ["strict", "moderate", "off"],
        "description": "Safe search level",
        "default": "moderate"
      }
    },
    "required": ["q"]
  },
  "returns": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "title": { "type": "string" },
        "url": { "type": "string" },
        "snippet": { "type": "string" }
      }
    }
  }
}
```

### 3.2 Tool Implementation (`brave-search.ts`)

```typescript
import axios from 'axios';

export interface BraveSearchParams {
  q: string;
  count?: number;
  safesearch?: 'strict' | 'moderate' | 'off';
}

export interface BraveWebResult {
  title: string;
  url: string;
  description: string;
}

export async function braveSearch(params: BraveSearchParams): Promise<BraveWebResult[]> {
  const API_KEY = process.env.BRAVE_API_KEY;
  if (!API_KEY) throw new Error('Missing BRAVE_API_KEY');

  const url = new URL('https://api.search.brave.com/res/v1/web/search');
  url.searchParams.append('q', params.q);
  if (params.count) url.searchParams.append('count', String(params.count));
  if (params.safesearch) url.searchParams.append('safesearch', params.safesearch);

  const response = await axios.get(url.toString(), {
    headers: { 'X-Subscription-Token': API_KEY, 'Accept': 'application/json' }
  });

  return response.data.web?.results.map((r: any) => ({
    title: r.title,
    url: r.url,
    snippet: r.description
  })) || [];
}
```

### 3.3 Tool Loader (`toolLoader.ts`)

```typescript
import fs from 'fs/promises';
import path from 'path';
import { z } from 'zod';
import { ZodSchema } from 'zod';

export interface ToolSpec {
  name: string;
  description: string;
  module: string;
  function: string;
  parameters: any; // JSON Schema
  returns?: any;
}

export interface LoadedTool {
  name: string;
  description: string;
  parameters: ZodSchema;
  execute: (args: any) => Promise<any>;
}

// Minimal JSON Schema to Zod converter (expand as needed)
function jsonSchemaToZod(schema: any): ZodSchema {
  if (schema.type === 'object') {
    const shape: Record<string, ZodSchema> = {};
    for (const [key, prop] of Object.entries(schema.properties || {})) {
      shape[key] = jsonSchemaToZod(prop);
    }
    let zodObj = z.object(shape);
    // Handle required fields: Zod object properties are required by default.
    // If a property is not in `required` array, we need to make it optional.
    if (schema.required && Array.isArray(schema.required)) {
      const requiredSet = new Set(schema.required);
      for (const key of Object.keys(shape)) {
        if (!requiredSet.has(key)) {
          zodObj = zodObj.extend({ [key]: shape[key].optional() });
        }
      }
    }
    return zodObj;
  }
  if (schema.type === 'string') {
    if (schema.enum) return z.enum(schema.enum as [string, ...string[]]);
    return z.string();
  }
  if (schema.type === 'number') return z.number();
  if (schema.type === 'boolean') return z.boolean();
  if (schema.type === 'array') {
    return z.array(jsonSchemaToZod(schema.items));
  }
  return z.any();
}

export async function loadToolsFromDirectory(dir: string): Promise<LoadedTool[]> {
  const files = await fs.readdir(dir);
  const toolFiles = files.filter(f => f.endsWith('.tsp.json'));
  const tools: LoadedTool[] = [];

  for (const file of toolFiles) {
    const filePath = path.join(dir, file);
    const content = await fs.readFile(filePath, 'utf-8');
    const spec: ToolSpec = JSON.parse(content);

    // Dynamically import the module
    const modulePath = path.resolve(process.cwd(), dir, spec.module);
    const toolModule = await import(modulePath);
    const toolFunc = toolModule[spec.function];
    if (!toolFunc || typeof toolFunc !== 'function') {
      throw new Error(`Function ${spec.function} not found in ${spec.module}`);
    }

    const parametersSchema = jsonSchemaToZod(spec.parameters);

    tools.push({
      name: spec.name,
      description: spec.description,
      parameters: parametersSchema,
      execute: async (args: any) => {
        const validated = parametersSchema.parse(args);
        return await toolFunc(validated);
      },
    });
  }
  return tools;
}
```

### 3.4 Custom Agent (`agent.ts`)

Here’s a simple **custom agent** using OpenAI’s function‑calling API, but the pattern applies to any LLM.

```typescript
import OpenAI from 'openai';
import { loadToolsFromDirectory } from './toolLoader';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function runAgent(userQuery: string) {
  // Load tools dynamically
  const tools = await loadToolsFromDirectory('./tools');

  // Convert to OpenAI function definitions
  const functions = tools.map(tool => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters, // Zod schema can be converted to JSON Schema
  }));

  let messages: any[] = [{ role: 'user', content: userQuery }];
  let response;

  while (true) {
    response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      functions,
      function_call: 'auto',
    });

    const message = response.choices[0].message;
    messages.push(message);

    // If no function call, we're done
    if (!message.function_call) break;

    const toolName = message.function_call.name;
    const toolArgs = JSON.parse(message.function_call.arguments);

    // Find the matching loaded tool
    const tool = tools.find(t => t.name === toolName);
    if (!tool) {
      throw new Error(`Tool ${toolName} not found`);
    }

    // Execute the tool
    const result = await tool.execute(toolArgs);
    messages.push({
      role: 'function',
      name: toolName,
      content: JSON.stringify(result),
    });
  }

  // Return final answer
  return response?.choices[0].message.content;
}

// Example usage
runAgent('What are the latest TypeScript best practices?')
  .then(console.log)
  .catch(console.error);
```

---

## 4. Official Documentation (Style Guide)

Below is how you could present this as **official documentation** for developers.

---

# Tool Specification (tsp.json) – Developer Guide

## Overview

The **Tool Specification** (`tsp.json`) system allows you to define tools for AI agents in a declarative way, using TypeScript for implementation. It replaces the need for running separate MCP servers, providing a simpler, more efficient integration.

## Why Use tsp.json?

- **Simplicity**: Define a tool with a JSON file and a TypeScript function. No separate process.
- **Performance**: Direct function calls → lower latency and reduced token usage.
- **Security**: API keys stay in your environment variables, never exposed.
- **Type Safety**: Full TypeScript support with automatic argument validation.

## Getting Started

### 1. Install Dependencies

```bash
npm install zod axios openai   # or your preferred AI SDK
```

### 2. Set Environment Variables

Create a `.env` file:

```
BRAVE_API_KEY=your_brave_api_key
OPENAI_API_KEY=your_openai_key
```

### 3. Create a Tool

Create `tools/brave-search.tsp.json`:

```json
{
  "name": "brave_search",
  "description": "Search the web using Brave Search",
  "module": "./brave-search.ts",
  "function": "braveSearch",
  "parameters": {
    "type": "object",
    "properties": {
      "q": { "type": "string", "description": "The search query" },
      "count": { "type": "number", "description": "Number of results", "default": 10 }
    },
    "required": ["q"]
  }
}
```

Create `tools/brave-search.ts`:

```typescript
export async function braveSearch({ q, count = 10 }) {
  // Implementation using Brave Search API
  // Returns array of results
}
```

### 4. Load and Use Tools in Your Agent

```typescript
import { loadToolsFromDirectory } from './toolLoader';

const tools = await loadToolsFromDirectory('./tools');
// Pass tools to your agent framework (OpenAI, LangChain, etc.)
```

## Tool Specification Reference

| Field          | Type   | Description |
|----------------|--------|-------------|
| `name`         | string | Unique identifier for the tool. |
| `description`  | string | Human-readable description (used by LLM). |
| `module`       | string | Path to the TypeScript file (relative to the tsp.json location). |
| `function`     | string | Name of the exported function to call. |
| `parameters`   | object | JSON Schema describing input parameters. |
| `returns`      | object | Optional JSON Schema for expected output (used for documentation). |

### Parameter Schema (JSON Schema subset)

Supported types: `string`, `number`, `boolean`, `object`, `array`.  
Use `enum` for strings, `minimum`/`maximum` for numbers, `properties` for objects, `items` for arrays.

## Custom Agent Integration

The loader returns an array of `LoadedTool` objects:

```typescript
interface LoadedTool {
  name: string;
  description: string;
  parameters: ZodSchema;   // validated schema
  execute: (args: any) => Promise<any>;
}
```

You can integrate this with any LLM or agent framework by converting the `parameters` Zod schema to your framework's required format (e.g., OpenAI functions, LangChain tools).

### Example with OpenAI Function Calling

```typescript
const functions = tools.map(t => ({
  name: t.name,
  description: t.description,
  parameters: zodToJsonSchema(t.parameters),
}));
```

## Security Best Practices

- **Never expose API keys in client‑side code**.
- Store keys in environment variables (e.g., using `dotenv`).
- Run the agent on a trusted backend or serverless function.

## Advanced: Customizing the Loader

The provided `toolLoader` can be extended to:

- Support additional validation (e.g., using JSON Schema directly).
- Cache loaded tools.
- Hot‑reload tools on file changes.

## Troubleshooting

- **Module not found**: Ensure the `module` path is correct relative to the `tsp.json` file.
- **Function not exported**: Check that the function name matches exactly.
- **Validation errors**: Ensure arguments match the defined JSON Schema.

## Conclusion

`tsp.json` offers a lightweight, type‑safe alternative to MCP servers. It reduces complexity, improves performance, and keeps your tools maintainable.

---

## 5. Complete Example Repository Structure

```
my-agent/
├── .env
├── package.json
├── tools/
│   ├── brave-search.tsp.json
│   ├── brave-search.ts
│   └── weather.tsp.json
├── agent/
│   ├── agent.ts
│   └── toolLoader.ts
└── README.md
```

---

This approach gives you the flexibility of a tool ecosystem without the operational overhead. It can be extended to support any number of tools and any LLM provider. If you need further help adapting this to your specific framework or adding more features, just let me know!
