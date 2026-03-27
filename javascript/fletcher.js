/**
 * Fletcher Node.js SDK - Async embedding client
 */

class FletcherClient {
  /**
   * @param {string} baseUrl - Base URL of Fletcher server
   * @param {string} [apiKey] - Optional API key for authentication
   */
  constructor(baseUrl = "http://localhost:8080", apiKey = null) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeout = 30000;
  }

  _getHeaders() {
    const headers = { "Content-Type": "application/json" };
    if (this.apiKey) {
      headers["Authorization"] = this.apiKey;
    }
    return headers;
  }

  /**
   * Generate embedding for a single text
   * @param {string} text - Text to embed
   * @param {string} [model="fletcher-embed"] - Model name
   * @returns {Promise<number[]>}
   */
  async embed(text, model = "fletcher-embed") {
    const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify({ input: text, model }),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Embedding failed: ${error}`);
    }

    const data = await response.json();
    return data.data[0].embedding;
  }

  /**
   * Generate embeddings for multiple texts
   * @param {string[]} texts - Texts to embed
   * @param {string} [model="fletcher-embed"] - Model name
   * @returns {Promise<number[][]>}
   */
  async embedBatch(texts, model = "fletcher-embed") {
    const response = await fetch(`${this.baseUrl}/v1/embeddings/batch`, {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify({ inputs: texts, model }),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Batch embedding failed: ${error}`);
    }

    const data = await response.json();
    return data.data.map((item) => item.embedding);
  }

  /**
   * Rerank documents based on query relevance
   * @param {string} query - Query text
   * @param {string[]} documents - Documents to rerank
   * @param {number} [topN] - Number of top results to return
   * @returns {Promise<Object[]>}
   */
  async rerank(query, documents, topN = null) {
    const payload = { query, documents };
    if (topN) payload.top_n = topN;

    const response = await fetch(`${this.baseUrl}/v1/rerank`, {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Rerank failed: ${error}`);
    }

    const data = await response.json();
    return data.results;
  }

  /**
   * List available models
   * @returns {Promise<Object[]>}
   */
  async listModels() {
    const response = await fetch(`${this.baseUrl}/v1/models/list`, {
      method: "GET",
      headers: this._getHeaders(),
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`List models failed: ${error}`);
    }

    const data = await response.json();
    return data.data || [];
  }

  /**
   * Check server health
   * @returns {Promise<boolean>}
   */
  async health() {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    });
    return response.ok;
  }
}

module.exports = { FletcherClient };
