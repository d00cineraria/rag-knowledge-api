package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// searchDocumentPrefix はnomic-embed-textの非対称検索用プレフィックス。
// APIパラメータではなくテキスト側に前置する必要がある。
const searchDocumentPrefix = "search_document: "

// OllamaClient はローカルOllamaの/api/embedエンドポイントを呼び出すクライアント。
type OllamaClient struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

// NewOllamaClient はOllama embedding APIを呼び出すOllamaClientを構築する。
func NewOllamaClient(baseURL, model string) *OllamaClient {
	return &OllamaClient{
		baseURL:    baseURL,
		model:      model,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

// Embed はtextsに"search_document: "を前置し、/api/embedでベクトル化してL2正規化して返す。
// 戻り値の長さと順序はtextsに対応する。
func (c *OllamaClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	result := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += maxBatchSize {
		end := min(start+maxBatchSize, len(texts))

		batch, err := c.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, fmt.Errorf("embedding: batch [%d:%d]: %w", start, end, err)
		}
		result = append(result, batch...)
	}
	return result, nil
}

// embedBatch はtexts（最大maxBatchSize件）を1回のHTTPリクエストでベクトル化する。
func (c *OllamaClient) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	input := make([]string, len(texts))
	for i, text := range texts {
		input[i] = searchDocumentPrefix + text
	}

	reqBody := ollamaEmbedRequest{Model: c.model, Input: input}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := c.baseURL + "/api/embed"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama api status %d: %s", resp.StatusCode, string(respBody))
	}

	var parsed ollamaEmbedResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	if len(parsed.Embeddings) != len(texts) {
		return nil, fmt.Errorf(
			"response size mismatch: got %d embeddings for %d texts",
			len(parsed.Embeddings), len(texts),
		)
	}

	out := make([][]float32, len(parsed.Embeddings))
	for i, values := range parsed.Embeddings {
		normalized := L2Normalize(values)
		vec := make([]float32, len(normalized))
		for j, v := range normalized {
			vec[j] = float32(v)
		}
		out[i] = vec
	}
	return out, nil
}
