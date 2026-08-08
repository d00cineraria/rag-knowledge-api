// Package embedding はGemini REST APIでテキストをベクトル化するクライアントを提供する。
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

// maxBatchSize は batchEmbedContents 1リクエストに含めるテキストの最大件数。
// これを超える件数は複数リクエストに分割する。
const maxBatchSize = 100

const defaultBaseURL = "https://generativelanguage.googleapis.com"

// Client はGemini embeddingモデルのRESTクライアント。
type Client struct {
	apiKey     string
	model      string
	dimension  int
	httpClient *http.Client
	baseURL    string // テストからモックサーバーへ差し替えるためのフィールド
}

// NewClient はGemini embedding APIを呼び出すClientを構築する。
func NewClient(apiKey, model string, dimension int) *Client {
	return &Client{
		apiKey:     apiKey,
		model:      model,
		dimension:  dimension,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		baseURL:    defaultBaseURL,
	}
}

// Embed はtextsをbatchEmbedContentsでベクトル化し、L2正規化して返す。
// 戻り値の長さと順序はtextsに対応する。
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
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

type batchEmbedRequest struct {
	Requests []embedContentRequest `json:"requests"`
}

type embedContentRequest struct {
	Model                string  `json:"model"`
	Content              content `json:"content"`
	TaskType             string  `json:"taskType"`
	OutputDimensionality int     `json:"outputDimensionality"`
}

type content struct {
	Parts []part `json:"parts"`
}

type part struct {
	Text string `json:"text"`
}

type batchEmbedResponse struct {
	Embeddings []struct {
		Values []float64 `json:"values"`
	} `json:"embeddings"`
}

// embedBatch はtexts（最大maxBatchSize件）を1回のHTTPリクエストでベクトル化する。
func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	modelPath := "models/" + c.model

	reqBody := batchEmbedRequest{Requests: make([]embedContentRequest, len(texts))}
	for i, text := range texts {
		reqBody.Requests[i] = embedContentRequest{
			Model:                modelPath,
			Content:              content{Parts: []part{{Text: text}}},
			TaskType:             "RETRIEVAL_DOCUMENT",
			OutputDimensionality: c.dimension,
		}
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/%s:batchEmbedContents", c.baseURL, modelPath)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", c.apiKey)

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
		return nil, fmt.Errorf("gemini api status %d: %s", resp.StatusCode, string(respBody))
	}

	var parsed batchEmbedResponse
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
	for i, e := range parsed.Embeddings {
		normalized := L2Normalize(e.Values)
		vec := make([]float32, len(normalized))
		for j, v := range normalized {
			vec[j] = float32(v)
		}
		out[i] = vec
	}
	return out, nil
}
