package embedding

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	c := NewClient("test-api-key", "gemini-embedding-001", 3)
	c.httpClient = server.Client()
	c.baseURL = server.URL
	return c
}

// レスポンスJSONのパース（values → L2正規化済みの[]float32）を確認する。
func TestEmbedParsesResponse(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-goog-api-key"); got != "test-api-key" {
			t.Errorf("x-goog-api-key header = %q, want test-api-key", got)
		}

		var req batchEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if len(req.Requests) != 1 {
			t.Fatalf("requests count = %d, want 1", len(req.Requests))
		}
		if req.Requests[0].TaskType != "RETRIEVAL_DOCUMENT" {
			t.Errorf("taskType = %q, want RETRIEVAL_DOCUMENT", req.Requests[0].TaskType)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[{"values":[3,4,0]}]}`))
	})

	got, err := client.Embed(t.Context(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("Embed() = %v, want 1 vector of length 3", got)
	}
	want := []float32{0.6, 0.8, 0}
	for i := range want {
		if diff := got[0][i] - want[i]; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("Embed()[0][%d] = %v, want %v", i, got[0][i], want[i])
		}
	}
}

// 100件超のテキストが複数リクエストへ分割されることを確認する。
func TestEmbedSplitsBatchesOver100(t *testing.T) {
	var requestSizes []int

	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		var req batchEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		requestSizes = append(requestSizes, len(req.Requests))

		embeddings := make([]struct {
			Values []float64 `json:"values"`
		}, len(req.Requests))
		for i := range embeddings {
			embeddings[i].Values = []float64{1, 0, 0}
		}
		resp := batchEmbedResponse{Embeddings: embeddings}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	texts := make([]string, 250)
	for i := range texts {
		texts[i] = "text"
	}

	got, err := client.Embed(t.Context(), texts)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(got) != 250 {
		t.Fatalf("Embed() returned %d vectors, want 250", len(got))
	}

	wantSizes := []int{100, 100, 50}
	if len(requestSizes) != len(wantSizes) {
		t.Fatalf("request count = %d, want %d (sizes: %v)", len(requestSizes), len(wantSizes), requestSizes)
	}
	for i, want := range wantSizes {
		if requestSizes[i] != want {
			t.Errorf("request[%d] size = %d, want %d", i, requestSizes[i], want)
		}
	}
}

func TestEmbedEmptyInput(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("HTTP request should not be made for empty input")
	})

	got, err := client.Embed(t.Context(), nil)
	if err != nil {
		t.Fatalf("Embed(nil) error = %v", err)
	}
	if got != nil {
		t.Fatalf("Embed(nil) = %v, want nil", got)
	}
}

func TestEmbedNonOKStatus(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	})

	if _, err := client.Embed(t.Context(), []string{"hello"}); err == nil {
		t.Fatal("Embed() expected error for non-200 status, got nil")
	}
}

func TestEmbedResponseSizeMismatch(t *testing.T) {
	client := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	})

	if _, err := client.Embed(t.Context(), []string{"hello"}); err == nil {
		t.Fatal("Embed() expected error for embeddings/texts size mismatch, got nil")
	}
}
