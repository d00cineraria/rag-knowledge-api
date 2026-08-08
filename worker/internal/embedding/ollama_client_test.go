package embedding

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestOllamaClient(t *testing.T, handler http.HandlerFunc) *OllamaClient {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	c := NewOllamaClient(server.URL, "nomic-embed-text")
	c.httpClient = server.Client()
	return c
}

// リクエストJSON形式と"search_document: "プレフィックスの前置を確認する。
func TestOllamaEmbedRequestFormat(t *testing.T) {
	client := newTestOllamaClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %q, want /api/embed", r.URL.Path)
		}

		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if req.Model != "nomic-embed-text" {
			t.Errorf("model = %q, want nomic-embed-text", req.Model)
		}
		if len(req.Input) != 1 || req.Input[0] != "search_document: hello" {
			t.Errorf("input = %v, want [\"search_document: hello\"]", req.Input)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[3,4,0]]}`))
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
func TestOllamaEmbedSplitsBatchesOver100(t *testing.T) {
	var requestSizes []int

	client := newTestOllamaClient(t, func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		requestSizes = append(requestSizes, len(req.Input))

		embeddings := make([][]float64, len(req.Input))
		for i := range embeddings {
			embeddings[i] = []float64{1, 0, 0}
		}
		resp := ollamaEmbedResponse{Embeddings: embeddings}
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

func TestOllamaEmbedEmptyInput(t *testing.T) {
	client := newTestOllamaClient(t, func(w http.ResponseWriter, r *http.Request) {
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

func TestOllamaEmbedNonOKStatus(t *testing.T) {
	client := newTestOllamaClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"model not found"}`))
	})

	if _, err := client.Embed(t.Context(), []string{"hello"}); err == nil {
		t.Fatal("Embed() expected error for non-200 status, got nil")
	}
}

func TestOllamaEmbedResponseSizeMismatch(t *testing.T) {
	client := newTestOllamaClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	})

	if _, err := client.Embed(t.Context(), []string{"hello"}); err == nil {
		t.Fatal("Embed() expected error for embeddings/texts size mismatch, got nil")
	}
}
