export type Collection = {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
};

export type DocumentStatus = {
  id: string;
  collection_id: string;
  filename: string;
  status: "pending" | "ready" | "error" | string;
  error?: string | null;
};

export type UploadResponse = {
  document_id: string;
  status: string;
};

export type Source = {
  index: number;
  chunk_id: string;
  filename: string;
  heading_path: string[];
  content: string;
  score: number;
};

export type QueryResponse = {
  answer: string;
  sources: Source[];
};

export type LatencyMs = {
  retrieval: number;
  generation: number;
};

export type ApiError = {
  status: number;
  detail: string;
};
