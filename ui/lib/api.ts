import type { Collection, DocumentStatus, UploadResponse } from "./types";

export class ApiRequestError extends Error {
  status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

function authHeaders(apiKey: string): HeadersInit {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

export async function listCollections(apiUrl: string, apiKey: string): Promise<Collection[]> {
  const res = await fetch(`${apiUrl}/v1/collections`, {
    headers: authHeaders(apiKey),
  });
  if (!res.ok) throw new ApiRequestError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function createCollection(
  apiUrl: string,
  apiKey: string,
  name: string,
  description: string
): Promise<Collection> {
  const res = await fetch(`${apiUrl}/v1/collections`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(apiKey) },
    body: JSON.stringify({ name, description: description || undefined }),
  });
  if (!res.ok) throw new ApiRequestError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function uploadDocument(
  apiUrl: string,
  apiKey: string,
  collectionId: string,
  file: File
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${apiUrl}/v1/collections/${collectionId}/documents`, {
    method: "POST",
    headers: authHeaders(apiKey),
    body: form,
  });
  if (!res.ok) throw new ApiRequestError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function getDocumentStatus(
  apiUrl: string,
  apiKey: string,
  documentId: string
): Promise<DocumentStatus> {
  const res = await fetch(`${apiUrl}/v1/documents/${documentId}`, {
    headers: authHeaders(apiKey),
  });
  if (!res.ok) throw new ApiRequestError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function openQueryStream(
  apiUrl: string,
  apiKey: string,
  collectionId: string,
  question: string,
  topK: number,
  signal?: AbortSignal
): Promise<Response> {
  const res = await fetch(`${apiUrl}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(apiKey) },
    body: JSON.stringify({
      collection_id: collectionId,
      question,
      top_k: topK,
      stream: true,
    }),
    signal,
  });
  if (!res.ok || !res.body) throw new ApiRequestError(res.status, await parseErrorDetail(res));
  return res;
}
